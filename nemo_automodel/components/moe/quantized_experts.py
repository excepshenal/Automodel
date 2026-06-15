# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MXFP4-resident expert storage for frozen MoE experts.

``GroupedExpertsMXFP4`` keeps the frozen routed-expert base weights packed as
fp4-e2m1 + e8m0 block scales (the DeepSeek V4 Flash checkpoint format) and
dequantizes on the fly inside the grouped GEMM, instead of holding them in
bf16. This is the storage win for LoRA / frozen-base training of large MoE
models, where the routed experts dominate parameter memory.

The format-specific pack/unpack/GEMM logic lives in ``MXFP4ExpertStorageMixin``
so a future integer-int4 (e.g. GLM) variant can reuse the same module wiring by
swapping the mixin's primitives.
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _PackedGroupedExpertsDeepEPForward,
    _PackedGroupedExpertsForward,
    _to_local,
)
from nemo_automodel.components.quantization.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    MXFP4GroupedMM,
    dequantize_mxfp4,
    quantize_mxfp4,
)


class MXFP4ExpertStorageMixin:
    """Packed-mxfp4 base-weight storage and grouped GEMM for routed experts.

    Mixed into a ``GroupedExperts`` (or ``GroupedExpertsLoRA``) subclass. The base
    projections ``gate_and_up_projs`` / ``down_projs`` are stored as packed fp4
    (int8, two e2m1 nibbles per byte) plus ``float8_e8m0fnu`` block scales, in
    checkpoint orientation ``[n_experts, out_dim, in_dim]`` so the block scales run
    along the contraction dim. The bf16 parameters are dropped once packed.

    Packing is deferred when the base weights are still on the meta device: the
    module behaves like its bf16 parent until ``pack_base_weights()`` runs (after
    the checkpoint is loaded).
    """

    _MXFP4_BASE_NAMES: tuple[str, ...] = ("gate_and_up_projs", "down_projs")
    # Storage-parameter suffixes, in pack/unpack order. Kept as a tuple so the
    # registration helper is format-driven rather than hardcoding two names.
    _PACKED_SUFFIXES: tuple[str, ...] = ("_packed", "_scales")

    def _init_mxfp4_storage(self) -> None:
        """Validate the backend and pack immediately if base weights are materialized."""
        if not self.use_torch_mm:
            raise NotImplementedError(
                "mxfp4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm'). "
                "The grouped_gemm path (backend.experts='gmm') has no packed variant; with DeepEP dispatch use "
                "backend.dispatcher='deepep' together with backend.experts='torch_mm'."
            )
        self._packed_resident = False
        if not _to_local(getattr(self, self._MXFP4_BASE_NAMES[0])).is_meta:
            self.pack_base_weights()

    @torch.no_grad()
    def _init_packed_placeholders(self) -> None:
        """Register meta packed storage params from config shapes (no bf16 weights).

        Used by the passthrough path so a packed fp4 checkpoint loads straight into
        these params without ever materializing bf16 experts. Config-driven, so it is
        shared by the torch (``GroupedExpertsMXFP4``) and DeepEP
        (``GroupedExpertsDeepEPMXFP4``) frozen variants.
        """
        cfg = self.config
        block = MXFP4_BLOCK_SIZE
        up_proj_dim = cfg.moe_inter_dim * 2 if self.is_gated else cfg.moe_inter_dim
        expert_dim = cfg.expert_dim
        moe_inter = cfg.moe_inter_dim
        e = cfg.n_routed_experts
        assert expert_dim % block == 0 and moe_inter % block == 0, (
            f"expert dims must be divisible by {block} for mxfp4 (expert_dim={expert_dim}, moe_inter={moe_inter})"
        )
        # Checkpoint orientation [E, out, in], packed along the contraction (in) dim.
        shapes = {
            "gate_and_up_projs": ((e, up_proj_dim, expert_dim // 2), (e, up_proj_dim, expert_dim // block)),
            "down_projs": ((e, expert_dim, moe_inter // 2), (e, expert_dim, moe_inter // block)),
        }
        for name, (packed_shape, scale_shape) in shapes.items():
            packed = torch.empty(packed_shape, dtype=torch.int8, device="meta")
            scales = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device="meta")
            self.register_packed_base_weight(name, (packed, scales))
        self._packed_resident = True

    @torch.no_grad()
    def register_packed_base_weight(self, name: str, tensors: tuple[torch.Tensor, ...], reference=None) -> None:
        """Register packed storage params for base projection ``name``.

        Decoupled from quantization so it can run either as a post-load conversion
        (``pack_base_weights`` passes freshly quantized tensors) or at module init
        (a chunk-loader passes meta placeholders, then loads the quantized
        checkpoint straight into them — the path that avoids ever materializing
        bf16 experts at GLM-744B scale). Replaces the bf16 parameter ``name`` if
        present.

        Args:
            name: Base projection name (e.g. ``"gate_and_up_projs"``).
            tensors: Storage tensors in ``_PACKED_SUFFIXES`` order.
            reference: Optional DTensor whose mesh/placements the storage tensors
                inherit (use the pre-pack bf16 param, or a meta DTensor at init).
        """
        assert len(tensors) == len(self._PACKED_SUFFIXES), (
            f"expected {len(self._PACKED_SUFFIXES)} tensors {self._PACKED_SUFFIXES}, got {len(tensors)}"
        )
        if isinstance(reference, DTensor):
            tensors = tuple(DTensor.from_local(t, reference.device_mesh, reference.placements) for t in tensors)
        if name in self._parameters:
            del self._parameters[name]
        for suffix, tensor in zip(self._PACKED_SUFFIXES, tensors):
            self.register_parameter(name + suffix, nn.Parameter(tensor, requires_grad=False))

    @torch.no_grad()
    def pack_base_weights(self) -> None:
        """Pack the frozen base projections to mxfp4 and free the bf16 tensors.

        No-op when already packed. Requires the base weights to be materialized.
        """
        if self._packed_resident:
            return
        for name in self._MXFP4_BASE_NAMES:
            param = getattr(self, name)
            local = _to_local(param)
            assert not local.is_meta, f"pack_base_weights requires materialized '{name}'"
            # [E, in, out] (compute layout) -> [E, out, in] (checkpoint layout) so the
            # mx block scales run along the contraction dim.
            tensors = quantize_mxfp4(local.transpose(-2, -1).contiguous())
            self.register_packed_base_weight(name, tensors, reference=param)
        self._packed_resident = True

    def _base_mm(self, x: torch.Tensor, name: str, offs: torch.Tensor) -> torch.Tensor:
        """Grouped GEMM ``x @ W`` over the packed base weight ``name`` (dequant on the fly)."""
        packed = _to_local(getattr(self, name + "_packed"))
        scales = _to_local(getattr(self, name + "_scales"))
        return MXFP4GroupedMM.apply(x, packed, scales, offs)

    def _dequant_expert0(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize expert 0 of base weight ``name`` to compute layout ``[in, out]``."""
        packed = _to_local(getattr(self, name + "_packed"))[0]
        scales = _to_local(getattr(self, name + "_scales"))[0]
        return dequantize_mxfp4(packed, scales, dtype).transpose(-2, -1)


class GroupedExpertsMXFP4(MXFP4ExpertStorageMixin, _PackedGroupedExpertsForward, GroupedExperts):
    """Frozen routed experts with mxfp4-resident base weights and no adapter.

    Drop-in replacement for ``GroupedExperts`` when the experts are frozen (e.g.
    LoRA training that targets only attention). The codec-agnostic forward lives in
    ``_PackedGroupedExpertsForward``; only the mxfp4 storage and ``__init__`` are here.
    """

    def __init__(self, orig_module: GroupedExperts, passthrough: bool = False):
        """
        Args:
            orig_module: The bf16 GroupedExperts to replace.
            passthrough: When True, register packed storage placeholders at init
                (no bf16 weights) so a quantized checkpoint loads straight into
                them — experts are never materialized in bf16. Requires the base
                weights to be meta (i.e. loaded later from a packed checkpoint).
        """
        super().__init__(orig_module.config, backend=None)
        if not self.use_torch_mm and not orig_module.use_torch_mm:
            raise NotImplementedError(
                "mxfp4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm')."
            )
        self.use_torch_mm = orig_module.use_torch_mm

        if passthrough:
            # The bf16 base params from super().__init__ are placeholders only
            # (meta under init_empty_weights); _init_packed_placeholders deletes
            # them and registers meta packed storage, so no bf16 experts are ever
            # materialized — the packed checkpoint loads straight into them.
            if self.expert_bias:
                self.gate_up_proj_bias.requires_grad_(False)
                self.down_proj_bias.requires_grad_(False)
            self._init_packed_placeholders()
            return

        if not getattr(orig_module, "gate_and_up_projs", None).is_meta:
            self.gate_and_up_projs.data = _to_local(orig_module.gate_and_up_projs).clone()
            self.down_projs.data = _to_local(orig_module.down_projs).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.gate_and_up_projs.requires_grad_(False)
        self.down_projs.requires_grad_(False)
        self._init_mxfp4_storage()


class GroupedExpertsDeepEPMXFP4(MXFP4ExpertStorageMixin, _PackedGroupedExpertsDeepEPForward, GroupedExpertsDeepEP):
    """Frozen routed experts with mxfp4-resident base weights under DeepEP dispatch.

    Drop-in replacement for ``GroupedExpertsDeepEP`` when the experts are frozen
    (e.g. LoRA on attention only). The DeepEP fused all-to-all token dispatch is reused
    unchanged — mxfp4 only changes the two post-dispatch grouped GEMMs, which read the
    packed base weights via ``MXFP4GroupedMM`` instead of bf16 ``torch._grouped_mm``.

    Requires the torch_mm experts backend (``backend.experts='torch_mm'``); the
    grouped_gemm (``gmm``) path has no packed variant.
    """

    def __init__(self, orig_module: GroupedExpertsDeepEP, passthrough: bool = False):
        """
        Args:
            orig_module: The bf16 GroupedExpertsDeepEP to replace.
            passthrough: When True, register packed storage placeholders at init (no
                bf16 weights) so a quantized checkpoint loads straight into them.
        """
        super().__init__(
            orig_module.config,
            backend=None,
            dispatcher_backend=orig_module.dispatcher_backend,
            dispatcher_num_sms=orig_module.dispatcher_num_sms,
            dispatcher_share_token_dispatcher=orig_module.dispatcher_share_token_dispatcher,
            dispatcher_async_dispatch=orig_module.dispatcher_async_dispatch,
        )
        # backend=None leaves use_torch_mm False; inherit the original's choice so the
        # mxfp4 storage guard enforces torch_mm (set before _init_mxfp4_storage runs).
        self.use_torch_mm = orig_module.use_torch_mm

        if passthrough:
            if self.expert_bias:
                self.gate_up_proj_bias.requires_grad_(False)
                self.down_proj_bias.requires_grad_(False)
            self._init_packed_placeholders()
            return

        if not _to_local(orig_module.gate_and_up_projs).is_meta:
            self.gate_and_up_projs.data = _to_local(orig_module.gate_and_up_projs).clone()
            self.down_projs.data = _to_local(orig_module.down_projs).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.gate_and_up_projs.requires_grad_(False)
        self.down_projs.requires_grad_(False)
        self._init_mxfp4_storage()
