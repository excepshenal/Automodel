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

"""INT4-resident expert storage for frozen MoE experts (GLM-5.1 and similar).

``GroupedExpertsInt4`` keeps the frozen routed-expert base weights packed as symmetric
W4A16 int4 (group-128, int32-packed + per-group scales) and dequantizes on the fly inside
the grouped GEMM, instead of holding them in bf16. This is the int4 analog of
``GroupedExpertsMXFP4``; the module wiring is identical and only the codec primitives differ
(int4 group-128 vs fp4 e2m1 block-32).

The resident layout is this repo's codec layout (``components/quantization/int4.py``): eight
signed int4 codes per ``int32`` along the contraction dim, one float scale per 128 columns.
Both the round-to-nearest path (``quantize_int4`` from the bf16 base) and the lossless
transcode of an externally-quantized AutoGPTQ/auto-round checkpoint land in this same layout,
so a single ``Int4GroupedMM`` serves every source.
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _apply_bias,
    _permute_tokens_for_grouped_mm,
)
from nemo_automodel.components.quantization.int4 import (
    INT4_GROUP_SIZE,
    Int4GroupedMM,
    dequantize_int4,
    quantize_int4,
)

# Eight int4 codes per packed int32 word (see components/quantization/int4.py).
_INT4_PACK_FACTOR = 8


def _to_local(t):
    """Return the local shard of a DTensor, or the tensor unchanged."""
    return t.to_local() if isinstance(t, DTensor) else t


class Int4ExpertStorageMixin:
    """Packed-int4 base-weight storage and grouped GEMM for routed experts.

    Mixed into a ``GroupedExperts`` (or ``GroupedExpertsLoRA``) subclass. The base
    projections ``gate_and_up_projs`` / ``down_projs`` are stored as int32-packed signed
    int4 (eight codes per word) plus bf16 per-group scales, in checkpoint orientation
    ``[n_experts, out_dim, in_dim]`` so the group scales run along the contraction dim. The
    bf16 parameters are dropped once packed.

    Packing is deferred when the base weights are still on the meta device: the module
    behaves like its bf16 parent until ``pack_base_weights()`` runs (after the checkpoint is
    loaded). Mirrors ``MXFP4ExpertStorageMixin``; group size is fixed at
    ``INT4_GROUP_SIZE`` (128).
    """

    _INT4_BASE_NAMES: tuple[str, ...] = ("gate_and_up_projs", "down_projs")
    # Storage-parameter suffixes, in pack/unpack order.
    _PACKED_SUFFIXES: tuple[str, ...] = ("_packed", "_scales")

    def _init_int4_storage(self) -> None:
        """Validate the backend and pack immediately if base weights are materialized."""
        if not self.use_torch_mm:
            raise NotImplementedError(
                "int4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm'). "
                "The grouped_gemm path (backend.experts='gmm') has no packed variant; with DeepEP dispatch use "
                "backend.dispatcher='deepep' together with backend.experts='torch_mm'."
            )
        self._int4_resident = False
        if not _to_local(getattr(self, self._INT4_BASE_NAMES[0])).is_meta:
            self.pack_base_weights()

    @torch.no_grad()
    def _init_packed_placeholders(self) -> None:
        """Register meta packed storage params from config shapes (no bf16 weights).

        Used by the passthrough path so a packed int4 checkpoint loads straight into these
        params without ever materializing bf16 experts. Config-driven, so it is shared by the
        torch (``GroupedExpertsInt4``) and DeepEP (``GroupedExpertsDeepEPInt4``) frozen
        variants.
        """
        cfg = self.config
        group = INT4_GROUP_SIZE
        up_proj_dim = cfg.moe_inter_dim * 2 if self.is_gated else cfg.moe_inter_dim
        expert_dim = cfg.expert_dim
        moe_inter = cfg.moe_inter_dim
        e = cfg.n_routed_experts
        assert expert_dim % group == 0 and moe_inter % group == 0, (
            f"expert dims must be divisible by {group} for int4 (expert_dim={expert_dim}, moe_inter={moe_inter})"
        )
        # Checkpoint orientation [E, out, in], packed along the contraction (in) dim: eight
        # int4 codes per int32 word, one bf16 scale per group of `group` columns.
        shapes = {
            "gate_and_up_projs": (
                (e, up_proj_dim, expert_dim // _INT4_PACK_FACTOR),
                (e, up_proj_dim, expert_dim // group),
            ),
            "down_projs": (
                (e, expert_dim, moe_inter // _INT4_PACK_FACTOR),
                (e, expert_dim, moe_inter // group),
            ),
        }
        for name, (packed_shape, scale_shape) in shapes.items():
            packed = torch.empty(packed_shape, dtype=torch.int32, device="meta")
            scales = torch.empty(scale_shape, dtype=torch.bfloat16, device="meta")
            self.register_packed_base_weight(name, (packed, scales))
        self._int4_resident = True

    @torch.no_grad()
    def register_packed_base_weight(self, name: str, tensors: tuple[torch.Tensor, ...], reference=None) -> None:
        """Register packed storage params for base projection ``name``.

        Decoupled from quantization so it can run either as a post-load conversion
        (``pack_base_weights`` passes freshly quantized tensors) or at module init (a
        chunk-loader passes meta placeholders, then loads the quantized checkpoint straight
        into them). Replaces the bf16 parameter ``name`` if present.

        Args:
            name: Base projection name (e.g. ``"gate_and_up_projs"``).
            tensors: Storage tensors in ``_PACKED_SUFFIXES`` order.
            reference: Optional DTensor whose mesh/placements the storage tensors inherit
                (use the pre-pack bf16 param, or a meta DTensor at init).
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
        """Round-to-nearest pack the frozen base projections to int4 and free the bf16 tensors.

        No-op when already packed. Requires the base weights to be materialized. This is the
        RTN path (from a bf16 base); an externally-quantized checkpoint instead loads directly
        into the placeholders registered by ``_init_packed_placeholders``.
        """
        if self._int4_resident:
            return
        for name in self._INT4_BASE_NAMES:
            param = getattr(self, name)
            local = _to_local(param)
            assert not local.is_meta, f"pack_base_weights requires materialized '{name}'"
            # [E, in, out] (compute layout) -> [E, out, in] (checkpoint layout) so the int4
            # group scales run along the contraction dim.
            packed, scales = quantize_int4(local.transpose(-2, -1).contiguous(), INT4_GROUP_SIZE)
            self.register_packed_base_weight(name, (packed, scales.to(torch.bfloat16)), reference=param)
        self._int4_resident = True

    def _int4_base_mm(self, x: torch.Tensor, name: str, offs: torch.Tensor) -> torch.Tensor:
        """Grouped GEMM ``x @ W`` over the packed base weight ``name`` (dequant on the fly)."""
        packed = _to_local(getattr(self, name + "_packed"))
        scales = _to_local(getattr(self, name + "_scales"))
        return Int4GroupedMM.apply(x, packed, scales, offs, INT4_GROUP_SIZE)

    def _int4_dequant_expert0(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize expert 0 of base weight ``name`` to compute layout ``[in, out]``."""
        packed = _to_local(getattr(self, name + "_packed"))[0]
        scales = _to_local(getattr(self, name + "_scales"))[0]
        return dequantize_int4(packed, scales, dtype, INT4_GROUP_SIZE).transpose(-2, -1)


class GroupedExpertsInt4(Int4ExpertStorageMixin, GroupedExperts):
    """Frozen routed experts with int4-resident base weights and no adapter.

    Drop-in replacement for ``GroupedExperts`` when the experts are frozen (e.g. LoRA
    training that targets only attention). Forward mirrors ``GroupedExpertsMXFP4`` over the
    int4 codec.
    """

    def __init__(self, orig_module: GroupedExperts, passthrough: bool = False):
        """
        Args:
            orig_module: The bf16 GroupedExperts to replace.
            passthrough: When True, register packed storage placeholders at init (no bf16
                weights) so a quantized checkpoint loads straight into them — experts are
                never materialized in bf16. Requires the base weights to be meta.
        """
        super().__init__(orig_module.config, backend=None)
        if not self.use_torch_mm and not orig_module.use_torch_mm:
            raise NotImplementedError(
                "int4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm')."
            )
        self.use_torch_mm = orig_module.use_torch_mm

        if passthrough:
            # The bf16 base params from super().__init__ are placeholders only (meta under
            # init_empty_weights); _init_packed_placeholders deletes them and registers meta
            # packed storage, so no bf16 experts are ever materialized.
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
        self._init_int4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward over int4 base weights. Falls back to bf16 until packing is done."""
        if not self._int4_resident:
            return super().forward(x, token_mask, weights, indices)

        assert not isinstance(x, DTensor)
        input_dtype = x.dtype

        if isinstance(self.gate_and_up_projs_packed, DTensor):
            ep_mesh = self.gate_and_up_projs_packed.device_mesh
            assert ep_mesh is not None and ep_mesh.ndim == 1, "We only support 1D mesh for MoE"
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        assert self.n_routed_experts % ep_size == 0

        if ep_size > 1:
            from torch.distributed.tensor import Partial, Shard

            x = DTensor.from_local(x, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor(
                grad_placements=[Partial()]
            )
            weights = DTensor.from_local(weights.float(), device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor(
                grad_placements=[Partial()]
            )
            indices = DTensor.from_local(indices, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            token_mask = DTensor.from_local(token_mask, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts

        y = self._forward_grouped_mm_int4(x, token_mask, weights, indices, n_local_experts, experts_start_idx)

        if ep_size > 1:
            from torch.distributed.tensor import Partial, Shard

            y = DTensor.from_local(y, device_mesh=ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()

        return y.to(input_dtype)

    def _forward_grouped_mm_int4(self, x, token_mask, weights, indices, n_local_experts, experts_start_idx):
        sorted_token_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices, weights, token_mask, n_local_experts, experts_start_idx
        )
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if tokens_per_expert.sum() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)

            output1 = self._int4_base_mm(permuted_x, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation_grouped(output1, permuted_probs)

            output2 = self._int4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)

            scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
            y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._int4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._int4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation_grouped(output1, weights[0, 0, None].unsqueeze(0))
            output2 = torch.matmul(output1_, down_w0)
            y[0] += output2[0]

        return y


class GroupedExpertsDeepEPInt4(Int4ExpertStorageMixin, GroupedExpertsDeepEP):
    """Frozen routed experts with int4-resident base weights under DeepEP dispatch.

    Drop-in replacement for ``GroupedExpertsDeepEP`` when the experts are frozen. The DeepEP
    fused all-to-all token dispatch is reused unchanged — int4 only changes the two
    post-dispatch grouped GEMMs, which read the packed base weights via ``Int4GroupedMM``
    instead of bf16 ``torch._grouped_mm``.

    Requires the torch_mm experts backend (``backend.experts='torch_mm'``); the grouped_gemm
    (``gmm``) path has no packed variant.
    """

    def __init__(self, orig_module: GroupedExpertsDeepEP, passthrough: bool = False):
        """
        Args:
            orig_module: The bf16 GroupedExpertsDeepEP to replace.
            passthrough: When True, register packed storage placeholders at init (no bf16
                weights) so a quantized checkpoint loads straight into them.
        """
        super().__init__(
            orig_module.config,
            backend=None,
            dispatcher_backend=orig_module.dispatcher_backend,
            dispatcher_num_sms=orig_module.dispatcher_num_sms,
            dispatcher_share_token_dispatcher=orig_module.dispatcher_share_token_dispatcher,
            dispatcher_async_dispatch=orig_module.dispatcher_async_dispatch,
        )
        # backend=None leaves use_torch_mm False; inherit the original's choice so the int4
        # storage guard enforces torch_mm (set before _init_int4_storage runs).
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
        self._init_int4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward over int4 base weights with DeepEP dispatch.

        Mirrors ``GroupedExpertsDeepEP.forward``, replacing the two base ``torch._grouped_mm``
        calls with ``Int4GroupedMM`` over the packed weights. Falls back to the bf16 parent
        while packing is still deferred.
        """
        if not self._int4_resident:
            return super().forward(x, token_mask, weights, indices)

        assert not isinstance(x, DTensor)
        assert self.use_torch_mm, "int4-resident DeepEP experts require the torch_mm experts backend."
        assert self.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )

        indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)
        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        if torch.count_nonzero(tokens_per_expert) > 0:
            tokens_per_expert_gpu = tokens_per_expert.to(device=permuted_local_hidden_states.device, non_blocking=True)
            offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

            output1 = self._int4_base_mm(permuted_local_hidden_states, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation(output1, permuted_probs)
            output2 = self._int4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._int4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._int4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_w0)

        y = self.token_dispatcher.token_unpermutation(output2)
        return y
