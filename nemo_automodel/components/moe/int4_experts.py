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

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _PackedExpertStorageBase,
    _PackedGroupedExpertsDeepEPForward,
    _PackedGroupedExpertsForward,
    _to_local,
)
from nemo_automodel.components.quantization.int4 import (
    INT4_GROUP_SIZE,
    Int4GroupedMM,
    dequantize_int4,
    quantize_int4,
)

# Eight int4 codes per packed int32 word (see components/quantization/int4.py).
_INT4_PACK_FACTOR = 8


class Int4ExpertStorageMixin(_PackedExpertStorageBase):
    """Packed-int4 (W4A16) base-weight storage and grouped GEMM for routed experts.

    Mixed into a ``GroupedExperts`` (or ``GroupedExpertsLoRA``) subclass. The base projections are
    stored as int32-packed signed int4 (eight codes per word, group-128) plus bf16 per-group
    scales; the shared layout, registration and (de)packing scaffolding lives in
    ``_PackedExpertStorageBase``. Only the int4 codec primitives are here. Both the round-to-nearest
    path (``_quantize_base`` from a bf16 base) and the lossless transcode of an externally-quantized
    AutoGPTQ/auto-round checkpoint land in the same layout, so one ``Int4GroupedMM`` serves both.
    """

    _codec_label = "int4"
    _packed_dtype = torch.int32
    _scale_dtype = torch.bfloat16
    _packed_codes_per_word = _INT4_PACK_FACTOR  # eight int4 codes per int32 word
    _packed_block = INT4_GROUP_SIZE

    @torch.no_grad()
    def _quantize_base(self, local_transposed: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Round-to-nearest quantize a materialized base (checkpoint layout) to int4."""
        packed, scales = quantize_int4(local_transposed, INT4_GROUP_SIZE)
        return packed, scales.to(torch.bfloat16)

    def _base_mm(self, x: torch.Tensor, name: str, offs: torch.Tensor) -> torch.Tensor:
        """Grouped GEMM ``x @ W`` over the packed base weight ``name`` (dequant on the fly)."""
        packed = _to_local(getattr(self, name + "_packed"))
        scales = _to_local(getattr(self, name + "_scales"))
        return Int4GroupedMM.apply(x, packed, scales, offs, INT4_GROUP_SIZE)

    def _dequant_expert0(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize expert 0 of base weight ``name`` to compute layout ``[in, out]``."""
        packed = _to_local(getattr(self, name + "_packed"))[0]
        scales = _to_local(getattr(self, name + "_scales"))[0]
        return dequantize_int4(packed, scales, dtype, INT4_GROUP_SIZE).transpose(-2, -1)


class GroupedExpertsInt4(Int4ExpertStorageMixin, _PackedGroupedExpertsForward, GroupedExperts):
    """Frozen routed experts with int4-resident base weights and no adapter.

    Drop-in replacement for ``GroupedExperts`` when the experts are frozen (e.g. LoRA
    training that targets only attention). The codec-agnostic forward lives in
    ``_PackedGroupedExpertsForward`` (shared with ``GroupedExpertsMXFP4``); only the int4
    storage and ``__init__`` are here.
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
        self._init_packed_storage()


class GroupedExpertsDeepEPInt4(Int4ExpertStorageMixin, _PackedGroupedExpertsDeepEPForward, GroupedExpertsDeepEP):
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
        # storage guard enforces torch_mm (set before _init_packed_storage runs).
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
        self._init_packed_storage()
