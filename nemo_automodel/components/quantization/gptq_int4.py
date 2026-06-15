# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Read AutoGPTQ / auto-round int4 (W4A16) tensors and transcode them to the resident layout.

Published GLM-5.1 int4 checkpoints (e.g. ``INC4AI/GLM-5.1-int4-mixed-AutoRound``, packing
format ``auto_round:auto_gptq``) store each quantized linear as:

  - ``qweight``: int32 ``[in_features // 8, out_features]`` -- eight unsigned 4-bit codes
    (``q in [0, 15]``) packed per word along the **input** dim; code for input row ``r`` is
    at bits ``[4*(r % 8), ...)`` of word ``r // 8``.
  - ``scales``:  ``[in_features // group_size, out_features]`` float (one per group, per column).
  - ``qzeros``:  int32 ``[in_features // group_size, out_features // 8]`` -- eight zero-points
    packed per word along the **output** dim. Absent (or constant 8) for symmetric quant.
  - ``g_idx``:   optional int32 ``[in_features]`` -- per-row group index (non-contiguous only
    under activation-ordering / ``desc_act``).

Dequant is ``w[r, c] = (q[r, c] - zero[g_idx[r], c]) * scale[g_idx[r], c]``.

``transcode_autogptq_to_resident`` rewrites this losslessly (no re-rounding -- AutoRound's
learned codes are preserved) into this repo's resident int4 layout
(``components/quantization/int4.py``: signed codes packed 8-per-int32 along the contraction
dim, per-group scale, orientation ``[out, in]``). It is only valid for **symmetric** quant
(zero == 8) with **contiguous** groups; asymmetric zero-points or ``desc_act`` cannot be
represented by the symmetric resident store and raise rather than silently corrupt weights.
"""

import torch

from nemo_automodel.components.quantization.int4 import INT4_GROUP_SIZE, pack_int4

# Symmetric int4 zero-point: q in [0, 15] maps to signed code q - 8 in [-8, 7].
_SYM_ZERO = 8
_BITS = 4
_PACK_FACTOR = 32 // _BITS  # 8 codes per int32


def unpack_autogptq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack AutoGPTQ ``qweight`` ``[in // 8, out]`` to unsigned codes ``[in, out]`` in ``[0, 15]``."""
    shifts = (torch.arange(_PACK_FACTOR, device=qweight.device, dtype=torch.int32) * _BITS).view(1, _PACK_FACTOR, 1)
    codes = (qweight.unsqueeze(1) >> shifts) & 0xF  # [in // 8, 8, out]
    return codes.reshape(qweight.shape[0] * _PACK_FACTOR, qweight.shape[1])  # [in, out]


def unpack_autogptq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """Unpack AutoGPTQ ``qzeros`` ``[G, out // 8]`` to zero-points ``[G, out]`` in ``[0, 15]``."""
    shifts = (torch.arange(_PACK_FACTOR, device=qzeros.device, dtype=torch.int32) * _BITS).view(1, 1, _PACK_FACTOR)
    zeros = (qzeros.unsqueeze(-1) >> shifts) & 0xF  # [G, out // 8, 8]
    return zeros.reshape(qzeros.shape[0], qzeros.shape[1] * _PACK_FACTOR)  # [G, out]


def _resolve_zeros(
    qzeros: torch.Tensor | None, num_groups: int, out_features: int, device, zero_point_plus_one: bool
) -> torch.Tensor:
    """Per-(group, column) integer zero-points ``[G, out]`` (constant 8 when symmetric/absent)."""
    if qzeros is None:
        return torch.full((num_groups, out_features), _SYM_ZERO, dtype=torch.int32, device=device)
    zeros = unpack_autogptq_qzeros(qzeros).to(torch.int32)
    # Legacy AutoGPTQ stored (true_zero - 1); gptq_v2 / auto-round store the true zero.
    if zero_point_plus_one:
        zeros = zeros + 1
    return zeros


def _trivial_g_idx(in_features: int, group_size: int, device) -> torch.Tensor:
    return torch.arange(in_features, device=device, dtype=torch.int32) // group_size


def dequantize_autogptq_int4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    group_size: int = INT4_GROUP_SIZE,
    zero_point_plus_one: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an AutoGPTQ int4 linear to a dense weight ``[out, in]`` (general path).

    Handles asymmetric zero-points and ``desc_act`` (non-contiguous ``g_idx``); used for the
    numerical cross-check against the resident transcode. Returns HF/native linear orientation
    ``[out_features, in_features]``.
    """
    codes = unpack_autogptq_qweight(qweight)  # [in, out] unsigned
    in_features, out_features = codes.shape
    num_groups = scales.shape[0]
    if g_idx is None:
        g_idx = _trivial_g_idx(in_features, group_size, codes.device)
    g_idx = g_idx.long()
    zeros = _resolve_zeros(qzeros, num_groups, out_features, codes.device, zero_point_plus_one)

    row_scale = scales.to(torch.float32)[g_idx]  # [in, out]
    row_zero = zeros[g_idx]  # [in, out]
    w = (codes.to(torch.float32) - row_zero.to(torch.float32)) * row_scale  # [in, out]
    return w.t().contiguous().to(dtype)  # [out, in]


def transcode_autogptq_to_resident(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    group_size: int = INT4_GROUP_SIZE,
    zero_point_plus_one: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Losslessly transcode an AutoGPTQ int4 linear into the resident packed layout.

    Preserves the stored codes exactly (no re-rounding). Returns ``(packed, scales)`` in
    resident orientation ``[out, in]``: ``packed`` int32 ``[out, in // 8]`` (eight signed codes
    per word along the contraction dim), ``scales`` bf16 ``[out, in // group_size]`` -- the
    layout ``dequantize_int4`` consumes.

    Raises:
        NotImplementedError: if the source is asymmetric (any zero-point != 8) or uses
            ``desc_act`` (non-contiguous ``g_idx``); neither is representable by the symmetric
            resident store, so we refuse rather than silently corrupt the weights.
    """
    codes = unpack_autogptq_qweight(qweight)  # [in, out] unsigned
    in_features, out_features = codes.shape
    assert in_features % group_size == 0, f"in_features {in_features} not divisible by group_size {group_size}"

    if g_idx is not None:
        expected = _trivial_g_idx(in_features, group_size, g_idx.device)
        if not torch.equal(g_idx.to(torch.int32), expected):
            raise NotImplementedError(
                "desc_act / non-contiguous g_idx is not supported for resident int4 transcode "
                "(the symmetric resident store assumes contiguous per-group scales)."
            )

    zeros = _resolve_zeros(qzeros, scales.shape[0], out_features, codes.device, zero_point_plus_one)
    if not torch.all(zeros == _SYM_ZERO):
        raise NotImplementedError(
            "asymmetric int4 (zero-point != 8) cannot be stored in the symmetric resident format; "
            "expected a symmetric (sym=True) AutoGPTQ/auto-round checkpoint."
        )

    signed = (codes.to(torch.int32) - _SYM_ZERO).t().contiguous()  # [out, in] in [-8, 7]
    packed = pack_int4(signed)  # [out, in // 8]
    scales_t = scales.t().contiguous().to(torch.bfloat16)  # [out, in // group_size]
    return packed, scales_t
