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

# Resident store width: only 4-bit codes are packed into the resident int4 layout. The
# AutoGPTQ unpack/dequant helpers below are bit-width generic (bits in {4, 8}) because GLM-5.1
# "mixed" checkpoints quantize routed experts to int4 but attention + shared experts to int8;
# the int8 tensors are dequantized to bf16 (never repacked), so only the int4 path reaches
# ``transcode_autogptq_to_resident``.
_INT4_BITS = 4


def _pack_factor(bits: int) -> int:
    """Number of ``bits``-wide codes packed per int32 word (8 for int4, 4 for int8)."""
    assert 32 % bits == 0, f"bits {bits} must divide 32"
    return 32 // bits


def _sym_zero(bits: int) -> int:
    """Symmetric zero-point: an unsigned code ``q`` maps to signed ``q - 2**(bits-1)``."""
    return 1 << (bits - 1)


def unpack_autogptq_qweight(qweight: torch.Tensor, bits: int = _INT4_BITS) -> torch.Tensor:
    """Unpack AutoGPTQ ``qweight`` ``[in // pack, out]`` to unsigned codes ``[in, out]``.

    ``pack = 32 // bits`` codes share one int32 word along the input dim; codes lie in
    ``[0, 2**bits - 1]`` (``[0, 15]`` for int4, ``[0, 255]`` for int8).
    """
    pack = _pack_factor(bits)
    mask = (1 << bits) - 1
    shifts = (torch.arange(pack, device=qweight.device, dtype=torch.int32) * bits).view(1, pack, 1)
    codes = (qweight.unsqueeze(1) >> shifts) & mask  # [in // pack, pack, out]
    return codes.reshape(qweight.shape[0] * pack, qweight.shape[1])  # [in, out]


def unpack_autogptq_qzeros(qzeros: torch.Tensor, bits: int = _INT4_BITS) -> torch.Tensor:
    """Unpack AutoGPTQ ``qzeros`` ``[G, out // pack]`` to zero-points ``[G, out]`` (``pack = 32 // bits``)."""
    pack = _pack_factor(bits)
    mask = (1 << bits) - 1
    shifts = (torch.arange(pack, device=qzeros.device, dtype=torch.int32) * bits).view(1, 1, pack)
    zeros = (qzeros.unsqueeze(-1) >> shifts) & mask  # [G, out // pack, pack]
    return zeros.reshape(qzeros.shape[0], qzeros.shape[1] * pack)  # [G, out]


def infer_autogptq_bits(qweight: torch.Tensor, scales: torch.Tensor, group_size: int = INT4_GROUP_SIZE) -> int:
    """Infer the quantization bit-width of an AutoGPTQ linear from its tensor shapes.

    ``scales`` is ``[in // group_size, out]`` so ``in = scales.shape[0] * group_size``; ``qweight``
    is ``[in // pack, out]`` so ``pack = in // qweight.shape[0]`` and ``bits = 32 // pack``. Lets a
    mixed-bit checkpoint be classified per tensor without consulting the model config.
    """
    in_features = scales.shape[0] * group_size
    pack = in_features // qweight.shape[0]
    bits = 32 // pack
    assert bits in (4, 8), (
        f"unsupported AutoGPTQ bit-width {bits} (qweight {tuple(qweight.shape)}, scales {tuple(scales.shape)}, group_size {group_size})"
    )
    return bits


def _resolve_zeros(
    qzeros: torch.Tensor | None,
    num_groups: int,
    out_features: int,
    device,
    zero_point_plus_one: bool,
    bits: int = _INT4_BITS,
) -> torch.Tensor:
    """Per-(group, column) integer zero-points ``[G, out]`` (constant ``2**(bits-1)`` when symmetric/absent)."""
    if qzeros is None:
        return torch.full((num_groups, out_features), _sym_zero(bits), dtype=torch.int32, device=device)
    zeros = unpack_autogptq_qzeros(qzeros, bits).to(torch.int32)
    # Legacy AutoGPTQ stored (true_zero - 1); gptq_v2 / auto-round store the true zero.
    if zero_point_plus_one:
        zeros = zeros + 1
    return zeros


def _trivial_g_idx(in_features: int, group_size: int, device) -> torch.Tensor:
    return torch.arange(in_features, device=device, dtype=torch.int32) // group_size


def dequantize_autogptq(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    group_size: int = INT4_GROUP_SIZE,
    zero_point_plus_one: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    bits: int | None = None,
) -> torch.Tensor:
    """Dequantize an AutoGPTQ int4/int8 linear to a dense weight ``[out, in]`` (general path).

    Handles both bit-widths (``bits`` inferred from shapes when ``None``), asymmetric zero-points,
    and ``desc_act`` (non-contiguous ``g_idx``). Used both to dequantize the int8 attention /
    shared-expert tensors of a GLM-5.1 mixed checkpoint to bf16 and as the numerical cross-check
    against the int4 resident transcode. Returns HF/native linear orientation ``[out, in]``.
    """
    if bits is None:
        bits = infer_autogptq_bits(qweight, scales, group_size)
    codes = unpack_autogptq_qweight(qweight, bits)  # [in, out] unsigned
    in_features, out_features = codes.shape
    num_groups = scales.shape[0]
    if g_idx is None:
        g_idx = _trivial_g_idx(in_features, group_size, codes.device)
    g_idx = g_idx.long()
    zeros = _resolve_zeros(qzeros, num_groups, out_features, codes.device, zero_point_plus_one, bits)

    row_scale = scales.to(torch.float32)[g_idx]  # [in, out]
    row_zero = zeros[g_idx]  # [in, out]
    w = (codes.to(torch.float32) - row_zero.to(torch.float32)) * row_scale  # [in, out]
    return w.t().contiguous().to(dtype)  # [out, in]


def dequantize_autogptq_int4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    group_size: int = INT4_GROUP_SIZE,
    zero_point_plus_one: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Int4-fixed wrapper around :func:`dequantize_autogptq` (kept for the original 4-bit callers)."""
    return dequantize_autogptq(qweight, scales, qzeros, g_idx, group_size, zero_point_plus_one, dtype, bits=_INT4_BITS)


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

    sym_zero = _sym_zero(_INT4_BITS)
    zeros = _resolve_zeros(qzeros, scales.shape[0], out_features, codes.device, zero_point_plus_one, _INT4_BITS)
    if not torch.all(zeros == sym_zero):
        raise NotImplementedError(
            "asymmetric int4 (zero-point != 8) cannot be stored in the symmetric resident format; "
            "expected a symmetric (sym=True) AutoGPTQ/auto-round checkpoint."
        )

    signed = (codes.to(torch.int32) - sym_zero).t().contiguous()  # [out, in] in [-8, 7]
    packed = pack_int4(signed)  # [out, in // 8]
    scales_t = scales.t().contiguous().to(torch.bfloat16)  # [out, in // group_size]
    return packed, scales_t
