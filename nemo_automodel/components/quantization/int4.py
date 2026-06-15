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

"""INT4 (symmetric W4A16, int32-packed + per-group float scales) pack/unpack utilities for MoE expert weights.

The packed layout matches the compressed-tensors / llm-compressor "pack-quantized" int4
format used by W4A16 checkpoints: eight signed 4-bit codes packed little-end-first into
one ``int32`` word along the contraction (last) dimension, with one floating-point scale
per ``group_size`` (default 128) contiguous columns. Quantization is symmetric (no
zero-point): a code ``q in [-8, 7]`` dequantizes to ``q * scale``.

``Int4GroupedMM`` provides a grouped GEMM over packed weights that re-dequantizes in
backward instead of saving the dequantized tensor, so frozen expert weights stay packed
at steady state during LoRA training. This mirrors ``mxfp4.MXFP4GroupedMM``; the only
differences are the group size (128 vs 32), the scale representation (an arbitrary float
per group vs a power-of-two ``float8_e8m0fnu``), and the packing width (eight int4 per
``int32`` vs two e2m1 per ``int8``).
"""

import torch

INT4_GROUP_SIZE = 128

# Signed int4 range for symmetric quantization. The negative-most code (-8) is kept so the
# packing round-trips any externally-produced symmetric int4 checkpoint, but the quantizer
# scales by ``INT4_QMAX`` (7) so that ``+amax`` maps to ``+7`` and the grid stays symmetric.
INT4_QMIN = -8
INT4_QMAX = 7

# Eight 4-bit codes per int32; nibble i occupies bits [4*i, 4*i + 4).
_PACK_FACTOR = 8
_NIBBLE_SHIFTS = torch.arange(_PACK_FACTOR, dtype=torch.int32) * 4


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 codes ``[..., K]`` (K divisible by 8) into int32 ``[..., K // 8]``.

    Little-end first: code ``8 * w + j`` lands in bits ``[4*j, 4*j + 4)`` of word ``w``. This
    is the inverse of ``unpack_int4`` and the single source of truth for the resident bit
    layout, shared by ``quantize_int4`` (round-to-nearest) and the AutoGPTQ transcoder
    (lossless repack of externally-quantized codes).
    """
    k = codes.shape[-1]
    assert k % _PACK_FACTOR == 0, f"last dim {k} must be divisible by {_PACK_FACTOR}"
    shifts = _NIBBLE_SHIFTS.to(codes.device)
    nibbles = (codes.to(torch.int32) & 0xF).view(*codes.shape[:-1], k // _PACK_FACTOR, _PACK_FACTOR)
    return (nibbles << shifts).sum(dim=-1).to(torch.int32).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 ``[..., K // 8]`` into signed int4 codes ``[..., K]`` in ``[-8, 7]`` (int32)."""
    shifts = _NIBBLE_SHIFTS.to(packed.device)
    # Extract each nibble as an unsigned [0, 15] value, then map to signed [-8, 7] via the
    # sign-bit-flip trick: (u ^ 8) - 8 reproduces 4-bit two's complement without a branch.
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF  # [..., K // 8, 8]
    return ((nibbles ^ 0x8) - 0x8).flatten(-2)  # [..., K]


def dequantize_int4(
    packed: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype, group_size: int = INT4_GROUP_SIZE
) -> torch.Tensor:
    """Unpack int32-packed signed int4 codes and apply the per-group symmetric scale.

    Args:
        packed: int32 tensor of shape ``[..., K // 8]`` holding eight signed 4-bit codes
            per word (little-end first: code ``j`` of word ``w`` is column ``8 * w + j``).
        scales: floating-point tensor of shape ``[..., K // group_size]``.
        dtype: Output dtype.
        group_size: Number of contiguous columns sharing one scale (default 128).

    Returns:
        Dequantized tensor of shape ``[..., K]`` in ``dtype``.
    """
    codes = unpack_int4(packed).to(torch.float32)  # [..., K]

    scale_f32 = scales.to(torch.float32)  # [..., K // group_size]
    # Stay blocked and broadcast the per-group scale instead of materializing a full
    # [..., K] scale tensor (cheaper, and fuses better under torch.compile).
    blocked = codes.view(*codes.shape[:-1], scale_f32.shape[-1], group_size)
    return (blocked * scale_f32.unsqueeze(-1)).flatten(-2).to(dtype)


def quantize_int4(weight: torch.Tensor, group_size: int = INT4_GROUP_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest symmetric int4 quantization along the last dim, packed into int32.

    Each ``group_size`` block shares one scale ``amax / 7``; values are rounded to the nearest
    integer code and clamped to ``[-8, 7]``. A weight that already lies on a symmetric int4
    grid (e.g. a dequantized W4A16 checkpoint) round-trips value-exactly.

    Args:
        weight: Floating-point tensor of shape ``[..., K]`` with ``K`` divisible by ``group_size``
            (and therefore by 8).
        group_size: Number of contiguous columns sharing one scale (default 128).

    Returns:
        Tuple of (int32 packed tensor ``[..., K // 8]``, fp32 scales ``[..., K // group_size]``).
    """
    k = weight.shape[-1]
    assert k % group_size == 0, f"last dim {k} must be divisible by group_size {group_size}"
    assert group_size % _PACK_FACTOR == 0, f"group_size {group_size} must be divisible by {_PACK_FACTOR}"

    w = weight.float()
    blocks = w.view(*w.shape[:-1], k // group_size, group_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)  # [..., K // group_size, 1]
    scale = (amax / INT4_QMAX).clamp_min(torch.finfo(torch.float32).tiny)

    codes = torch.round(blocks / scale).clamp_(INT4_QMIN, INT4_QMAX).to(torch.int32)  # [..., K // gs, gs]
    codes = codes.reshape(*w.shape[:-1], k)

    packed = pack_int4(codes)  # [..., K // 8]
    scales = scale.squeeze(-1)  # [..., K // group_size]
    return packed.contiguous(), scales.contiguous()


class Int4GroupedMM(torch.autograd.Function):
    """Grouped GEMM over int4-packed frozen weights with dequantization on the fly.

    Saves only the packed weights for backward and re-dequantizes there, so the bf16 weight
    tensor is a transient in both passes instead of being kept alive by autograd. Weights are
    stored as ``[E, N, K]`` packed along ``K``, which is the natural dequantization output and
    the operand the backward GEMM needs directly (``grad_x = grad_out @ W``). The forward needs
    ``[E, K, N]``, which ``torch._grouped_mm`` consumes as a transposed view (cuBLAS transB) --
    no contiguous copy required.

    No weight gradient is produced -- the base weights are frozen under LoRA. Mirrors
    ``mxfp4.MXFP4GroupedMM`` over the int4 codec.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        offs: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        w_t = dequantize_int4(packed, scales, x.dtype, group_size)  # [E, N, K]
        out = torch._grouped_mm(x, w_t.transpose(-2, -1), offs=offs)
        ctx.save_for_backward(packed, scales, offs)
        ctx.group_size = group_size
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        packed, scales, offs = ctx.saved_tensors
        w_t = dequantize_int4(packed, scales, grad_out.dtype, ctx.group_size)  # [E, N, K] == W^T
        grad_x = torch._grouped_mm(grad_out.contiguous(), w_t, offs=offs)
        return grad_x, None, None, None, None
