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

"""CPU tests for the AutoGPTQ -> resident int4 transcoder.

Synthetic AutoGPTQ tensors are built with reference packers (the inverse of the production
unpackers) so correctness is self-verified without the real checkpoint or a GPU.
"""

import pytest
import torch

from nemo_automodel.components.quantization.gptq_int4 import (
    dequantize_autogptq,
    dequantize_autogptq_int4,
    infer_autogptq_bits,
    transcode_autogptq_to_resident,
    unpack_autogptq_qweight,
)
from nemo_automodel.components.quantization.int4 import dequantize_int4


def _ref_pack_qweight_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Reference AutoGPTQ qweight packer: codes [in, out] -> int32 [in // pack, out] (pack=32//bits)."""
    in_f, out_f = codes.shape
    pack = 32 // bits
    mask = (1 << bits) - 1
    codes = codes.to(torch.int32)
    qw = torch.zeros(in_f // pack, out_f, dtype=torch.int32)
    for j in range(pack):
        qw |= (codes[j::pack] & mask) << (j * bits)
    return qw


def _ref_pack_qweight(codes: torch.Tensor) -> torch.Tensor:
    """Reference AutoGPTQ qweight packer: codes [in, out] in [0, 15] -> int32 [in // 8, out]."""
    return _ref_pack_qweight_bits(codes, 4)


def _ref_pack_qzeros(zeros: torch.Tensor) -> torch.Tensor:
    """Reference AutoGPTQ qzeros packer: zeros [G, out] in [0, 15] -> int32 [G, out // 8]."""
    g, out_f = zeros.shape
    zeros = zeros.to(torch.int32)
    qz = torch.zeros(g, out_f // 8, dtype=torch.int32)
    for j in range(8):
        qz |= (zeros[:, j::8] & 0xF) << (j * 4)
    return qz


def test_unpack_qweight_is_inverse_of_pack():
    torch.manual_seed(0)
    in_f, out_f = 64, 16
    codes = torch.randint(0, 16, (in_f, out_f), dtype=torch.int32)
    qweight = _ref_pack_qweight(codes)
    assert qweight.shape == (in_f // 8, out_f)
    recovered = unpack_autogptq_qweight(qweight)
    torch.testing.assert_close(recovered, codes, rtol=0, atol=0)


def test_symmetric_transcode_matches_dequant():
    # A symmetric (zero == 8) AutoGPTQ linear transcoded to resident layout must dequantize to
    # exactly the same weight as the direct AutoGPTQ dequant.
    torch.manual_seed(1)
    in_f, out_f, gs = 32, 16, 8
    codes = torch.randint(0, 16, (in_f, out_f), dtype=torch.int32)  # unsigned q in [0, 15]
    qweight = _ref_pack_qweight(codes)
    scales = (torch.rand(in_f // gs, out_f) + 0.1) * 0.02  # [G, out]

    ref_w = dequantize_autogptq_int4(qweight, scales, qzeros=None, group_size=gs, dtype=torch.float32)  # [out, in]

    packed, res_scales = transcode_autogptq_to_resident(qweight, scales, qzeros=None, group_size=gs)
    assert packed.shape == (out_f, in_f // 8)
    assert res_scales.shape == (out_f, in_f // gs)
    res_w = dequantize_int4(packed, res_scales, torch.float32, group_size=gs)  # [out, in]

    # res_scales is bf16; compare with a bf16-rounded reference scale to isolate the layout.
    ref_w_bf16_scale = dequantize_autogptq_int4(
        qweight, scales.to(torch.bfloat16).float(), qzeros=None, group_size=gs, dtype=torch.float32
    )
    torch.testing.assert_close(res_w, ref_w_bf16_scale, rtol=0, atol=1e-6)
    # And it should track the full-precision-scale reference closely.
    torch.testing.assert_close(res_w, ref_w, rtol=1e-2, atol=1e-2)


def test_transcode_rejects_asymmetric_zeropoint():
    torch.manual_seed(2)
    in_f, out_f, gs = 32, 16, 8
    codes = torch.randint(0, 16, (in_f, out_f), dtype=torch.int32)
    qweight = _ref_pack_qweight(codes)
    scales = torch.rand(in_f // gs, out_f) * 0.02 + 0.01
    zeros = torch.full((in_f // gs, out_f), 7, dtype=torch.int32)  # != 8 -> asymmetric
    qzeros = _ref_pack_qzeros(zeros)
    with pytest.raises(NotImplementedError, match="asymmetric"):
        transcode_autogptq_to_resident(qweight, scales, qzeros=qzeros, group_size=gs)


def test_transcode_rejects_desc_act_g_idx():
    torch.manual_seed(3)
    in_f, out_f, gs = 32, 16, 8
    codes = torch.randint(0, 16, (in_f, out_f), dtype=torch.int32)
    qweight = _ref_pack_qweight(codes)
    scales = torch.rand(in_f // gs, out_f) * 0.02 + 0.01
    g_idx = torch.arange(in_f, dtype=torch.int32).flip(0) // gs  # reversed -> non-contiguous
    with pytest.raises(NotImplementedError, match="desc_act|g_idx"):
        transcode_autogptq_to_resident(qweight, scales, qzeros=None, g_idx=g_idx, group_size=gs)


def test_unpack_qweight_int8_is_inverse_of_pack():
    # int8 packs 4 codes per int32 along the input dim; the unpacker must invert that.
    torch.manual_seed(5)
    in_f, out_f = 32, 12
    codes = torch.randint(0, 256, (in_f, out_f), dtype=torch.int32)
    qweight = _ref_pack_qweight_bits(codes, bits=8)
    assert qweight.shape == (in_f // 4, out_f)
    recovered = unpack_autogptq_qweight(qweight, bits=8)
    torch.testing.assert_close(recovered, codes, rtol=0, atol=0)


def test_infer_bits_from_shapes():
    # The same logical [in, out] linear has differently-shaped qweight for int4 vs int8;
    # group_size anchors in_features so the bit-width is recoverable from shapes alone.
    in_f, out_f, gs = 64, 16, 32
    scales = torch.rand(in_f // gs, out_f)
    qw4 = _ref_pack_qweight_bits(torch.zeros(in_f, out_f, dtype=torch.int32), bits=4)
    qw8 = _ref_pack_qweight_bits(torch.zeros(in_f, out_f, dtype=torch.int32), bits=8)
    assert infer_autogptq_bits(qw4, scales, group_size=gs) == 4
    assert infer_autogptq_bits(qw8, scales, group_size=gs) == 8


def test_int8_dequant_matches_manual_symmetric():
    # Symmetric int8 (zero == 128): dequant must equal (code - 128) * scale, transposed to [out, in].
    torch.manual_seed(6)
    in_f, out_f, gs = 32, 8, 16
    codes = torch.randint(0, 256, (in_f, out_f), dtype=torch.int32)
    qweight = _ref_pack_qweight_bits(codes, bits=8)
    scales = (torch.rand(in_f // gs, out_f) + 0.1) * 0.01  # [G, out]

    w = dequantize_autogptq(qweight, scales, qzeros=None, group_size=gs, dtype=torch.float32)
    assert w.shape == (out_f, in_f)

    g_idx = torch.arange(in_f) // gs
    expected = ((codes.float() - 128.0) * scales[g_idx].float()).t()
    torch.testing.assert_close(w, expected, rtol=0, atol=1e-6)
    # bits is inferred from shapes when not passed.
    torch.testing.assert_close(w, dequantize_autogptq(qweight, scales, group_size=gs, dtype=torch.float32, bits=8))


def test_int4_dequant_wrapper_matches_general():
    torch.manual_seed(7)
    in_f, out_f, gs = 32, 8, 8
    codes = torch.randint(0, 16, (in_f, out_f), dtype=torch.int32)
    qweight = _ref_pack_qweight(codes)
    scales = (torch.rand(in_f // gs, out_f) + 0.1) * 0.02
    a = dequantize_autogptq_int4(qweight, scales, group_size=gs, dtype=torch.float32)
    b = dequantize_autogptq(qweight, scales, group_size=gs, dtype=torch.float32, bits=4)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_constant_eight_qzeros_transcodes_like_symmetric():
    # An explicit all-8 qzeros must behave identically to qzeros=None (symmetric).
    torch.manual_seed(4)
    in_f, out_f, gs = 32, 16, 8
    codes = torch.randint(0, 16, (in_f, out_f), dtype=torch.int32)
    qweight = _ref_pack_qweight(codes)
    scales = torch.rand(in_f // gs, out_f) * 0.02 + 0.01
    qzeros = _ref_pack_qzeros(torch.full((in_f // gs, out_f), 8, dtype=torch.int32))

    p0, s0 = transcode_autogptq_to_resident(qweight, scales, qzeros=None, group_size=gs)
    p1, s1 = transcode_autogptq_to_resident(qweight, scales, qzeros=qzeros, group_size=gs)
    torch.testing.assert_close(p0, p1, rtol=0, atol=0)
    torch.testing.assert_close(s0, s1, rtol=0, atol=0)
