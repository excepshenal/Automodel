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

"""CPU unit tests for the symmetric W4A16 int4 codec.

These cover only the pack/unpack/quantize/dequantize primitives (pure CPU torch). The
``Int4GroupedMM`` autograd function depends on ``torch._grouped_mm`` (CUDA) and is exercised
by the expert-layer integration tests, not here.
"""

import pytest
import torch

from nemo_automodel.components.quantization.int4 import (
    INT4_GROUP_SIZE,
    dequantize_int4,
    quantize_int4,
)


@pytest.mark.parametrize("group_size", [8, 32, 128])
def test_packed_and_scale_shapes(group_size):
    rows, k = 5, group_size * 4
    w = torch.randn(rows, k)
    packed, scales = quantize_int4(w, group_size=group_size)
    assert packed.dtype == torch.int32
    assert packed.shape == (rows, k // 8)
    assert scales.shape == (rows, k // group_size)


def test_on_grid_weights_round_trip_exactly():
    # A weight on the symmetric int4 grid must round-trip value-exactly. Codes are confined
    # to [-7, 7] (the symmetric range a scale = amax / 7 quantizer produces; -8 is unused),
    # with +7 forced into each group so the per-group amax recovers the original scale.
    torch.manual_seed(0)
    rows, k = 7, INT4_GROUP_SIZE * 3
    codes = torch.randint(-7, 8, (rows, k // INT4_GROUP_SIZE, INT4_GROUP_SIZE), dtype=torch.float32)
    codes[..., 0] = 7.0
    scale = (torch.rand(rows, k // INT4_GROUP_SIZE, 1) + 0.1) * 0.05
    w = (codes * scale).reshape(rows, k)

    packed, scales = quantize_int4(w)
    deq = dequantize_int4(packed, scales, torch.float32)
    torch.testing.assert_close(deq, w, rtol=0, atol=1e-6)


def test_rtn_error_bounded_by_half_scale():
    # Round-to-nearest error per element is at most scale/2 within the representable range.
    torch.manual_seed(1)
    rows, k = 4, INT4_GROUP_SIZE * 2
    w = torch.randn(rows, k)
    packed, scales = quantize_int4(w)
    deq = dequantize_int4(packed, scales, torch.float32)

    per_elem_scale = scales.repeat_interleave(INT4_GROUP_SIZE, dim=-1)
    # amax of each group maps to +7; only that element sits at the clamp edge, all others
    # are interior and bounded by half a step.
    err = (deq - w).abs()
    assert torch.all(err <= per_elem_scale / 2 + 1e-5)


def test_signed_unpack_covers_full_range():
    # Hand-pack eight nibbles per word spanning the full signed range, including the -8 code
    # (which the quantizer never emits but the unpacker must still decode), and check the
    # dequant matches a manual two's-complement decode with unit scale.
    codes = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [-8, -7, -6, -5, -4, -3, -2, -1]], dtype=torch.int32)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    packed = ((codes & 0xF) << shifts).sum(dim=-1).to(torch.int32).view(2, 1)  # [rows, K // 8]
    scales = torch.ones(2, 1)
    deq = dequantize_int4(packed, scales, torch.float32, group_size=8)
    torch.testing.assert_close(deq, codes.float(), rtol=0, atol=1e-6)


def test_dtype_is_respected():
    w = torch.randn(3, INT4_GROUP_SIZE)
    packed, scales = quantize_int4(w)
    deq = dequantize_int4(packed, scales, torch.bfloat16)
    assert deq.dtype == torch.bfloat16


def test_non_divisible_last_dim_raises():
    w = torch.randn(2, INT4_GROUP_SIZE + 1)
    with pytest.raises(AssertionError):
        quantize_int4(w)
