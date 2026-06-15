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

"""CPU tests for the GLM-5.1 mixed-precision (int4 routed experts / int8 attention+shared)
AutoGPTQ -> native int4-resident state-dict adapter path.

Synthetic AutoGPTQ tensors are built with reference packers so correctness is self-verified
without the real INC4AI checkpoint or a GPU. ``group_size`` is shrunk to fit the tiny dims.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm4_moe.state_dict_adapter import Glm4MoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.quantization.gptq_int4 import (
    dequantize_autogptq,
    transcode_autogptq_to_resident,
)
from nemo_automodel.components.quantization.int4 import dequantize_int4, quantize_int4

HIDDEN = 16
MOE_INTER = 16
N_EXPERTS = 2
GS = 8


def _ref_pack_qweight_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """codes [in, out] -> AutoGPTQ int32 qweight [in // pack, out] (pack = 32 // bits)."""
    in_f, out_f = codes.shape
    pack = 32 // bits
    mask = (1 << bits) - 1
    codes = codes.to(torch.int32)
    qw = torch.zeros(in_f // pack, out_f, dtype=torch.int32)
    for j in range(pack):
        qw |= (codes[j::pack] & mask) << (j * bits)
    return qw


def _make_autogptq_sym(weight: torch.Tensor, bits: int, group_size: int = GS):
    """Build a symmetric AutoGPTQ {qweight, scales} for a linear weight ``[out, in]``.

    Round-to-nearest onto the symmetric grid (zero == 2**(bits-1)); returns the AutoGPTQ tensors
    plus the dense weight that this checkpoint dequantizes to (the RTN reconstruction of ``weight``).
    """
    out_f, in_f = weight.shape
    qmax = (1 << (bits - 1)) - 1  # 7 for int4, 127 for int8
    zero = 1 << (bits - 1)
    w_io = weight.t().contiguous()  # [in, out]
    blocks = w_io.view(in_f // group_size, group_size, out_f)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax  # [G, 1, out]
    codes = (torch.round(blocks / scale).clamp(-qmax, qmax) + zero).to(torch.int32)  # [G, gs, out]
    codes = codes.reshape(in_f, out_f)
    qweight = _ref_pack_qweight_bits(codes, bits)
    scales = scale.squeeze(1)  # [G, out]
    return qweight, scales


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=HIDDEN,
        inter_dim=MOE_INTER,
        moe_inter_dim=MOE_INTER,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
    )


@pytest.fixture
def adapter(moe_config):
    config = SimpleNamespace(quantization_config={"bits": 4, "group_size": GS})
    backend = BackendConfig(enable_hf_state_dict_adapter=False)
    return Glm4MoeStateDictAdapter(
        config=config, moe_config=moe_config, backend=backend, dtype=torch.bfloat16, expert_storage_format="int4"
    )


def _build_mixed_checkpoint():
    """A one-layer GLM-5.1-style mixed checkpoint plus the reference dense weights it encodes."""
    torch.manual_seed(0)
    sd = {}
    ref = {}

    # int4 routed experts (gate/up: [moe_inter, hidden]; down: [hidden, moe_inter]).
    for e in range(N_EXPERTS):
        for proj, shape in (
            ("gate_proj", (MOE_INTER, HIDDEN)),
            ("up_proj", (MOE_INTER, HIDDEN)),
            ("down_proj", (HIDDEN, MOE_INTER)),
        ):
            w = torch.randn(shape) * 0.05
            qweight, scales = _make_autogptq_sym(w, bits=4)
            base = f"model.layers.0.mlp.experts.{e}.{proj}"
            sd[f"{base}.qweight"] = qweight
            sd[f"{base}.scales"] = scales
            ref[base] = dequantize_autogptq(qweight, scales, group_size=GS, dtype=torch.float32, bits=4)

    # int8 attention + shared experts.
    for base, shape in (
        ("model.layers.0.self_attn.q_proj", (HIDDEN, HIDDEN)),
        ("model.layers.0.self_attn.o_proj", (HIDDEN, HIDDEN)),
        ("model.layers.0.mlp.shared_experts.gate_proj", (MOE_INTER, HIDDEN)),
        ("model.layers.0.mlp.shared_experts.down_proj", (HIDDEN, MOE_INTER)),
    ):
        w = torch.randn(shape) * 0.05
        qweight, scales = _make_autogptq_sym(w, bits=8)
        sd[f"{base}.qweight"] = qweight
        sd[f"{base}.scales"] = scales
        ref[base] = dequantize_autogptq(qweight, scales, group_size=GS, dtype=torch.float32, bits=8)

    # bf16 pass-through tensors (router gate, norms, embeddings, lm_head).
    passthrough = {
        "model.layers.0.mlp.gate.weight": torch.randn(N_EXPERTS, HIDDEN),
        "model.layers.0.mlp.gate.e_score_correction_bias": torch.zeros(N_EXPERTS),
        "model.layers.0.input_layernorm.weight": torch.randn(HIDDEN),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(HIDDEN),
        "model.embed_tokens.weight": torch.randn(8, HIDDEN),
        "lm_head.weight": torch.randn(8, HIDDEN),
    }
    sd.update(passthrough)
    return sd, ref, passthrough


def test_from_hf_int4_routed_experts_are_packed(adapter):
    sd, ref, _ = _build_mixed_checkpoint()
    native = adapter.from_hf(sd)

    gup_packed = native["model.layers.0.mlp.experts.gate_and_up_projs_packed"]
    gup_scales = native["model.layers.0.mlp.experts.gate_and_up_projs_scales"]
    down_packed = native["model.layers.0.mlp.experts.down_projs_packed"]
    down_scales = native["model.layers.0.mlp.experts.down_projs_scales"]

    # gate||up concatenated along the output dim, experts stacked on dim 0.
    assert gup_packed.shape == (N_EXPERTS, 2 * MOE_INTER, HIDDEN // 8)
    assert gup_scales.shape == (N_EXPERTS, 2 * MOE_INTER, HIDDEN // GS)
    assert down_packed.shape == (N_EXPERTS, HIDDEN, MOE_INTER // 8)
    assert gup_packed.dtype == torch.int32 and gup_scales.dtype == torch.bfloat16

    # Per-expert resident dequant must reproduce the AutoGPTQ reference (lossless transcode).
    for e in range(N_EXPERTS):
        gate = dequantize_int4(gup_packed[e][:MOE_INTER], gup_scales[e][:MOE_INTER], torch.float32, group_size=GS)
        up = dequantize_int4(gup_packed[e][MOE_INTER:], gup_scales[e][MOE_INTER:], torch.float32, group_size=GS)
        down = dequantize_int4(down_packed[e], down_scales[e], torch.float32, group_size=GS)
        torch.testing.assert_close(gate, ref[f"model.layers.0.mlp.experts.{e}.gate_proj"], rtol=0, atol=2e-2)
        torch.testing.assert_close(up, ref[f"model.layers.0.mlp.experts.{e}.up_proj"], rtol=0, atol=2e-2)
        torch.testing.assert_close(down, ref[f"model.layers.0.mlp.experts.{e}.down_proj"], rtol=0, atol=2e-2)


def test_from_hf_int4_dequantizes_int8_attention_and_shared(adapter):
    sd, ref, _ = _build_mixed_checkpoint()
    native = adapter.from_hf(sd)

    for base in (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.shared_experts.gate_proj",
        "model.layers.0.mlp.shared_experts.down_proj",
    ):
        assert f"{base}.qweight" not in native and f"{base}.scales" not in native
        w = native[f"{base}.weight"]
        assert w.dtype == torch.bfloat16
        torch.testing.assert_close(w.float(), ref[base], rtol=0, atol=1e-2)


def test_from_hf_int4_passes_through_bf16_tensors(adapter):
    sd, _, passthrough = _build_mixed_checkpoint()
    native = adapter.from_hf(sd)
    for key, value in passthrough.items():
        assert key in native
        torch.testing.assert_close(native[key], value)


def test_to_hf_packed_split_round_trips_aggregation(adapter):
    sd, _, _ = _build_mixed_checkpoint()
    native = adapter.from_hf(sd)

    # Split the stacked gate_and_up packed param back to per-expert keys, then verify the
    # split tensors re-stack to the original aggregated param.
    fqn = "model.layers.0.mlp.experts.gate_and_up_projs_packed"
    split = dict(adapter.convert_single_tensor_to_hf(fqn, native[fqn]))
    assert set(split) == {
        f"model.layers.0.mlp.experts.{e}.{p}.weight_packed" for e in range(N_EXPERTS) for p in ("gate_proj", "up_proj")
    }
    restacked = torch.stack(
        [
            torch.cat(
                [
                    split[f"model.layers.0.mlp.experts.{e}.gate_proj.weight_packed"],
                    split[f"model.layers.0.mlp.experts.{e}.up_proj.weight_packed"],
                ],
                dim=0,
            )
            for e in range(N_EXPERTS)
        ],
        dim=0,
    )
    torch.testing.assert_close(restacked, native[fqn], rtol=0, atol=0)


def test_rtn_vs_autoround_resident_closeness():
    # The resident transcode of an AutoGPTQ (auto-round) int4 linear and a direct RTN int4
    # quantization of the same dense weight must land within ~one group scale of each other:
    # both approximate the weight on the same symmetric group-128 grid.
    torch.manual_seed(3)
    out_f, in_f = 32, 64
    w = torch.randn(out_f, in_f) * 0.1

    # "auto-round" stand-in: a symmetric AutoGPTQ checkpoint of w, transcoded to resident.
    qweight, scales = _make_autogptq_sym(w, bits=4, group_size=GS)
    packed_ar, scales_ar = transcode_autogptq_to_resident(qweight, scales, group_size=GS)
    w_ar = dequantize_int4(packed_ar, scales_ar, torch.float32, group_size=GS)

    # RTN: quantize the dense weight directly.
    packed_rtn, scales_rtn = quantize_int4(w, group_size=GS)
    w_rtn = dequantize_int4(packed_rtn, scales_rtn, torch.float32, group_size=GS)

    per_group_scale = scales_rtn.float().repeat_interleave(GS, dim=-1)
    assert torch.all((w_ar - w_rtn).abs() <= per_group_scale + 1e-5)
    # Both should track the original weight to within half a step (RTN error bound).
    assert torch.all((w_rtn - w).abs() <= per_group_scale / 2 + 1e-5)
