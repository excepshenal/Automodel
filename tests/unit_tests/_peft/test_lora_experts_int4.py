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

"""GPU tests for int4-resident MoE experts (mirrors test_lora_experts_mxfp4.py over the int4 codec).

These exercise the CUDA ``Int4GroupedMM`` grouped GEMM (re-dequant-in-backward) that the CPU
codec tests skip: forward/backward parity against the bf16 grouped-mm path on int4-representable
weights, the no-routed-tokens dummy path, base-weight freezing, and the torch_mm backend guard.
"""

import pytest
import torch

from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA, GroupedExpertsLoRAInt4
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.int4_experts import GroupedExpertsInt4
from nemo_automodel.components.moe.layers import GroupedExperts
from nemo_automodel.components.quantization.int4 import INT4_GROUP_SIZE, dequantize_int4, quantize_int4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def device():
    return torch.device(f"cuda:{torch.cuda.current_device()}")


@pytest.fixture
def moe_config():
    # Contraction dims (dim, moe_inter_dim) must be divisible by the int4 group size (128).
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=INT4_GROUP_SIZE,
        inter_dim=2 * INT4_GROUP_SIZE,
        moe_inter_dim=INT4_GROUP_SIZE,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


def _make_representable_(experts):
    """Round each base projection onto the symmetric int4 group-128 grid so packed == bf16."""
    with torch.no_grad():
        for name in ("gate_and_up_projs", "down_projs"):
            param = getattr(experts, name)
            w_t = param.data.transpose(-2, -1).contiguous()  # [E, out, in]
            packed, scales = quantize_int4(w_t, INT4_GROUP_SIZE)
            deq = dequantize_int4(packed, scales.to(torch.bfloat16), param.dtype, INT4_GROUP_SIZE)
            param.data.copy_(deq.transpose(-2, -1))


def _routing_inputs(moe_config, num_tokens, device, dtype):
    torch.manual_seed(7)
    x = torch.randn(num_tokens, moe_config.dim, dtype=dtype, device=device, requires_grad=True)
    indices = torch.stack(
        [
            torch.randperm(moe_config.n_routed_experts, device=device)[: moe_config.n_activated_experts]
            for _ in range(num_tokens)
        ]
    )
    weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=dtype, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    return x, token_mask, weights, indices


def test_int4_module_packs_base_weights(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)

    lora = GroupedExpertsLoRAInt4(orig, lora_dim=8, alpha=16)

    assert lora._int4_resident
    assert not hasattr(lora, "gate_and_up_projs")
    assert not hasattr(lora, "down_projs")
    assert lora.gate_and_up_projs_packed.dtype == torch.int32
    assert lora.gate_and_up_projs_scales.dtype == torch.bfloat16
    assert not lora.gate_and_up_projs_packed.requires_grad
    # Checkpoint orientation [E, out, in//8] for gate+up ([E, 2*inter, dim//8]).
    assert lora.gate_and_up_projs_packed.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // 8)
    assert lora.gate_and_up_projs_scales.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // INT4_GROUP_SIZE)
    assert lora.down_projs_packed.shape == (4, moe_config.dim, moe_config.moe_inter_dim // 8)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"}


def test_int4_forward_backward_matches_bf16(moe_config, device):
    """With int4-representable base weights, the packed module must match the bf16 module."""
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    _make_representable_(orig)

    ref = GroupedExpertsLoRA(orig, lora_dim=8, alpha=16).to(device)
    q = GroupedExpertsLoRAInt4(orig, lora_dim=8, alpha=16).to(device)
    with torch.no_grad():
        for name in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"):
            getattr(q, name).data.copy_(getattr(ref, name).data)

    x_ref, token_mask, weights, indices = _routing_inputs(moe_config, 32, device, torch.bfloat16)
    x_q = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref(x_ref, token_mask, weights, indices)
    y_q = q(x_q, token_mask, weights, indices)
    torch.testing.assert_close(y_q, y_ref, atol=1e-5, rtol=1e-5)

    y_ref.float().pow(2).sum().backward()
    y_q.float().pow(2).sum().backward()
    torch.testing.assert_close(x_q.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
    for name in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"):
        torch.testing.assert_close(getattr(q, name).grad, getattr(ref, name).grad, atol=1e-4, rtol=1e-4)
    assert q.gate_and_up_projs_packed.grad is None
    assert q.down_projs_packed.grad is None


def test_int4_no_routed_tokens(moe_config, device):
    """The all-masked dummy path must run and keep LoRA gradients flowing."""
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)

    q = GroupedExpertsLoRAInt4(orig, lora_dim=8, alpha=16).to(device)
    x, _, weights, indices = _routing_inputs(moe_config, 8, device, torch.bfloat16)
    token_mask = torch.zeros(8, dtype=torch.bool, device=device)

    y = q(x, token_mask, weights, indices)
    assert y.shape == x.shape
    y.float().sum().backward()
    assert x.grad is not None


def test_int4_requires_torch_mm_backend(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)  # default per-expert loop backend
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    with pytest.raises(NotImplementedError, match="torch_mm"):
        GroupedExpertsLoRAInt4(orig, lora_dim=8, alpha=16)


def test_patch_moe_module_int4(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)

    patched = patch_moe_module(orig, dim=4, alpha=8, expert_weight_format="int4")
    assert isinstance(patched, GroupedExpertsLoRAInt4)

    patched_bf16 = patch_moe_module(orig, dim=4, alpha=8)
    assert isinstance(patched_bf16, GroupedExpertsLoRA)
    assert not isinstance(patched_bf16, GroupedExpertsLoRAInt4)


def test_frozen_int4_packs_and_freezes(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)

    q = GroupedExpertsInt4(orig)
    assert q._int4_resident
    assert not hasattr(q, "gate_and_up_projs")
    assert q.gate_and_up_projs_packed.dtype == torch.int32
    assert q.down_projs_scales.dtype == torch.bfloat16
    assert [n for n, p in q.named_parameters() if p.requires_grad] == []


def test_frozen_int4_forward_matches_bf16(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    _make_representable_(orig)

    q = GroupedExpertsInt4(orig)
    x, token_mask, weights, indices = _routing_inputs(moe_config, 32, device, torch.bfloat16)

    y_ref = orig(x.detach(), token_mask, weights, indices)
    y_q = q(x.detach(), token_mask, weights, indices)
    torch.testing.assert_close(y_q, y_ref, atol=1e-5, rtol=1e-5)
