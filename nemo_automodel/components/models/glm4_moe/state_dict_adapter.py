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

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_submesh,
    is_dtensor,
    should_load_expert_for_rank,
    split_experts_weights_dtensor_aware,
)
from nemo_automodel.components.quantization.gptq_int4 import (
    dequantize_autogptq,
    infer_autogptq_bits,
    transcode_autogptq_to_resident,
)
from nemo_automodel.components.quantization.int4 import INT4_GROUP_SIZE

logger = logging.getLogger(__name__)

# Routed-expert linear in a GLM-5.1 HF checkpoint: model.layers.{L}.mlp.experts.{E}.{proj}.
_ROUTED_EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:model\.)?)layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)$"
)
# AutoGPTQ / auto-round store each quantized linear as a {qweight, qzeros, scales} triple keyed
# off a shared base (the would-be ``.weight`` key with the suffix stripped).
_QUANT_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")


class Glm4MoeStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF GLM4-MoE checkpoints and our grouped-experts native format.

    GLM4-MoE HF experts use keys:
      model.layers.{L}.mlp.experts.{E}.gate_proj.weight
      model.layers.{L}.mlp.experts.{E}.up_proj.weight
      model.layers.{L}.mlp.experts.{E}.down_proj.weight
      model.layers.{L}.mlp.shared_experts.gate_proj.weight
      model.layers.{L}.mlp.shared_experts.up_proj.weight
      model.layers.{L}.mlp.shared_experts.down_proj.weight

    Our native format groups them into:
      model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, dim, 2*moe_inter_dim]
      model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter_dim, dim]
      model.layers.{L}.mlp.shared_expert.gate_proj.weight
      model.layers.{L}.mlp.shared_expert.up_proj.weight
      model.layers.{L}.mlp.shared_expert.down_proj.weight
    """

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
        expert_storage_format: str = "bf16",
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        # "bf16": dequantize/merge experts to bf16 on load (default, GLM-4 path).
        # "int4": GLM-5.1 mixed AutoGPTQ/auto-round checkpoint — transcode the int4 routed
        # experts straight into resident packed storage (never materialized in bf16) and
        # dequantize the int8 attention + shared experts to bf16. Set by the PEFT/infra wiring
        # when peft.expert_weight_format='int4'.
        self.expert_storage_format = expert_storage_format

    @property
    def _quant_group_size(self) -> int:
        """Group size of the source AutoGPTQ checkpoint (per-group scales); defaults to 128."""
        qcfg = getattr(self.config, "quantization_config", None)
        if isinstance(qcfg, dict):
            gs = qcfg.get("group_size")
            if gs:
                return int(gs)
        return INT4_GROUP_SIZE

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Detect whether HF checkpoints use the "model." prefix
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and (key.endswith(".weight") or key.endswith(".qweight")):
                self._uses_model_prefix = key.startswith("model.")
                break
        if self.expert_storage_format == "int4":
            return self._from_hf_int4(hf_state_dict, device_mesh)
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    def _from_hf_int4(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> dict[str, Any]:
        """Convert a GLM-5.1 mixed AutoGPTQ/auto-round checkpoint to native int4-resident format.

        - Routed experts (int4) are transcoded losslessly into resident packed storage
          (``gate_and_up_projs_packed`` / ``_scales``, ``down_projs_packed`` / ``_scales``),
          mirroring DeepSeek-V4's ``_aggregate_experts_packed``: gate and up are concatenated
          along the output dim and experts are stacked on dim 0, both layout-preserving because
          packing runs along the contraction dim. No bf16 expert is ever materialized.
        - Attention and shared-expert linears (int8) are dequantized to bf16 ``.weight``.
        - The router gate, norms, first-``first_k_dense_replace`` dense MLPs, embeddings, and
          lm_head are bf16 and pass through unchanged.
        """
        group_size = self._quant_group_size
        n_experts = self.moe_config.n_routed_experts

        if device_mesh is not None:
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_per_rank = end_expert - start_expert
        else:
            rank = None
            expected_per_rank = n_experts

        # Group {qweight, qzeros, scales, g_idx} triples by their shared base key.
        triples: dict[str, dict[str, torch.Tensor]] = {}
        consumed: set[str] = set()
        for key in hf_state_dict.keys():
            for suffix in _QUANT_SUFFIXES:
                if key.endswith("." + suffix):
                    triples.setdefault(key[: -(len(suffix) + 1)], {})[suffix] = hf_state_dict[key]
                    consumed.add(key)
                    break

        # layer -> {"gate_and_up": {eid: {gate_proj/up_proj: (packed, scales)}}, "down": {eid: (packed, scales)}}
        by_layer: dict[str, dict] = {}
        out: dict[str, Any] = {}

        # Pass through every non-quantized tensor (norms, router gate, dense MLPs, embeddings,
        # lm_head, attention biases, qk-norms) untouched — GLM native keys match HF here.
        for key, value in hf_state_dict.items():
            if key not in consumed:
                out[key] = value

        def _local(t):
            return (t.to_local() if is_dtensor(t) else t) if t is not None else None

        for base, triple in triples.items():
            qweight = _local(triple.get("qweight"))
            scales = _local(triple.get("scales"))
            assert qweight is not None and scales is not None, f"AutoGPTQ base '{base}' missing qweight/scales"
            qzeros = _local(triple.get("qzeros"))
            g_idx = _local(triple.get("g_idx"))

            m = _ROUTED_EXPERT_RE.match(base)
            if m is None:
                # int8 attention / shared-expert linear -> dequantize to a bf16 weight.
                bits = infer_autogptq_bits(qweight, scales, group_size)
                out[base + ".weight"] = dequantize_autogptq(
                    qweight, scales, qzeros, g_idx, group_size, dtype=self.dtype, bits=bits
                )
                continue

            expert_num = int(m.group("expert"))
            if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                continue
            prefix, layer_num, proj = m.group("prefix"), m.group("layer"), m.group("proj")
            packed, res_scales = transcode_autogptq_to_resident(qweight, scales, qzeros, g_idx, group_size)

            layer = by_layer.setdefault(layer_num, {"gate_and_up": {}, "down": {}})
            if proj in ("gate_proj", "up_proj"):
                layer["gate_and_up"].setdefault(expert_num, {})[proj] = (packed, res_scales)
            else:  # down_proj
                layer["down"][expert_num] = (packed, res_scales)

            # The sub-dicts are popped once complete, so later keys touching the same layer
            # (e.g. the paired down_proj after gate/up finished) must tolerate their absence.
            gu = layer.get("gate_and_up")
            if (
                gu is not None
                and len(gu) == expected_per_rank
                and all("gate_proj" in d and "up_proj" in d for d in gu.values())
            ):
                eids = sorted(gu.keys())
                # cat(gate, up) along the output dim (dim 0 of [out, in // 8]); gate first to
                # match the bf16 path and the swiglu split in the expert forward.
                packed_stack = torch.stack(
                    [torch.cat([gu[e]["gate_proj"][0], gu[e]["up_proj"][0]], dim=0) for e in eids], dim=0
                )
                scale_stack = torch.stack(
                    [torch.cat([gu[e]["gate_proj"][1], gu[e]["up_proj"][1]], dim=0) for e in eids], dim=0
                )
                b = f"{prefix}layers.{layer_num}.mlp.experts.gate_and_up_projs"
                out[b + "_packed"] = create_dtensor_from_local(packed_stack, device_mesh, rank)
                out[b + "_scales"] = create_dtensor_from_local(scale_stack, device_mesh, rank)
                del layer["gate_and_up"]

            down = layer.get("down")
            if down is not None and len(down) == expected_per_rank:
                eids = sorted(down.keys())
                packed_stack = torch.stack([down[e][0] for e in eids], dim=0)
                scale_stack = torch.stack([down[e][1] for e in eids], dim=0)
                b = f"{prefix}layers.{layer_num}.mlp.experts.down_projs"
                out[b + "_packed"] = create_dtensor_from_local(packed_stack, device_mesh, rank)
                out[b + "_scales"] = create_dtensor_from_local(scale_stack, device_mesh, rank)
                del layer["down"]

        return out

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        quantization = kwargs.get("quantization", False)

        # int4 + quantization: build AutoGPTQ destination placeholders so the DCP planner can
        # straight-copy the on-disk int4 routed-expert tensors into them before from_hf
        # transcodes. Without quantization (e.g. resident DCP re-save) keep the resident split.
        packed_result = None
        if self.expert_storage_format == "int4":
            packed_result = self._split_packed_expert(fqn, tensor, quantization=quantization)
        if packed_result is not None:
            result = packed_result
        else:
            expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
            if expert_result is not None:
                result = expert_result
            else:
                result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    # Resident packed expert params (set by ``_from_hf_int4``): stacked [E, out, in // 8] codes
    # plus [E, out, in // group_size] scales. The two suffixes are stripped to recover the proj.
    _PACKED_EXPERT_RE = re.compile(
        r"^(?P<prefix>(?:model\.)?)layers\.(?P<layer>\d+)\.mlp\.experts\."
        r"(?P<which>gate_and_up_projs|down_projs)_(?P<kind>packed|scales)$"
    )

    def _split_packed_expert(
        self, fqn: str, tensor: Any, quantization: bool = False
    ) -> Optional[list[tuple[str, Any]]]:
        """Split a stacked resident-packed expert param into per-expert HF-style keys.

        Structural inverse of the expert aggregation in ``_from_hf_int4``: experts are split off
        dim 0 and (for the fused gate/up param) gate and up are split along the output dim.

        With ``quantization=True`` the per-expert tensors become **AutoGPTQ destination
        placeholders** (``.qweight`` / ``.scales`` with the on-disk int4 shapes/dtypes) so the
        DCP planner can straight-copy the checkpoint into them; ``from_hf`` then transcodes those
        loaded codes into resident packed storage. Empty tensors suffice because DCP overwrites
        their contents. Without ``quantization`` the per-expert tensors stay in resident packed
        layout (a plain resident re-save). Returns ``None`` for any non-packed-expert key.
        """
        m = self._PACKED_EXPERT_RE.match(fqn)
        if m is None:
            return None
        prefix, layer_num, which, kind = m.group("prefix"), m.group("layer"), m.group("which"), m.group("kind")
        expert_tensors, expert_ids = split_experts_weights_dtensor_aware(tensor, self.moe_config.n_routed_experts)

        result: list[tuple[str, Any]] = []
        base = f"{prefix}layers.{layer_num}.mlp.experts"

        def emit(eid: int, proj: str, t: torch.Tensor) -> None:
            if quantization:
                # Resident orientation is [out, in // {8, group_size}]; AutoGPTQ packs along the
                # input dim, so the destination is the transposed shape. Codes are int32; scales
                # match the resident scale dtype (the checkpoint stores per-group float scales).
                if kind == "packed":
                    dest = torch.empty(t.shape[1], t.shape[0], dtype=torch.int32, device=t.device)
                    result.append((f"{base}.{eid}.{proj}.qweight", dest))
                else:
                    dest = torch.empty(t.shape[1], t.shape[0], dtype=t.dtype, device=t.device)
                    result.append((f"{base}.{eid}.{proj}.scales", dest))
            else:
                suffix = "weight_packed" if kind == "packed" else "weight_scales"
                result.append((f"{base}.{eid}.{proj}.{suffix}", t))

        for t, eid in zip(expert_tensors, expert_ids):
            if which == "gate_and_up_projs":
                out_dim = t.shape[0] // 2  # gate || up concatenated along the output dim
                emit(eid, "gate_proj", t[:out_dim])
                emit(eid, "up_proj", t[out_dim:])
            else:  # down_projs
                emit(eid, "down_proj", t)
        return result
