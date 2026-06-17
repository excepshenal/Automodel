# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Build a CrusoeAI GLM-5.1 checkpoint: Intel/AutoRound int4 routed experts + original bf16 rest.

The published INC4AI GLM-5.1 checkpoint is mixed-bit: routed experts int4, but attention /
indexer / shared experts int8 and a few layers bf16. Our int4-resident LoRA path keeps only the
routed experts packed (int4) and holds every other frozen base weight in bf16, dequantizing the
int8 on load. That int8 -> bf16 round-trip is both lossy and forces a not-yet-implemented int8
``to_hf`` emission on the multi-GPU DCP load path.

This script sidesteps both by sourcing the non-expert weights straight from the original bf16
GLM-5.1 (full precision) and keeping only Intel's auto-round int4 experts:

    derived = { original[k]  for k NOT a routed expert (bf16 .weight) }
            U { intel[k]     for k a routed expert    (int4 qweight/qzeros/scales[/g_idx]) }

The result is an "int4 experts + bf16 everything-else" checkpoint that the existing GLM
state-dict adapter loads via validated paths only (expert triples -> resident transcode;
bf16 .weight -> passthrough). The derived config keeps a quantization_config so the adapter's
int4 mode engages, but its per-module bit map is rewritten so non-expert linears read as bf16
(bits 8 -> 16), matching the actual tensors.

Usage:
    python scripts/build_glm51_int4_crusoe_checkpoint.py \
        --orig  /raid0/data/models/GLM-5.1 \
        --intel /raid0/data/models/GLM-5.1-int4-mixed-AutoRound \
        --out   /raid0/GLM-5.1-int4-experts-bf16-base-AutoRound \
        [--dry-run] [--shard-size-gb 10]
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# A routed-expert weight: ``...mlp.experts.<id>.<proj>...``. Shared experts (``mlp.shared_experts``)
# deliberately do NOT match — they stay bf16 from the original, like attention.
_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")

# Aux files copied verbatim from the original bf16 checkpoint (tokenizer, chat template, etc.).
_AUX_COPY = (
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
)


def _load_index(path: str) -> dict[str, str]:
    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _partition_keys(orig_map: dict[str, str], intel_map: dict[str, str]) -> tuple[dict, dict]:
    """Return ({intel expert keys}, {original non-expert keys}) as key -> source shard filename."""
    intel_expert = {k: v for k, v in intel_map.items() if _ROUTED_EXPERT_RE.search(k)}
    orig_nonexpert = {k: v for k, v in orig_map.items() if not _ROUTED_EXPERT_RE.search(k)}
    return intel_expert, orig_nonexpert


def _validate(orig_map: dict[str, str], intel_map: dict[str, str], intel_expert: dict, orig_nonexpert: dict) -> None:
    """Sanity-check coverage before writing hundreds of GB."""
    # No key is sourced twice.
    overlap = set(intel_expert) & set(orig_nonexpert)
    assert not overlap, f"key sourced from both checkpoints: {sorted(overlap)[:5]}"

    # Every routed-expert weight the original has (bf16) is replaced by an Intel int4 triple, i.e.
    # the set of expert *modules* matches between the two checkpoints.
    def _expert_modules(m):
        return {k.rsplit(".", 1)[0] for k in m if _ROUTED_EXPERT_RE.search(k)}

    orig_exp_mods = _expert_modules(orig_map)
    intel_exp_mods = {k.rsplit(".", 1)[0] for k in intel_expert}  # strip qweight/qzeros/scales/g_idx
    missing = orig_exp_mods - intel_exp_mods
    extra = intel_exp_mods - orig_exp_mods
    assert not missing, f"{len(missing)} expert modules in orig have no Intel int4 source, e.g. {sorted(missing)[:5]}"
    assert not extra, f"{len(extra)} Intel expert modules absent from orig, e.g. {sorted(extra)[:5]}"

    # Intel's non-expert content (int8 triples + bf16) is intentionally dropped; report it.
    intel_nonexpert = {k for k in intel_map if not _ROUTED_EXPERT_RE.search(k)}
    print(f"  validation OK: {len(orig_exp_mods)} routed-expert modules aligned")
    print(f"  derived non-expert keys (from orig bf16): {len(orig_nonexpert)}")
    print(f"  derived expert keys     (from intel int4): {len(intel_expert)}")
    print(f"  dropped intel non-expert keys (int8/bf16):  {len(intel_nonexpert)}")


def _derive_config(orig_path: str, intel_path: str, out_path: str) -> None:
    """Write config.json: original GLM config + int4 quant config with non-expert promoted to bf16."""
    with open(os.path.join(orig_path, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(intel_path, "config.json")) as f:
        intel_cfg = json.load(f)
    qc = dict(intel_cfg.get("quantization_config", {}))
    # Rewrite the per-module bit map so the non-expert linears (Intel's bits=8) read as bf16,
    # matching the actual derived tensors; routed experts keep the default bits=4.
    ec = qc.get("extra_config")
    if isinstance(ec, dict):
        qc["extra_config"] = {
            k: ({**v, "bits": 16, "data_type": "float"} if v.get("bits") == 8 else v) for k, v in ec.items()
        }
    cfg["quantization_config"] = qc
    with open(os.path.join(out_path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def _copy_aux(orig_path: str, out_path: str) -> None:
    for name in _AUX_COPY:
        src = os.path.join(orig_path, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_path, name))


def _stream_build(
    intel_expert: dict, orig_nonexpert: dict, orig_path: str, intel_path: str, out_path: str, shard_size_gb: float
) -> None:
    """Read each needed source shard once, write derived shards of ~shard_size_gb, build the index."""
    shard_budget = int(shard_size_gb * (1 << 30))

    # Group target keys by (source_path, source_shard) so each source shard is opened once.
    by_src: dict[tuple[str, str], list[str]] = {}
    for k, shard in orig_nonexpert.items():
        by_src.setdefault((orig_path, shard), []).append(k)
    for k, shard in intel_expert.items():
        by_src.setdefault((intel_path, shard), []).append(k)

    weight_map: dict[str, str] = {}
    total_size = 0
    out_idx = 0
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0

    def _flush():
        nonlocal buf, buf_bytes, out_idx
        if not buf:
            return
        out_idx += 1
        fname = f"model-{out_idx:05d}.safetensors"
        save_file(buf, os.path.join(out_path, fname), metadata={"format": "pt"})
        for kk in buf:
            weight_map[kk] = fname
        print(f"  wrote {fname}: {len(buf)} tensors, {buf_bytes / (1 << 30):.2f} GB", flush=True)
        buf = {}
        buf_bytes = 0

    n_src = len(by_src)
    for i, ((src_path, shard), keys) in enumerate(sorted(by_src.items())):
        with safe_open(os.path.join(src_path, shard), framework="pt", device="cpu") as f:
            for k in keys:
                t = f.get_tensor(k)
                buf[k] = t
                nbytes = t.nelement() * t.element_size()
                buf_bytes += nbytes
                total_size += nbytes
                if buf_bytes >= shard_budget:
                    _flush()
        if (i + 1) % 25 == 0 or i + 1 == n_src:
            print(f"  read source shard {i + 1}/{n_src}", flush=True)
    _flush()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(os.path.join(out_path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"  index: {len(weight_map)} keys across {out_idx} shards, {total_size / (1 << 30):.2f} GB total")


def main() -> None:
    """Parse args, validate key coverage, and (unless --dry-run) build the derived checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig", default="/raid0/data/models/GLM-5.1")
    ap.add_argument("--intel", default="/raid0/data/models/GLM-5.1-int4-mixed-AutoRound")
    ap.add_argument("--out", default="/raid0/GLM-5.1-int4-experts-bf16-base-AutoRound")
    ap.add_argument("--shard-size-gb", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true", help="Validate key coverage; write nothing.")
    args = ap.parse_args()

    orig_map = _load_index(args.orig)
    intel_map = _load_index(args.intel)
    intel_expert, orig_nonexpert = _partition_keys(orig_map, intel_map)

    print(f"[build] orig={args.orig} intel={args.intel} out={args.out}")
    _validate(orig_map, intel_map, intel_expert, orig_nonexpert)

    if args.dry_run:
        print("[build] --dry-run: validation passed, nothing written.")
        return

    os.makedirs(args.out, exist_ok=True)
    _derive_config(args.orig, args.intel, args.out)
    _copy_aux(args.orig, args.out)
    _stream_build(intel_expert, orig_nonexpert, args.orig, args.intel, args.out, args.shard_size_gb)
    print(f"[build] done -> {args.out}")


if __name__ == "__main__":
    main()
