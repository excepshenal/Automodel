# Experiment: mxfp4 dequant throughput at 4096-token packed sequence

## Goal

Measure the **new vs old mxfp4 expert-dequant** training throughput for
DeepSeek-V4-Flash LoRA at **4096-token packed sequence** with **tilelang
attention**, on one 8×H200 node, and compare against the established baselines.
This validates (or corrects) a projection that couldn't be measured locally —
the bare `.venv` has no tilelang kernel, so the local A/B was forced onto
`attn=torch` and a short-sequence regime.

## Background — what changed

HEAD of this branch (`dshen/feat/mxfp4-expert-lora-local`) carries
`perf(mxfp4): faster, spec-correct expert dequant`, which reworks
`dequantize_mxfp4` / `quantize_mxfp4` (single byte-table gather, blocked-scale
broadcast, direct `float8_e8m0fnu→float32` cast, `frexp` exponent). It is
**bit-exact** with the previous kernel (round-trip max_diff 0.0; loss identical
at every step in a local run), so this is a pure throughput question.

Numbers already in hand:
- **Microbench** (per-rank expert shapes `[32,4096,4096]`/`[32,4096,2048]`):
  dequant **2.39× faster**; dequant is ~95–98% of the expert-op cost at
  128 tok/expert.
- **Local 8-GPU e2e** (attn=torch, short unpacked HellaSwag, dequant-bound):
  **~2.5× tokens/sec**, loss bit-identical to the old kernel.
- **Prior baselines at 4096 packed seq (tilelang attn):** old mxfp4 **~4,970
  tps**, bf16 **~7,740 tps** (mxfp4 was 1.56× slower).
- **Projection for 4096 packed (unverified):** new mxfp4 **~6,300 tps**
  (~1.27× over old 4,970), narrowing the bf16 gap from 1.56× → ~1.23×. The gain
  is smaller than the short-seq 2.5× because at 4096 seq the grouped-GEMM
  compute is a larger share of the step, so the dequant speedup is amortized.

**MEASURED 2026-06-13 (8×H200, this procedure):** new mxfp4 **6,567 tps**,
old mxfp4 **5,040 tps** → **1.30× over old**, slightly beating the ~6,300
projection. Old-kernel parity check: 5,040 ≈ the historical 4,970 baseline
(within 1.4%), so the `dispatcher=torch` regime matches the original baseline
regime and the ratio maps onto the published numbers. The new kernel also cuts
**~6 GiB/rank** (48.9 vs 54.9 GiB), narrowing the bf16 gap from 1.56× → **~1.18×**
(6,567 vs ~7,740). See the Results section below for the full table.

## Why it must run in the container

`scripts/launch_dsv4_lora_docker.sh` runs `torchrun` inside
`nvcr.io/nvidia/nemo-automodel:26.04` (TE / deep_ep prebuilt) and
**pip-installs `tilelang==0.1.10 tile_kernels==1.0.0`** in-container, which the
bare host `.venv` lacks. Tilelang attention is required to reproduce the
4,970/7,740 baseline regime. mxfp4 experts require the torch grouped-GEMM path
(`dispatcher=torch`, `experts=torch_mm`) — DeepEP + mxfp4 is unsupported.

## Prerequisites on the run node

- Docker + nvidia runtime; image `nvcr.io/nvidia/nemo-automodel:26.04` pulled.
- This branch checked out at `$REPO_DIR` (default `$HOME/excepshenal/Automodel`).
- **fp4** DeepSeek-V4-Flash checkpoint at `MODEL_PATH` (the packed checkpoint,
  not the BF16 one) — default below uses `/raid0/data/models/DeepSeek-V4-Flash`.
- A writable `CKPT_DIR`.

## Procedure (A/B)

Run both arms with **identical** config and `seed: 1234` (deterministic data
order ⇒ matched tokens/step ⇒ directly comparable tps). Record **steady-state
tps** from the per-step log lines (`| step N | ... | tps X(.../gpu)`), dropping
the first ~3 warmup steps; average steps ~5–15. **Kill after ~15 steps** — do
not wait for the end-of-training validation (it runs a full val epoch; we only
want training throughput).

Common env for both arms:

```bash
export REPO_DIR=$HOME/excepshenal/Automodel
export MODEL_PATH=/raid0/data/models/DeepSeek-V4-Flash      # fp4 checkpoint
export CKPT_DIR=/external-disk/mxfp4_tps_experiment
export CONFIG_YAML=examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag_lora_mxfp4.yaml
export ATTN_BACKEND=tilelang        # reproduce baseline regime
export DISPATCHER=torch             # mxfp4 requires torch dispatch + torch_mm experts
export GBS=16 LBS=1
# packed 4096; push val/ckpt out of the way; high max_steps (we kill early)
export EXTRA_ARGS="--packed_sequence.packed_sequence_size 4096 \
  --step_scheduler.max_steps 50 \
  --step_scheduler.val_every_steps 100000 \
  --checkpoint.enabled false"
```

### Arm A — NEW dequant (this HEAD)

```bash
cd $REPO_DIR
git rev-parse --short HEAD          # expect the perf(mxfp4) commit
bash scripts/launch_dsv4_lora_docker.sh 2>&1 | tee /tmp/mxfp4_new.log
# watch for ~15 steady steps, then Ctrl-C / docker stop
```

### Arm B — OLD dequant (parent of the perf commit)

Swap only `mxfp4.py` back to the pre-optimization version (the repo is mounted
into the container, so a host-side checkout is what the run sees):

```bash
cd $REPO_DIR
git checkout HEAD~1 -- nemo_automodel/components/quantization/mxfp4.py
grep -c frexp nemo_automodel/components/quantization/mxfp4.py   # expect 0 (old)
bash scripts/launch_dsv4_lora_docker.sh 2>&1 | tee /tmp/mxfp4_old.log
# ...record steady tps, then:
git checkout HEAD -- nemo_automodel/components/quantization/mxfp4.py   # restore NEW
```

### Optional Arm C — bf16 reference

Re-confirm the ~7,740 bf16 number if desired: point `MODEL_PATH` at the BF16
checkpoint and use the bf16 recipe
(`CONFIG_YAML=...deepseek_v4_flash_hellaswag_lora.yaml`, drop
`expert_weight_format`); keep `attn=tilelang`, same packed-4096 EXTRA_ARGS.

## Results (measured 2026-06-13, 8×H200)

Steady-state avg over steps 5–15 (step 0 excluded — one-time compile warmup,
524 tps). Tokens/step matched exactly across arms (identical `num_label_tokens`
per step), so tps is directly comparable.

| arm | tps (steady avg) | vs old | vs bf16 | mem/rank |
|---|---|---|---|---|
| old mxfp4 | **5,040** | 1.0× | 0.65× | 54.9 GiB |
| **new mxfp4** | **6,567** | **1.30×** | **0.85×** | **48.9 GiB** |
| bf16 (historical) | ~7,740 | 1.54× | 1.0× | — |

- **1.30× new-over-old**, slightly beating the ~1.27× / ~6,300 projection.
- **Parity confirmed:** old-kernel-here 5,040 ≈ historical 4,970 (within 1.4%),
  so `dispatcher=torch` reproduces the original baseline regime.
- bf16 gap narrows **1.56× → ~1.18×** (bf16 is 7,740/6,567 = 1.18× the new mxfp4).
- New kernel uses **~6 GiB/rank less** (48.9 vs 54.9 GiB) — a memory bonus on top
  of throughput.

**Step-0 loss was NOT bit-identical** (new 8.9743 vs old 8.9569, ~0.2%). Under
`attn=tilelang` the forward is non-deterministic run-to-run (atomics in the
sparse-attn / indexer kernels), so loss-identity is not a usable cross-arm
sanity in this regime — unlike the `attn=torch` local run where it held. The
dequant itself is still bit-exact (microbench round-trip max_diff 0.0); losses
tracking within ~0.2% is consistent with that. Use matched `num_label_tokens`
(not loss identity) to confirm the arms saw the same data.

The 1.30× measurement is within ~2% of the ~1.27× projection (not a material
difference), so the `perf(mxfp4)` commit message is left as-is.

## Notes / gotchas

- mxfp4 needs `DISPATCHER=torch` (sets `experts=torch_mm` automatically in the
  launcher); the YAML default `deepep` will not run with mxfp4 experts.
- The launcher's DeepEP-OOM-patch preflight warning is irrelevant here (we use
  torch dispatch, not deepep).
- **tilelang import abort:** an unpinned `tilelang==0.1.10` install resolves
  `apache-tvm-ffi` to `0.1.12`, which double-registers TVM FFI type index 130
  and aborts all ranks at import (SIGABRT, before step 0). The launcher now pins
  `apache-tvm-ffi==0.1.11` (highest in-spec version that imports cleanly).
- **Offline dataset:** the recipe's `dataset.path_or_dataset: rowan/hellaswag`
  is a Hub ID, but the launcher forces `HF_DATASETS_OFFLINE=1`. Override both
  train and val to the mounted local copy:
  `--dataset.path_or_dataset /external-disk/data/dataset/hellaswag`
  `--validation_dataset.path_or_dataset /external-disk/data/dataset/hellaswag`
  (`/external-disk` is already bind-mounted since `CKPT_DIR` lives under it).
- `seed: 1234` is in the recipe; keep it so tokens/step match across arms.
- The local short-seq result (~2.5×) and this long-seq result bracket the win:
  the optimization helps most where dequant dominates (short seq) and least
  where GEMM compute dominates (long seq). Expect this number on the low end.
