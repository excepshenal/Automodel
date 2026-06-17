#!/usr/bin/env bash
# launch_glm51_lora_docker.sh — GLM-5.1 LoRA launch via the NGC nemo-automodel container.
# 2-node default (16xH200, EP=16, PP=1); single-node also works via env vars (no Slurm needed).
#
# Modeled on scripts/launch_dsv4_lora_docker.sh. GLM-5.1 maps to the
# GlmMoeDsaForCausalLM architecture (DeepSeek-style MLA + sparse-attention
# indexer + 256-expert MoE). The recipe loads weights via from_pretrained, so
# we override the model path (and thus the tokenizer, which the recipe derives
# from the model name) to a local directory for offline runs.
#
# The container ships TE, deep_ep, mamba-ssm, causal-conv1d, etc. pre-built,
# so no bare-metal CUDA / cuDNN dance is needed.  We mount the local fork over
# /opt/Automodel so any local edits take effect.
#
# Usage (2 nodes — same command on both, only NODE_RANK differs):
#   # node0:
#   NODE_RANK=0 NNODES=2 RDZV_ENDPOINT=<node0-ip>:29500 \
#       bash scripts/launch_glm51_lora_docker.sh
#   # node1:
#   NODE_RANK=1 NNODES=2 RDZV_ENDPOINT=<node0-ip>:29500 \
#       bash scripts/launch_glm51_lora_docker.sh
#
# Required (on each node):
#   - Docker with the nvidia container runtime (`docker info | grep Runtimes`
#     must list `nvidia`).
#   - The NGC image already pulled:
#       docker pull nvcr.io/nvidia/nemo-automodel:26.04
#   - GLM-5.1 safetensors at $MODEL_PATH (same path on both nodes), e.g.
#       hf download zai-org/GLM-5.1 --local-dir /raid0/data/models/GLM-5.1
#   - $CKPT_DIR writable.
#
# Env knobs (with defaults):
#   IMAGE       nvcr.io/nvidia/nemo-automodel:26.04
#   REPO_DIR    $HOME/excepshenal/Automodel   (bind-mounted to /opt/Automodel)
#   MODEL_PATH  /raid0/data/models/GLM-5.1
#   CKPT_DIR    /external-disk/GLM-5.1-hellaswag-lora-dshen-run-1
#   EP_SIZE     16
#   PP_SIZE     1
#   GBS         16
#   LBS         1
#   ACT_CKPT    true
#   ATTN_BACKEND sdpa   (the validated path for the GLM DSA sparse attention)
#   DISPATCHER  deepep  (requires NVSHMEM + IBGDA; switch to `torch` if your
#                        fabric/container lacks IBGDA support)
#   NNODES      2
#   NODE_RANK   0
#   NEMO_BASE_MODEL_LOAD_LAYER_CHUNK
#               unset/0  load the full model in one to_hf pass (original behavior).
#               GLM-5.1 (~744B) OOMs at load on 2 nodes because to_hf materializes a
#               2nd copy of the EP-sharded experts. Set to e.g. 4 to load 4 decoder
#               layers per chunk and cap the load-time peak (~92GB + a few GB/rank).
#   RDZV_ENDPOINT  127.0.0.1:29500   (override to <node0-ip>:29500 for 2-node)
#   EXTRA_ARGS  ""  appended to the automodel CLI call

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-automodel:26.04}"
REPO_DIR="${REPO_DIR:-$HOME/excepshenal/Automodel}"
MODEL_PATH="${MODEL_PATH:-/raid0/data/models/GLM-5.1}"
CKPT_DIR="${CKPT_DIR:-/external-disk/GLM-5.1-hellaswag-lora-dshen-run-1}"
# HF_CACHE is intentionally not used. Datasets are loaded from local paths
# under /external-disk/data/dataset, and model weights/tokenizer from
# $MODEL_PATH. Offline mode env vars below prevent any Hub fallback.
EP_SIZE="${EP_SIZE:-16}"
PP_SIZE="${PP_SIZE:-1}"
GBS="${GBS:-16}"
LBS="${LBS:-1}"
ACT_CKPT="${ACT_CKPT:-true}"
# GLM-5.1 uses DeepSeek-style sparse attention (DSA indexer). The shipped recipe
# uses sdpa, which is the validated non-TE path in DeepseekV32MLA. Unlike DSv4
# we do NOT pip-install tilelang here.
ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
# YAML default dispatcher: deepep, which requires NVSHMEM + IBGDA. If your
# fabric/container lacks IBGDA support, switch to `torch` (standard NCCL
# all-to-all). Slower at scale but works anywhere.
DISPATCHER="${DISPATCHER:-deepep}"
# Match the dispatcher's expected experts impl (gmm for deepep, torch_mm otherwise).
EXPERTS="${EXPERTS:-}"
if [[ -z "$EXPERTS" ]]; then
    case "$DISPATCHER" in
        deepep) EXPERTS="gmm" ;;
        *)      EXPERTS="torch_mm" ;;
    esac
fi
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-127.0.0.1:29500}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Recipe to run. Default is the bf16 GLM-5.1 LoRA recipe; point at
# examples/llm_finetune/glm/glm_5.1_lora_int4.yaml for the mixed-int4 checkpoint
# (set MODEL_PATH to the int4 dir and EXPERTS=torch_mm; int4 needs the torch_mm path).
RECIPE="${RECIPE:-examples/llm_finetune/glm/glm_5.1_lora.yaml}"
# Checkpoint knobs. For int4 a consolidated save would hit the not-yet-implemented int8
# to_hf emission, so for int4 smoke runs set CKPT_ENABLED=false (adapter-only or no save).
CKPT_ENABLED="${CKPT_ENABLED:-true}"
SAVE_CONSOLIDATED="${SAVE_CONSOLIDATED:-true}"
# Optional host HF cache to mount (datasets/tokenizers) when the recipe pulls a hub dataset
# id (e.g. rowan/hellaswag) under HF_*_OFFLINE=1. Mounted read-write because HF `datasets`
# acquires a FileLock in the cache dir on load. Empty = not mounted.
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

# --- preflight on the host ---------------------------------------------------
[[ -d "$REPO_DIR" ]] || { echo "FATAL: REPO_DIR not found: $REPO_DIR" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "FATAL: MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }
mkdir -p "$CKPT_DIR"

# This branch is off main and does NOT carry the lazy DeepEP buffer OOM patch
# (that lives on dshen-deepseek-v4). With eager init, a single node will OOM at
# load when dispatcher=deepep. 2+ nodes (the default here) are fine.
if grep -q "if self.dispatcher_backend == \"deepep\":" "$REPO_DIR/nemo_automodel/components/moe/experts.py" \
   && grep -A1 "if self.dispatcher_backend == \"deepep\":" "$REPO_DIR/nemo_automodel/components/moe/experts.py" \
        | grep -q "self._init_deepep_buffer(ep_group)"; then
    if [[ "$NNODES" -lt 2 && "$DISPATCHER" == "deepep" ]]; then
        echo "WARN: eager DeepEP buffer init active and NNODES<2 with dispatcher=deepep." >&2
        echo "      Single-node will likely OOM at load. Use 2+ nodes or DISPATCHER=torch." >&2
    fi
fi

# --- the in-container command -----------------------------------------------
# We launch torchrun from the container's python (canonical TE / deep_ep / cuDNN
# paths). --network=host means RDZV_ENDPOINT addresses resolve against the
# host's interface table, which is what makes the 2-node form work without
# extra port-forwarding.
read -r -d '' IN_CONTAINER_CMD <<EOF || true
set -euo pipefail
cd /opt/Automodel
echo "[container] image=$IMAGE node_rank=$NODE_RANK / $NNODES rdzv=$RDZV_ENDPOINT"
echo "[container] python: \$(python -V 2>&1)"
echo "[container] torch:  \$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
exec torchrun \\
    --nnodes=$NNODES \\
    --nproc-per-node=8 \\
    --node-rank=$NODE_RANK \\
    --master-addr=${RDZV_ENDPOINT%:*} \\
    --master-port=${RDZV_ENDPOINT#*:} \\
    -m nemo_automodel.cli.app \\
    $RECIPE \\
    --model.pretrained_model_name_or_path $MODEL_PATH \\
    --checkpoint.enabled $CKPT_ENABLED \\
    --checkpoint.checkpoint_dir $CKPT_DIR \\
    --checkpoint.model_save_format safetensors \\
    --checkpoint.save_consolidated $SAVE_CONSOLIDATED \\
    --distributed.pp_size $PP_SIZE \\
    --distributed.ep_size $EP_SIZE \\
    --distributed.activation_checkpointing $ACT_CKPT \\
    --model.backend.attn $ATTN_BACKEND \\
    --model.backend.dispatcher $DISPATCHER \\
    --model.backend.experts $EXPERTS \\
    --step_scheduler.global_batch_size $GBS \\
    --step_scheduler.local_batch_size $LBS \\
    $EXTRA_ARGS
EOF

echo "[host] launching docker run …"
exec docker run --rm \
    --gpus all \
    --network=host \
    --device=/dev/infiniband \
    --ipc=host \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    ${NCCL_SOCKET_IFNAME:+-e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"} \
    -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-ens7}}" \
    -e NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_0}" \
    ${NCCL_IB_DISABLE:+-e NCCL_IB_DISABLE="$NCCL_IB_DISABLE"} \
    ${NCCL_DEBUG:+-e NCCL_DEBUG="$NCCL_DEBUG"} \
    ${NVSHMEM_IB_ENABLE_IBGDA:+-e NVSHMEM_IB_ENABLE_IBGDA="$NVSHMEM_IB_ENABLE_IBGDA"} \
    ${NVSHMEM_IBGDA_NIC_HANDLER:+-e NVSHMEM_IBGDA_NIC_HANDLER="$NVSHMEM_IBGDA_NIC_HANDLER"} \
    ${NVSHMEM_DISABLE_LOCAL_ONLY_PROXY:+-e NVSHMEM_DISABLE_LOCAL_ONLY_PROXY="$NVSHMEM_DISABLE_LOCAL_ONLY_PROXY"} \
    ${NVSHMEM_REMOTE_TRANSPORT:+-e NVSHMEM_REMOTE_TRANSPORT="$NVSHMEM_REMOTE_TRANSPORT"} \
    ${NVSHMEM_DEBUG:+-e NVSHMEM_DEBUG="$NVSHMEM_DEBUG"} \
    ${NVSHMEM_DISABLE_CUDA_VMM:+-e NVSHMEM_DISABLE_CUDA_VMM="$NVSHMEM_DISABLE_CUDA_VMM"} \
    ${NVSHMEM_HCA_LIST:+-e NVSHMEM_HCA_LIST="$NVSHMEM_HCA_LIST"} \
    ${NVSHMEM_HCA_PE_MAPPING:+-e NVSHMEM_HCA_PE_MAPPING="$NVSHMEM_HCA_PE_MAPPING"} \
    ${NEMO_BASE_MODEL_LOAD_LAYER_CHUNK:+-e NEMO_BASE_MODEL_LOAD_LAYER_CHUNK="$NEMO_BASE_MODEL_LOAD_LAYER_CHUNK"} \
    ${MEMORY_PROFILE:+-e MEMORY_PROFILE="$MEMORY_PROFILE"} \
    ${MEMORY_PROFILE_STEP:+-e MEMORY_PROFILE_STEP="$MEMORY_PROFILE_STEP"} \
    ${MEMORY_SNAPSHOT_DIR:+-e MEMORY_SNAPSHOT_DIR="$MEMORY_SNAPSHOT_DIR"} \
    ${TORCH_PROFILE:+-e TORCH_PROFILE="$TORCH_PROFILE"} \
    ${TORCH_PROFILE_DIR:+-e TORCH_PROFILE_DIR="$TORCH_PROFILE_DIR"} \
    ${TORCH_PROFILE_WAIT:+-e TORCH_PROFILE_WAIT="$TORCH_PROFILE_WAIT"} \
    ${TORCH_PROFILE_WARMUP:+-e TORCH_PROFILE_WARMUP="$TORCH_PROFILE_WARMUP"} \
    ${TORCH_PROFILE_ACTIVE:+-e TORCH_PROFILE_ACTIVE="$TORCH_PROFILE_ACTIVE"} \
    -e HF_HUB_OFFLINE=1 \
    -e HF_DATASETS_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    ${HF_CACHE_DIR:+-e HF_HOME="$HF_CACHE_DIR"} \
    -v "$REPO_DIR:/opt/Automodel" \
    -v "$MODEL_PATH:$MODEL_PATH:ro" \
    -v "$(dirname "$CKPT_DIR"):$(dirname "$CKPT_DIR")" \
    ${HF_CACHE_DIR:+-v "$HF_CACHE_DIR:$HF_CACHE_DIR"} \
    -w /opt/Automodel \
    "$IMAGE" \
    bash -c "$IN_CONTAINER_CMD"
