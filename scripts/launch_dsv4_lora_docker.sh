#!/usr/bin/env bash
# launch_dsv4_docker.sh — DSv4-Flash launch via the NGC nemo-automodel container.
# Single-node default; supports 2-node via env vars (no Slurm needed).
#
# The container ships TE, deep_ep, mamba-ssm, causal-conv1d, etc. pre-built,
# so no bare-metal CUDA / cuDNN / pynvml dance is needed.  We mount the local
# fork over /opt/Automodel so any local edits (e.g. the lazy DeepEP buffer
# OOM fix) take effect.
#
# Usage (single node):
#   bash scripts/launch_dsv4_docker.sh
#
# Usage (2 nodes — same command on both, only NODE_RANK differs):
#   # node0:
#   NODE_RANK=0 RDZV_ENDPOINT=<node0-ip>:29500 \
#       bash scripts/launch_dsv4_docker.sh
#   # node1:
#   NODE_RANK=1 RDZV_ENDPOINT=<node0-ip>:29500 \
#       bash scripts/launch_dsv4_docker.sh
#
# Required (on each node):
#   - Docker with the nvidia container runtime (`docker info | grep Runtimes`
#     must list `nvidia`).
#   - The NGC image already pulled:
#       docker pull nvcr.io/nvidia/nemo-automodel:26.04
#   - DSv4-Flash safetensors at $MODEL_PATH (same path on both nodes).
#   - $CKPT_DIR writable.
#
# Env knobs (with defaults):
#   IMAGE       nvcr.io/nvidia/nemo-automodel:26.04
#   REPO_DIR    $HOME/excepshenal/Automodel   (bind-mounted to /opt/Automodel)
#   MODEL_PATH  /raid0/data/models/DeepSeek-V4-Flash
#   CKPT_DIR    /external-disk/deepseek_v4_flash_hellaswag
#   NUM_LAYERS  0   (0 = use yaml default = 43, the full Flash model;
#                    set to a smaller integer to cut layers for an OOM-safe smoke)
#   EP_SIZE     8
#   PP_SIZE     1
#   GBS         8
#   LBS         1
#   ACT_CKPT    true
#   NNODES      1
#   NODE_RANK   0
#   RDZV_ENDPOINT  127.0.0.1:29500   (override to <node0-ip>:29500 for 2-node)
#   EXTRA_ARGS  ""  appended to the automodel CLI call

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-automodel:26.04}"
REPO_DIR="${REPO_DIR:-$HOME/excepshenal/Automodel}"
MODEL_PATH="${MODEL_PATH:-/raid0/data/models/DeepSeek-V4-Flash-BF16}"
CKPT_DIR="${CKPT_DIR:-/external-disk/DeepSeek-V4-Flash-hellaswag-dshen-run-1}"
# HF_CACHE is intentionally not used. Datasets are loaded from local paths
# under /external-disk/data/dataset, and model weights/tokenizer from
# /raid0/data/models. Offline mode env vars below prevent any Hub fallback.
NUM_LAYERS="${NUM_LAYERS:-0}"
EP_SIZE="${EP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-1}"
GBS="${GBS:-8}"
LBS="${LBS:-1}"
ACT_CKPT="${ACT_CKPT:-true}"
# tilelang for long-seq production. IN_CONTAINER_CMD pip-installs the deps.
ATTN_BACKEND="${ATTN_BACKEND:-tilelang}"
# YAML defaults dispatcher: deepep, which requires NVSHMEM + IBGDA. If your
# fabric/container lacks IBGDA support, switch to `torch` (standard NCCL
# all-to-all). Slower at scale but works anywhere. Known limitation: ckpt
# resume of optimizer state misses gate.weight under EP -- see PR #2092.
DISPATCHER="${DISPATCHER:-deepep}"
# Match the dispatcher's expected experts impl (gmm for deepep, torch_mm otherwise).
EXPERTS="${EXPERTS:-}"
if [[ -z "$EXPERTS" ]]; then
    case "$DISPATCHER" in
        deepep) EXPERTS="gmm" ;;
        *)      EXPERTS="torch_mm" ;;
    esac
fi
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-127.0.0.1:29500}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- preflight on the host ---------------------------------------------------
[[ -d "$REPO_DIR" ]] || { echo "FATAL: REPO_DIR not found: $REPO_DIR" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "FATAL: MODEL_PATH not found: $MODEL_PATH" >&2; exit 1; }
mkdir -p "$CKPT_DIR"

# Make sure the lazy DeepEP buffer OOM patch is present in the mounted repo —
# without it, single-node DSv4-Flash OOMs at _aggregate_experts during load.
# The patch removes the eager _init_deepep_buffer call from init_token_dispatcher
# (introduced in #2076 / e42584e3) and reverts to lazy allocation via
# FusedDispatch.forward.
if grep -q "if self.dispatcher_backend == \"deepep\":" "$REPO_DIR/nemo_automodel/components/moe/experts.py" \
   && grep -A1 "if self.dispatcher_backend == \"deepep\":" "$REPO_DIR/nemo_automodel/components/moe/experts.py" \
        | grep -q "self._init_deepep_buffer(ep_group)"; then
    echo "WARN: DSv4 eager DeepEP buffer init still active in $REPO_DIR." >&2
    echo "      Single-node will OOM at load. Apply the OOM patch (commit 6d0cb54b) or use 2+ nodes." >&2
fi

LAYER_OVERRIDE=""
if [[ "$NUM_LAYERS" != "0" ]]; then
    LAYER_OVERRIDE="--model.config.num_hidden_layers $NUM_LAYERS"
fi

# --- the in-container command -----------------------------------------------
# We launch /opt/venv/bin/torchrun (the venv that ships with the container),
# not the system python — the venv has the canonical TE / deep_ep / cuDNN paths
# wired up.  --network=host means RDZV_ENDPOINT addresses resolve against the
# host's interface table, which is what makes the 2-node form work without
# extra port-forwarding.
read -r -d '' IN_CONTAINER_CMD <<EOF || true
set -euo pipefail
cd /opt/Automodel
echo "[container] image=$IMAGE node_rank=$NODE_RANK / $NNODES rdzv=$RDZV_ENDPOINT"
echo "[container] python: \$(python -V 2>&1)"
echo "[container] torch:  \$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
if ! python -c 'import tilelang, tile_kernels' 2>/dev/null; then
    # Pin apache-tvm-ffi: tilelang 0.1.10 declares apache-tvm-ffi>=0.1.10,~=0.1.0,
    # so an unpinned install now resolves to 0.1.12, which double-registers TVM
    # FFI type index 130 against tilelang's vendored runtime and aborts at import
    # (SIGABRT on all ranks). 0.1.11 is the highest in-spec version that imports
    # cleanly; this is what the original 4,970/7,740 baseline effectively used.
    pip install --quiet tilelang==0.1.10 tile_kernels==1.0.0 apache-tvm-ffi==0.1.11
fi
exec torchrun \\
    --nnodes=$NNODES \\
    --nproc-per-node=8 \\
    --node-rank=$NODE_RANK \\
    --master-addr=${RDZV_ENDPOINT%:*} \\
    --master-port=${RDZV_ENDPOINT#*:} \\
    -m nemo_automodel.cli.app \\
    examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag_lora.yaml \\
    --model.config.pretrained_model_name_or_path $MODEL_PATH \\
    --model.config.name_or_path $MODEL_PATH \\
    $LAYER_OVERRIDE \\
    --dataset.tokenizer.pretrained_model_name_or_path $MODEL_PATH \\
    --validation_dataset.tokenizer.pretrained_model_name_or_path $MODEL_PATH \\
    --checkpoint.enabled true \\
    --checkpoint.checkpoint_dir $CKPT_DIR \\
    --checkpoint.model_save_format safetensors \\
    --checkpoint.save_consolidated true \\
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
    ${NVSHMEM_REMOTE_TRANSPORT:+-e NVSHMEM_REMOTE_TRANSPORT="$NVSHMEM_REMOTE_TRANSPORT"} \
    ${NVSHMEM_DEBUG:+-e NVSHMEM_DEBUG="$NVSHMEM_DEBUG"} \
    ${NVSHMEM_DISABLE_CUDA_VMM:+-e NVSHMEM_DISABLE_CUDA_VMM="$NVSHMEM_DISABLE_CUDA_VMM"} \
    ${NVSHMEM_HCA_LIST:+-e NVSHMEM_HCA_LIST="$NVSHMEM_HCA_LIST"} \
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
    -v "$REPO_DIR:/opt/Automodel" \
    -v "$MODEL_PATH:$MODEL_PATH:ro" \
    -v "$(dirname "$CKPT_DIR"):$(dirname "$CKPT_DIR")" \
    -w /opt/Automodel \
    "$IMAGE" \
    bash -c "$IN_CONTAINER_CMD"
