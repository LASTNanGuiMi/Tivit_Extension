#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
GPUS="${GPUS:?Set GPUS to one to four physical GPU indices.}"
SEED="${SEED:-2022}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-aaai27_binary_seed2022_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/aaai27_binary_seed2022_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-$SERVER_ROOT/logs/aaai27_binary_seed2022_${TIMESTAMP}}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-$SERVER_ROOT/feature_cache/aaai27_binary_activity_graph_clip_h14_mantis8m}"
STATUS_DIR="$LOG_DIR/status"
SUMMARY_FILE="$RESULT_ROOT/summary.csv"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-512}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

if [[ "$SEED" != "2022" ]]; then
  echo "AAAI27 binary queue requires SEED=2022; got: $SEED" >&2
  exit 1
fi

read -r -a GPU_VALUES <<< "$GPUS"
NUM_WORKERS=${#GPU_VALUES[@]}
if (( NUM_WORKERS < 1 || NUM_WORKERS > 4 )); then
  echo "GPUS must contain between one and four GPU indices: $GPUS" >&2
  exit 1
fi

declare -A SEEN_GPUS=()
for gpu in "${GPU_VALUES[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $gpu" >&2; exit 1; }
  [[ -z "${SEEN_GPUS[$gpu]+x}" ]] || { echo "Duplicate GPU index: $gpu" >&2; exit 1; }
  SEEN_GPUS[$gpu]=1
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT" "$LOG_DIR" "$STATUS_DIR" "$FEATURE_CACHE_ROOT"

for worker in "${!GPU_VALUES[@]}"; do
  gpu=${GPU_VALUES[$worker]}
  log_file="$LOG_DIR/gpu${gpu}.log"
  status_file="$STATUS_DIR/worker_${worker}.status"
  worker_command="cd '$PROJECT_DIR'; set -o pipefail; SERVER_ROOT='$SERVER_ROOT' PROJECT_DIR='$PROJECT_DIR' MODEL_DIR='$MODEL_DIR' MANTIS_DIR='$MANTIS_DIR' DATA_DIR='$DATA_DIR' PYTHON_BIN='$PYTHON_BIN' RESULT_ROOT='$RESULT_ROOT' FEATURE_CACHE_ROOT='$FEATURE_CACHE_ROOT' SEED='$SEED' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' PHYSICAL_GPU='$gpu' CUDA_VISIBLE_DEVICES='$gpu' WAIT_FOR_GPU_FREE='$WAIT_FOR_GPU_FREE' GPU_FREE_MEMORY_MAX_MB='$GPU_FREE_MEMORY_MAX_MB' GPU_POLL_SECONDS='$GPU_POLL_SECONDS' bash scripts/run_aaai27_binary_worker.sh 2>&1 | tee -a '$log_file'; status=\${PIPESTATUS[0]}; printf '%s\n' \"\$status\" > '$status_file'; exit \"\$status\""
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$worker_command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$worker_command"
  fi
done

summary_command="cd '$PROJECT_DIR'; '$PYTHON_BIN' scripts/aggregate_aaai27_binary.py --result-root '$RESULT_ROOT' --status-dir '$STATUS_DIR' --workers '$NUM_WORKERS' --output '$SUMMARY_FILE'"
tmux new-window -d -t "$SESSION" -n summary "$summary_command"

cat <<EOF
Started tmux session: $SESSION
GPUs: $GPUS
Seed: $SEED
Datasets: PADS 9/10, Shimmer 11/12
Labels: original 0 -> 0; original 1/2 -> 1
Fusion modes: concat concat_attn cross_attn_gate masked_pretrain
Training: balanced class weights, 128-d small-sample head, 40 epochs, patience 8
Feature cache: $FEATURE_CACHE_ROOT
Wait for free GPUs: $WAIT_FOR_GPU_FREE (memory threshold ${GPU_FREE_MEMORY_MAX_MB} MiB)
Results: $RESULT_ROOT
Logs: $LOG_DIR
Automatic summary: $SUMMARY_FILE
EOF
