#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
GPUS="${GPUS:?Set GPUS to physical GPU indices}"
OUTER_SEEDS="${OUTER_SEEDS:-2020 2021 2022 2023 2024}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-aaai27_outer_seed_fixed_split42_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/aaai27_outer_seed_fixed_split42_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-$SERVER_ROOT/logs/aaai27_outer_seed_fixed_split42_${TIMESTAMP}}"
UNIMODAL_CACHE_ROOT="${UNIMODAL_CACHE_ROOT:-$SERVER_ROOT/feature_cache/aaai27_binary_remaining_ablation_seed2022}"
MULTIMODAL_CACHE_ROOT="${MULTIMODAL_CACHE_ROOT:-$SERVER_ROOT/feature_cache/aaai27_binary_activity_graph_clip_h14_mantis8m}"
SPLIT_REFERENCE="${SPLIT_REFERENCE:-$DATA_DIR/med_data/AAAI_Data/data_loading/split_reference_seed42.csv}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-2048}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
STATUS_DIR="$LOG_DIR/status"

read -r -a GPU_VALUES <<< "$GPUS"
NUM_WORKERS=${#GPU_VALUES[@]}
read -r -a SEED_VALUES <<< "$OUTER_SEEDS"
if (( ${#SEED_VALUES[@]} < 2 )); then
  echo "At least two outer seeds are required: $OUTER_SEEDS" >&2
  exit 1
fi
if (( NUM_WORKERS < 1 || NUM_WORKERS > 8 )); then
  echo "GPUS must contain between one and eight GPU indices: $GPUS" >&2
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

mkdir -p "$RESULT_ROOT" "$LOG_DIR" "$STATUS_DIR" "$UNIMODAL_CACHE_ROOT" "$MULTIMODAL_CACHE_ROOT"

for worker in "${!GPU_VALUES[@]}"; do
  gpu=${GPU_VALUES[$worker]}
  log_file="$LOG_DIR/gpu${gpu}.log"
  status_file="$STATUS_DIR/worker_${worker}.status"
  worker_command="cd '$PROJECT_DIR'; set -o pipefail; SERVER_ROOT='$SERVER_ROOT' PROJECT_DIR='$PROJECT_DIR' MODEL_DIR='$MODEL_DIR' MANTIS_DIR='$MANTIS_DIR' DATA_DIR='$DATA_DIR' PYTHON_BIN='$PYTHON_BIN' RESULT_ROOT='$RESULT_ROOT' UNIMODAL_CACHE_ROOT='$UNIMODAL_CACHE_ROOT' MULTIMODAL_CACHE_ROOT='$MULTIMODAL_CACHE_ROOT' SPLIT_REFERENCE='$SPLIT_REFERENCE' OUTER_SEEDS='$OUTER_SEEDS' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' PHYSICAL_GPU='$gpu' WAIT_FOR_GPU_FREE='$WAIT_FOR_GPU_FREE' GPU_FREE_MEMORY_MAX_MB='$GPU_FREE_MEMORY_MAX_MB' GPU_POLL_SECONDS='$GPU_POLL_SECONDS' bash scripts/run_aaai27_outer_seed_worker.sh 2>&1 | tee -a '$log_file'; status=\${PIPESTATUS[0]}; printf '%s\n' \"\$status\" > '$status_file'; exit \"\$status\""
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$worker_command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$worker_command"
  fi
done

summary_command="cd '$PROJECT_DIR'; '$PYTHON_BIN' scripts/aggregate_aaai27_outer_seeds.py --result-root '$RESULT_ROOT' --status-dir '$STATUS_DIR' --workers '$NUM_WORKERS' --outer-seeds $OUTER_SEEDS --split-reference '$SPLIT_REFERENCE'"
tmux new-window -d -t "$SESSION" -n summary "$summary_command"

cat <<EOF
Started tmux session: $SESSION
GPUs: $GPUS
Outer training seeds: $OUTER_SEEDS
Fixed inner subject split: seed 42 ($SPLIT_REFERENCE)
Datasets: PADS 9/10, Shimmer 11/12
Unique conditions: 7 (representation, feature extraction, fusion ablations)
Total training runs: $((${#SEED_VALUES[@]} * 4 * 7))
Results: $RESULT_ROOT
Logs: $LOG_DIR
Mean/std summaries are generated automatically after all workers finish.
EOF
