#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
GPUS="${GPUS:?Set GPUS to available physical GPU indices}"
SEED="${SEED:-2022}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-aaai27_remaining_ablation_seed2022_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/aaai27_remaining_ablation_seed2022_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-$SERVER_ROOT/logs/aaai27_remaining_ablation_seed2022_${TIMESTAMP}}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-$SERVER_ROOT/feature_cache/aaai27_binary_remaining_ablation_seed2022}"
MULTIMODAL_CACHE_ROOT="${MULTIMODAL_CACHE_ROOT:-$SERVER_ROOT/feature_cache/aaai27_binary_activity_graph_clip_h14_mantis8m}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-1}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-512}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
STATUS_DIR="$LOG_DIR/status"
ALL_SUMMARY="$RESULT_ROOT/all_conditions_summary.csv"
REPRESENTATION_SUMMARY="$RESULT_ROOT/activity_graph_ablation_summary.csv"
FEATURE_SUMMARY="$RESULT_ROOT/multimodal_feature_ablation_summary.csv"

if [[ "$SEED" != "2022" ]]; then
  echo "AAAI27 binary ablation requires SEED=2022; got $SEED" >&2
  exit 1
fi
read -r -a GPU_VALUES <<< "$GPUS"
NUM_WORKERS=${#GPU_VALUES[@]}
if (( NUM_WORKERS < 1 || NUM_WORKERS > 5 )); then
  echo "GPUS must list between one and five physical GPU indices" >&2
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
  worker_command="cd '$PROJECT_DIR'; set -o pipefail; SERVER_ROOT='$SERVER_ROOT' PROJECT_DIR='$PROJECT_DIR' MODEL_DIR='$MODEL_DIR' MANTIS_DIR='$MANTIS_DIR' DATA_DIR='$DATA_DIR' PYTHON_BIN='$PYTHON_BIN' RESULT_ROOT='$RESULT_ROOT' FEATURE_CACHE_ROOT='$FEATURE_CACHE_ROOT' MULTIMODAL_CACHE_ROOT='$MULTIMODAL_CACHE_ROOT' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' PHYSICAL_GPU='$gpu' SEED='$SEED' WAIT_FOR_GPU_FREE='$WAIT_FOR_GPU_FREE' GPU_FREE_MEMORY_MAX_MB='$GPU_FREE_MEMORY_MAX_MB' GPU_POLL_SECONDS='$GPU_POLL_SECONDS' bash scripts/run_aaai27_remaining_ablation_worker.sh 2>&1 | tee -a '$log_file'; status=\${PIPESTATUS[0]}; printf '%s\n' \"\$status\" > '$status_file'; exit \"\$status\""
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$worker_command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$worker_command"
  fi
done

summary_command="cd '$PROJECT_DIR'; '$PYTHON_BIN' scripts/aggregate_aaai27_remaining_ablation.py --result-root '$RESULT_ROOT' --status-dir '$STATUS_DIR' --workers '$NUM_WORKERS' --output-all '$ALL_SUMMARY' --output-representation '$REPRESENTATION_SUMMARY' --output-feature '$FEATURE_SUMMARY'"
tmux new-window -d -t "$SESSION" -n summary "$summary_command"

cat <<EOF
Started tmux session: $SESSION
GPUs: $GPUS
Seed: $SEED
Datasets: PADS 9/10, Shimmer 11/12
Conditions: vision_line_plot vision_activity_graph timeseries_mantis multimodal_concat
Training: balanced class weights, 128-d small-sample head, 40 epochs, patience 8
Results: $RESULT_ROOT
Logs: $LOG_DIR
Activity Graph summary: $REPRESENTATION_SUMMARY
Feature extraction summary: $FEATURE_SUMMARY
EOF
