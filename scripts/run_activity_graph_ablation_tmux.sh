#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
GPUS="${GPUS:?Set GPUS to the available GPU indices.}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
REPEATS="${REPEATS:-3}"
FIXED_ABLATION_SEED=2022
SEED="${SEED:-$FIXED_ABLATION_SEED}"
ABLATION_KIND="${ABLATION_KIND:-image}"

if [[ "$SEED" != "$FIXED_ABLATION_SEED" ]]; then
  echo "Activity-graph ablation requires SEED=$FIXED_ABLATION_SEED; got: $SEED" >&2
  exit 1
fi

case "$ABLATION_KIND" in
  image) RUN_NAME=activity_graph_ablation ;;
  fusion) RUN_NAME=multimodal_fusion_ablation ;;
  *) echo "Invalid ABLATION_KIND: $ABLATION_KIND" >&2; exit 1 ;;
esac

SESSION="${SESSION:-${RUN_NAME}_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/${RUN_NAME}_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-$SERVER_ROOT/logs/${RUN_NAME}_${TIMESTAMP}}"

read -r -a GPU_VALUES <<< "$GPUS"
NUM_WORKERS=${#GPU_VALUES[@]}
(( NUM_WORKERS > 0 )) || { echo "No GPUs supplied." >&2; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session exists: $SESSION" >&2; exit 1; }
mkdir -p "$RESULT_ROOT" "$LOG_DIR"

for worker in "${!GPU_VALUES[@]}"; do
  gpu=${GPU_VALUES[$worker]}
  log_file="$LOG_DIR/gpu${gpu}.log"
  command="cd '$PROJECT_DIR'; set -o pipefail; SERVER_ROOT='$SERVER_ROOT' PROJECT_DIR='$PROJECT_DIR' RESULT_ROOT='$RESULT_ROOT' REPEATS='$REPEATS' SEED='$SEED' ABLATION_KIND='$ABLATION_KIND' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' CUDA_VISIBLE_DEVICES='$gpu' bash scripts/run_activity_graph_ablation_worker.sh 2>&1 | tee -a '$log_file'"
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$command"
  fi
done

printf 'Session: %s\nGPUs: %s\nSeed/repeats: %s / %s\nResults: %s\nLogs: %s\n' \
  "$SESSION" "$GPUS" "$SEED" "$REPEATS" "$RESULT_ROOT" "$LOG_DIR"
