#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
GPUS="${GPUS:-3 4}"
REPEATS="${REPEATS:-3}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-tivit_seed2022_3repeat_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/main_seed2022_3repeat_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-$SERVER_ROOT/logs/main_seed2022_3repeat_${TIMESTAMP}}"
STATUS_DIR="$LOG_DIR/status"
SUMMARY_FILE="$RESULT_ROOT/three_repeat_average.csv"

read -r -a GPU_VALUES <<< "$GPUS"
NUM_WORKERS=${#GPU_VALUES[@]}
if (( NUM_WORKERS < 1 || NUM_WORKERS > 5 )); then
  echo "GPUS must list between one and five GPU indices: $GPUS" >&2
  exit 1
fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer: $REPEATS" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT" "$LOG_DIR" "$STATUS_DIR"

for worker in "${!GPU_VALUES[@]}"; do
  gpu=${GPU_VALUES[$worker]}
  log_file="$LOG_DIR/gpu${gpu}.log"
  status_file="$STATUS_DIR/worker_${worker}.status"
  worker_command="cd '$PROJECT_DIR'; set -o pipefail; PROJECT_DIR='$PROJECT_DIR' RESULT_ROOT='$RESULT_ROOT' REPEATS='$REPEATS' CUDA_VISIBLE_DEVICES='$gpu' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' bash scripts/run_seed2022_repeat_worker.sh 2>&1 | tee -a '$log_file'; status=\${PIPESTATUS[0]}; printf '%s\\n' \"\$status\" > '$status_file'; exit \"\$status\""
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$worker_command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$worker_command"
  fi
done

summary_command="cd '$PROJECT_DIR'; python3 scripts/aggregate_seed2022_repeats.py --result-root '$RESULT_ROOT' --repeats '$REPEATS' --status-dir '$STATUS_DIR' --workers '$NUM_WORKERS' --output '$SUMMARY_FILE'"
tmux new-window -d -t "$SESSION" -n summary "$summary_command"

cat <<EOF
Started tmux session: $SESSION
GPUs: $GPUS
Seed: 2022
Repeats: $REPEATS
Datasets: Feng FallTL UCIHAR
Results: $RESULT_ROOT
Logs: $LOG_DIR
Automatic summary: $SUMMARY_FILE
EOF
