#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
GPUS="${GPUS:-0 1 2 3 4}"
SEED="${SEED:-2022}"
REPEATS="${REPEATS:-3}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-har6_seed2022_3repeat_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/har6_seed2022_3repeat_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-$SERVER_ROOT/logs/har6_seed2022_3repeat_${TIMESTAMP}}"
STATUS_DIR="$LOG_DIR/status"
SUMMARY_FILE="$RESULT_ROOT/three_repeat_average.csv"

if [[ "$SEED" != "2022" || "$REPEATS" != "3" ]]; then
  echo "This queue requires SEED=2022 and REPEATS=3; got $SEED/$REPEATS." >&2
  exit 1
fi

read -r -a GPU_VALUES <<< "$GPUS"
NUM_WORKERS=${#GPU_VALUES[@]}
if (( NUM_WORKERS < 1 || NUM_WORKERS > 5 )); then
  echo "GPUS must list between one and five GPU indices: $GPUS" >&2
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

mkdir -p "$RESULT_ROOT" "$LOG_DIR" "$STATUS_DIR"

for worker in "${!GPU_VALUES[@]}"; do
  gpu=${GPU_VALUES[$worker]}
  log_file="$LOG_DIR/gpu${gpu}.log"
  status_file="$STATUS_DIR/worker_${worker}.status"
  worker_command="cd '$PROJECT_DIR'; set -o pipefail; SERVER_ROOT='$SERVER_ROOT' PROJECT_DIR='$PROJECT_DIR' RESULT_ROOT='$RESULT_ROOT' REPEATS='$REPEATS' SEED='$SEED' CUDA_VISIBLE_DEVICES='$gpu' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' bash scripts/run_har6_worker.sh 2>&1 | tee -a '$log_file'; status=\${PIPESTATUS[0]}; printf '%s\\n' \"\$status\" > '$status_file'; exit \"\$status\""
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$worker_command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$worker_command"
  fi
done

summary_command="cd '$PROJECT_DIR'; python3 scripts/aggregate_har6_repeats.py --result-root '$RESULT_ROOT' --repeats '$REPEATS' --status-dir '$STATUS_DIR' --workers '$NUM_WORKERS' --output '$SUMMARY_FILE'"
tmux new-window -d -t "$SESSION" -n summary "$summary_command"

cat <<EOF
Started tmux session: $SESSION
GPUs: $GPUS
Seed: $SEED
Repeats: $REPEATS
Datasets: FLAAP UCIHAR
Channels: Acc XYZ + Gyro XYZ (UCI uses total_acc + body_gyro)
Fusion modes: concat concat_attn cross_attn_gate masked_pretrain
Total tasks: 24
Results: $RESULT_ROOT
Logs: $LOG_DIR
Automatic summary: $SUMMARY_FILE
EOF
