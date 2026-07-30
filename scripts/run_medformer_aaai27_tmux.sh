#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension/Tivit_Extension-main}"
SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
MEDFORMER_ROOT="${MEDFORMER_ROOT:-/home/xuzheyuan/guoyin/Medformer}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/medformer/bin/python}"
GPUS="${GPUS:?Set GPUS to available physical GPU indices}"
SEED="${SEED:-42}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-medformer_aaai27_binary_seed42_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/medformer_aaai27_binary_seed42_${TIMESTAMP}}"
LOG_ROOT="${LOG_ROOT:-$SERVER_ROOT/logs/medformer_aaai27_binary_seed42_${TIMESTAMP}}"
TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-1}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-512}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

read -r -a GPU_VALUES <<< "$GPUS"
NUM_WORKERS=${#GPU_VALUES[@]}
if (( NUM_WORKERS < 1 )); then
  echo "At least one GPU is required" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

for worker in "${!GPU_VALUES[@]}"; do
  gpu=${GPU_VALUES[$worker]}
  log_file="$LOG_ROOT/gpu${gpu}.log"
  worker_command="cd '$PROJECT_ROOT'; set -o pipefail; PROJECT_ROOT='$PROJECT_ROOT' MEDFORMER_ROOT='$MEDFORMER_ROOT' DATA_DIR='$DATA_DIR' PYTHON_BIN='$PYTHON_BIN' RESULT_ROOT='$RESULT_ROOT' LOG_ROOT='$LOG_ROOT' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' PHYSICAL_GPU='$gpu' SEED='$SEED' TARGET_SEQ_LEN='$TARGET_SEQ_LEN' BATCH_SIZE='$BATCH_SIZE' TRAIN_EPOCHS='$TRAIN_EPOCHS' PATIENCE='$PATIENCE' WAIT_FOR_GPU_FREE='$WAIT_FOR_GPU_FREE' GPU_FREE_MEMORY_MAX_MB='$GPU_FREE_MEMORY_MAX_MB' GPU_POLL_SECONDS='$GPU_POLL_SECONDS' bash scripts/run_medformer_aaai27_worker.sh 2>&1 | tee -a '$log_file'; status=\${PIPESTATUS[0]}; printf '%s\n' \"\$status\" > '$LOG_ROOT/worker_${worker}.status'; exit \"\$status\""
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$worker_command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$worker_command"
  fi
done

summary_command="while [ \$(find '$LOG_ROOT' -maxdepth 1 -name 'worker_*.status' | wc -l) -lt '$NUM_WORKERS' ]; do sleep 15; done; cd '$PROJECT_ROOT'; '$PYTHON_BIN' scripts/aggregate_medformer_aaai27.py --result-root '$RESULT_ROOT' --output '$RESULT_ROOT/summary.csv'"
tmux new-window -d -t "$SESSION" -n summary "$summary_command"

cat <<EOF
Started tmux session: $SESSION
GPUs: $GPUS
Seed: $SEED
Datasets: PADS_09_task06_DrinkGlas, Shimmer_11_session11_DRINK
Models: Autoformer Transformer PatchTST Medformer Crossformer FEDformer
Settings: d_model=128, e_layers=6, d_ff=256, lr=1e-4, epochs=$TRAIN_EPOCHS, patience=$PATIENCE, SWA, balanced loss
Input: linear resize to ${TARGET_SEQ_LEN} time steps; batch_size=$BATCH_SIZE
Results: $RESULT_ROOT
Logs: $LOG_ROOT
Summary: $RESULT_ROOT/summary.csv
EOF
