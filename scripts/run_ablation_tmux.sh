#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=ablation_config.sh
source "$SCRIPT_DIR/ablation_config.sh"

GPUS="${GPUS:?Set GPUS to one to five available GPU indices.}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-tivit_ablation_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/ablation_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-$SERVER_ROOT/logs/ablation_${TIMESTAMP}}"
STATUS_DIR="$LOG_DIR/status"
SUMMARY_FILE="$RESULT_ROOT/ablation_summary.csv"

if [[ "$SEEDS" != "$FIXED_ABLATION_SEED" ]]; then
  echo "Feature ablation requires SEEDS=$FIXED_ABLATION_SEED; got: $SEEDS" >&2
  exit 1
fi

read -r -a GPU_VALUES <<< "$GPUS"
NUM_WORKERS=${#GPU_VALUES[@]}
if (( NUM_WORKERS < 1 || NUM_WORKERS > 5 )); then
  echo "GPUS must list between one and five indices: $GPUS" >&2
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
  worker_command="cd '$PROJECT_DIR'; set -o pipefail; SERVER_ROOT='$SERVER_ROOT' PROJECT_DIR='$PROJECT_DIR' MODEL_DIR='$MODEL_DIR' MANTIS_DIR='$MANTIS_DIR' DATA_DIR='$DATA_DIR' PYTHON_BIN='$PYTHON_BIN' SEEDS='$SEEDS' REPEATS='$REPEATS' DATASET_GROUPS='$DATASET_GROUPS' ABLATION_CONDITIONS='$ABLATION_CONDITIONS' PROPOSED_FUSION='$PROPOSED_FUSION' FENG_MAX_WINDOWS='$FENG_MAX_WINDOWS' FALLTL_MAX_WINDOWS='$FALLTL_MAX_WINDOWS' WINDOW_SIZE='$WINDOW_SIZE' WINDOW_STRIDE='$WINDOW_STRIDE' CUSTOM_TEST_RATIO='$CUSTOM_TEST_RATIO' VAL_RATIO='$VAL_RATIO' FUSION_DIM='$FUSION_DIM' FUSION_HEADS='$FUSION_HEADS' CROSS_ATTN_QUERY='$CROSS_ATTN_QUERY' MASK_PROB='$MASK_PROB' PRETRAIN_EPOCHS='$PRETRAIN_EPOCHS' MLP_HIDDEN_DIM='$MLP_HIDDEN_DIM' MLP_NUM_LAYERS='$MLP_NUM_LAYERS' MLP_DROPOUT='$MLP_DROPOUT' MLP_LR='$MLP_LR' MLP_WEIGHT_DECAY='$MLP_WEIGHT_DECAY' MLP_EPOCHS='$MLP_EPOCHS' MLP_PATIENCE='$MLP_PATIENCE' BATCH_SIZE='$BATCH_SIZE' RESULT_ROOT='$RESULT_ROOT' CUDA_VISIBLE_DEVICES='$gpu' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' bash scripts/run_ablation_worker.sh 2>&1 | tee -a '$log_file'; status=\${PIPESTATUS[0]}; printf '%s\\n' \"\$status\" > '$status_file'; exit \"\$status\""
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$worker_command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$worker_command"
  fi
done

summary_command="cd '$PROJECT_DIR'; python3 scripts/aggregate_ablation.py --result-root '$RESULT_ROOT' --seeds $SEEDS --repeats '$REPEATS' --datasets $DATASET_GROUPS --conditions $ABLATION_CONDITIONS --status-dir '$STATUS_DIR' --workers '$NUM_WORKERS' --output '$SUMMARY_FILE'"
tmux new-window -d -t "$SESSION" -n summary "$summary_command"

cat <<EOF
Started tmux session: $SESSION
GPUs: $GPUS
Seeds: $SEEDS
Repeats per seed: $REPEATS
Datasets: $DATASET_GROUPS
Conditions: $ABLATION_CONDITIONS
Proposed fusion: $PROPOSED_FUSION
Results: $RESULT_ROOT
Logs: $LOG_DIR
Automatic summary: $SUMMARY_FILE
EOF
