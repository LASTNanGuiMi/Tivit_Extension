#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension/Tivit_Extension-main}"
MEDFORMER_ROOT="${MEDFORMER_ROOT:-/home/xuzheyuan/guoyin/Medformer}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/medformer/bin/python}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT}"
LOG_ROOT="${LOG_ROOT:?Set LOG_ROOT}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS}"
PHYSICAL_GPU="${PHYSICAL_GPU:?Set PHYSICAL_GPU}"
SEED="${SEED:-42}"
TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-1}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-512}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

DATASETS=(PADS_09_task06_DrinkGlas Shimmer_11_session11_DRINK)
MODELS=(Autoformer Transformer PatchTST Medformer Crossformer FEDformer)

if [[ "$SEED" != "42" ]]; then
  echo "Medformer comparison requires SEED=42; got $SEED" >&2
  exit 1
fi

wait_for_gpu() {
  [[ "$WAIT_FOR_GPU_FREE" == "1" ]] || return 0
  while true; do
    used=$(nvidia-smi -i "$PHYSICAL_GPU" \
      --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)
    used=${used// /}
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= GPU_FREE_MEMORY_MAX_MB )); then
      echo "GPU $PHYSICAL_GPU is available (${used} MiB used)."
      return 0
    fi
    echo "Waiting for GPU $PHYSICAL_GPU (used=${used:-unknown} MiB)."
    sleep "$GPU_POLL_SECONDS"
  done
}

run_job() {
  local dataset=$1
  local model=$2
  local job_log="$LOG_ROOT/${dataset}_${model}.log"
  local metrics="$RESULT_ROOT/$dataset/$model/metrics.json"
  if [[ -s "$metrics" ]]; then
    echo "Skip completed | $dataset | $model" | tee -a "$job_log"
    return 0
  fi
  echo "Run | worker=$WORKER_ID/$NUM_WORKERS | gpu=$PHYSICAL_GPU | $dataset | $model" \
    | tee -a "$job_log"
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" "$PYTHON_BIN" -u \
    "$PROJECT_ROOT/scripts/run_medformer_aaai27_binary.py" \
    --dataset "$dataset" \
    --model "$model" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --target-seq-len "$TARGET_SEQ_LEN" \
    --train-epochs "$TRAIN_EPOCHS" \
    --patience "$PATIENCE" \
    --data-dir "$DATA_DIR" \
    --medformer-root "$MEDFORMER_ROOT" \
    --result-root "$RESULT_ROOT" 2>&1 | tee -a "$job_log"
}

wait_for_gpu
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/medformer_aaai27_mpl_gpu_${PHYSICAL_GPU}}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
job_index=0
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    if (( job_index % NUM_WORKERS != WORKER_ID )); then
      ((job_index += 1))
      continue
    fi
    run_job "$dataset" "$model"
    ((job_index += 1))
  done
done
