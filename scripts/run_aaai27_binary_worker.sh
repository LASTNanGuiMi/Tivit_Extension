#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT.}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:?Set FEATURE_CACHE_ROOT.}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID.}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS.}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
PHYSICAL_GPU="${PHYSICAL_GPU:-$CUDA_VISIBLE_DEVICES}"
SEED="${SEED:-2022}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-512}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

if [[ "$SEED" != "2022" ]]; then
  echo "AAAI27 binary queue requires SEED=2022; got: $SEED" >&2
  exit 1
fi
if ! [[ "$WORKER_ID" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[1-4]$ ]] ||
  (( WORKER_ID >= NUM_WORKERS )); then
  echo "Invalid worker configuration: $WORKER_ID/$NUM_WORKERS" >&2
  exit 1
fi
if ! [[ "$PHYSICAL_GPU" =~ ^[0-9]+$ ]]; then
  echo "Invalid physical GPU index: $PHYSICAL_GPU" >&2
  exit 1
fi
if [[ "$WAIT_FOR_GPU_FREE" != "0" && "$WAIT_FOR_GPU_FREE" != "1" ]]; then
  echo "WAIT_FOR_GPU_FREE must be 0 or 1: $WAIT_FOR_GPU_FREE" >&2
  exit 1
fi
if ! [[ "$GPU_FREE_MEMORY_MAX_MB" =~ ^[0-9]+$ && "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid GPU wait settings: $GPU_FREE_MEMORY_MAX_MB/$GPU_POLL_SECONDS" >&2
  exit 1
fi

DATASETS=(
  PADS_09_task06_DrinkGlas
  PADS_10_task07_CrossArms
  Shimmer_11_session11_DRINK
  Shimmer_12_session12_PICK
)
FUSIONS=(concat concat_attn cross_attn_gate masked_pretrain)
COMMON_ARGS=(
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
  --image_mode activity_graph
  --mantis
  --mantis_name "$MANTIS_DIR"
  --classifier_type mlp
  --fusion_dim 128
  --fusion_heads 2
  --cross_attn_query ts
  --mask_prob 0.2
  --pretrain_epochs 3
  --mlp_hidden_dim 128
  --mlp_num_layers 1
  --mlp_dropout 0.3
  --mlp_lr 3e-4
  --mlp_weight_decay 1e-3
  --mlp_class_weight balanced
  --mlp_epochs 40
  --mlp_early_stop_patience 8
  --batch_size 16
  --data_dir "$DATA_DIR"
  --val_ratio 0.25
  --aaai27_label_mode zero_vs_rest
  --feature_cache_dir "$FEATURE_CACHE_ROOT"
)

wait_for_gpu() {
  local used_memory

  [[ "$WAIT_FOR_GPU_FREE" == "1" ]] || return 0
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is required when WAIT_FOR_GPU_FREE=1" >&2
    exit 1
  }

  while true; do
    if ! used_memory=$(nvidia-smi -i "$PHYSICAL_GPU" \
      --query-gpu=memory.used --format=csv,noheader,nounits 2>&1); then
      echo "GPU $PHYSICAL_GPU status query failed; retrying in ${GPU_POLL_SECONDS}s: $used_memory"
      sleep "$GPU_POLL_SECONDS"
      continue
    fi
    used_memory=${used_memory// /}
    if ! [[ "$used_memory" =~ ^[0-9]+$ ]]; then
      echo "Could not parse GPU $PHYSICAL_GPU memory usage; retrying in ${GPU_POLL_SECONDS}s: $used_memory"
      sleep "$GPU_POLL_SECONDS"
      continue
    fi
    if (( used_memory <= GPU_FREE_MEMORY_MAX_MB )); then
      echo "GPU $PHYSICAL_GPU is available (${used_memory} MiB used)."
      return 0
    fi
    echo "Waiting for GPU $PHYSICAL_GPU: ${used_memory} MiB used, threshold=${GPU_FREE_MEMORY_MAX_MB} MiB."
    sleep "$GPU_POLL_SECONDS"
  done
}

result_complete() {
  local task_result=$1
  local dataset=$2
  local fusion=$3
  local args_file
  local result_csv

  for args_file in "$task_result"/*/args.json; do
    result_csv="${args_file%/args.json}/train_val.csv"
    if [[ -s "$result_csv" ]] &&
      grep -Fq '"datasets": "aaai27"' "$args_file" &&
      grep -Fq "\"random_seed\": $SEED" "$args_file" &&
      grep -Fq '"aaai27_label_mode": "zero_vs_rest"' "$args_file" &&
      grep -Fq "\"modal_interaction\": \"$fusion\"" "$args_file" &&
      grep -Fq "\"feature_cache_dir\": \"$FEATURE_CACHE_ROOT\"" "$args_file" &&
      grep -Fq "\"$dataset\"" "$args_file" &&
      awk -F, -v expected="$dataset" \
        'NR > 1 && $1 == expected { found = 1 } END { exit !found }' \
        "$result_csv"; then
      return 0
    fi
  done
  return 1
}

run_dataset() {
  local dataset=$1
  local fusion
  local task_result

  for fusion in "${FUSIONS[@]}"; do
    task_result="$RESULT_ROOT/$dataset/$fusion"
    if result_complete "$task_result" "$dataset" "$fusion"; then
      echo "Skip completed | seed=$SEED | dataset=$dataset | fusion=$fusion"
      continue
    fi

    echo "Run | worker=$WORKER_ID/$NUM_WORKERS | gpu=$PHYSICAL_GPU | seed=$SEED | dataset=$dataset | fusion=$fusion"
    [[ "$DRY_RUN" == "1" ]] && continue

    mkdir -p "$task_result"
    "$PYTHON_BIN" "$PROJECT_DIR/main.py" \
      "${COMMON_ARGS[@]}" \
      --datasets aaai27 \
      --dataset_names "$dataset" \
      --modal_interaction "$fusion" \
      --random_seed "$SEED" \
      --result_dir "$task_result"
  done
}

if [[ "$DRY_RUN" != "1" ]]; then
  for path in "$PROJECT_DIR" "$MODEL_DIR" "$MANTIS_DIR" "$DATA_DIR"; do
    [[ -e "$path" ]] || { echo "Missing path: $path" >&2; exit 1; }
  done
  [[ -x "$PYTHON_BIN" ]] || { echo "Missing executable Python: $PYTHON_BIN" >&2; exit 1; }
fi

export CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tivit_aaai27_binary_mpl_gpu_${PHYSICAL_GPU}}"
export HF_HOME="${HF_HOME:-/tmp/tivit_aaai27_binary_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

wait_for_gpu

for dataset_index in "${!DATASETS[@]}"; do
  (( dataset_index % NUM_WORKERS == WORKER_ID )) || continue
  run_dataset "${DATASETS[$dataset_index]}"
done
