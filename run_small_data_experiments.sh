#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
RESULT_DIR="${RESULT_DIR:-$SERVER_ROOT/results/small_gpu}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORKER_ID="${WORKER_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tivit_mplconfig}"
HF_HOME="${HF_HOME:-/tmp/tivit_hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

export CUDA_VISIBLE_DEVICES MPLCONFIGDIR HF_HOME TRANSFORMERS_CACHE
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

if ! [[ "$WORKER_ID" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  (( WORKER_ID >= NUM_WORKERS )); then
  echo "Invalid worker configuration: WORKER_ID=$WORKER_ID NUM_WORKERS=$NUM_WORKERS" >&2
  exit 1
fi

mkdir -p "$RESULT_DIR"

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
  --mlp_epochs 40
  --mlp_early_stop_patience 8
  --batch_size 16
  --data_dir "$DATA_DIR"
  --result_dir "$RESULT_DIR"
  --val_ratio 0.25
)

CUSTOM_DATA_ARGS=(
  --window_size 200
  --window_stride 100
  --custom_test_ratio 0.2
)

result_completed() {
  local dataset=$1
  local fusion=$2
  local seed=$3
  local dir
  local rows

  for dir in "$RESULT_DIR"/*; do
    if [[ -f "$dir/args.json" && -f "$dir/train_val.csv" ]] &&
      grep -q "\"datasets\": \"${dataset}\"" "$dir/args.json" &&
      grep -q "\"modal_interaction\": \"${fusion}\"" "$dir/args.json" &&
      grep -q "\"random_seed\": ${seed}" "$dir/args.json" &&
      grep -q '"val_ratio": 0.25' "$dir/args.json" &&
      grep -q '"custom_test_ratio": 0.2' "$dir/args.json"; then
      rows=$(awk 'NR > 1 {count++} END {print count + 0}' "$dir/train_val.csv")
      if (( rows >= 1 )); then
        return 0
      fi
    fi
  done

  return 1
}

run_experiment() {
  local dataset=$1
  local fusion=$2
  local seed=$3
  local task_index=$TASK_INDEX
  shift 3

  TASK_INDEX=$((TASK_INDEX + 1))
  if (( task_index % NUM_WORKERS != WORKER_ID )); then
    return
  fi

  if result_completed "$dataset" "$fusion" "$seed"; then
    echo "Skip completed | seed=${seed} | ${dataset} | fusion=${fusion}"
    return
  fi

  echo "Run small-data | worker=${WORKER_ID}/${NUM_WORKERS} | gpu=${CUDA_VISIBLE_DEVICES} | seed=${seed} | ${dataset} | fusion=${fusion}"
  "$PYTHON_BIN" main.py \
    "${COMMON_ARGS[@]}" \
    --random_seed "$seed" \
    --modal_interaction "$fusion" \
    "$@"
}

for path in "$MODEL_DIR" "$MANTIS_DIR" "$DATA_DIR"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing path: $path" >&2
    exit 1
  fi
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing executable Python: $PYTHON_BIN" >&2
  exit 1
fi

SEEDS=(2022)
FUSIONS=(concat concat_attn cross_attn_gate masked_pretrain)
TASK_INDEX=0

for SEED in "${SEEDS[@]}"; do
  for FUSION in "${FUSIONS[@]}"; do
    run_experiment uea "$FUSION" "$SEED" \
      --datasets uea \
      --dataset_names BasicMotions
  done

  for FUSION in "${FUSIONS[@]}"; do
    run_experiment ucr "$FUSION" "$SEED" \
      --datasets ucr \
      --dataset_names ECG200
  done

  for FUSION in "${FUSIONS[@]}"; do
    run_experiment feng "$FUSION" "$SEED" \
      "${CUSTOM_DATA_ARGS[@]}" \
      --max_windows_per_file 20 \
      --datasets feng
  done

  for FUSION in "${FUSIONS[@]}"; do
    run_experiment uci "$FUSION" "$SEED" \
      --custom_test_ratio 0.2 \
      --datasets uci
  done
done
