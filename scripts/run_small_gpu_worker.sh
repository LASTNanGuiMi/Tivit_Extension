#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-$SERVER_ROOT/data}"
RESULT_DIR="${RESULT_DIR:-$SERVER_ROOT/results/small_gpu}"
LOCK_DIR="${LOCK_DIR:-$SERVER_ROOT/results/.locks/small_gpu}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID, starting at 0.}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS.}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tivit_mplconfig}"
export HF_HOME="${HF_HOME:-/tmp/tivit_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

mkdir -p "$RESULT_DIR" "$LOCK_DIR" "$SERVER_ROOT/logs"
cd "$PROJECT_DIR"

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

result_completed() {
  local dataset=$1
  local fusion=$2
  local seed=$3
  local dir

  for dir in "$RESULT_DIR"/*; do
    if [[ -f "$dir/args.json" && -f "$dir/train_val.csv" ]] &&
      grep -q "\"datasets\": \"${dataset}\"" "$dir/args.json" &&
      grep -q "\"modal_interaction\": \"${fusion}\"" "$dir/args.json" &&
      grep -q "\"random_seed\": ${seed}" "$dir/args.json" &&
      grep -q '"val_ratio": 0.25' "$dir/args.json"; then
      return 0
    fi
  done

  return 1
}

run_task() {
  local dataset=$1
  local fusion=$2
  local seed=$3
  local key="${seed}_${dataset}_${fusion}"
  shift 3

  if result_completed "$dataset" "$fusion" "$seed"; then
    echo "Skip completed | seed=${seed} | ${dataset} | fusion=${fusion}"
    return
  fi

  if ! mkdir "$LOCK_DIR/$key" 2>/dev/null; then
    echo "Skip locked | seed=${seed} | ${dataset} | fusion=${fusion}"
    return
  fi

  if result_completed "$dataset" "$fusion" "$seed"; then
    echo "Skip completed after lock | seed=${seed} | ${dataset} | fusion=${fusion}"
    rmdir "$LOCK_DIR/$key"
    return
  fi

  echo "Run worker=${WORKER_ID}/${NUM_WORKERS} | gpu=${CUDA_VISIBLE_DEVICES:-unset} | seed=${seed} | ${dataset} | fusion=${fusion}"
  "$PYTHON_BIN" main.py \
    "${COMMON_ARGS[@]}" \
    --random_seed "$seed" \
    --modal_interaction "$fusion" \
    "$@"

  rmdir "$LOCK_DIR/$key"
}

TASKS=()
for seed in 2022; do
  for fusion in concat concat_attn cross_attn_gate masked_pretrain; do
    TASKS+=("$seed|uea|$fusion|--datasets uea --dataset_names BasicMotions")
    TASKS+=("$seed|ucr|$fusion|--datasets ucr --dataset_names ECG200")
    TASKS+=("$seed|feng|$fusion|--window_size 200 --window_stride 100 --custom_test_ratio 0.2 --max_windows_per_file 20 --datasets feng")
  done
done

for i in "${!TASKS[@]}"; do
  if (( i % NUM_WORKERS != WORKER_ID )); then
    continue
  fi

  IFS="|" read -r seed dataset fusion extra <<< "${TASKS[$i]}"
  # shellcheck disable=SC2086
  run_task "$dataset" "$fusion" "$seed" $extra
done
