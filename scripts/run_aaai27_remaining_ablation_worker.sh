#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:?Set FEATURE_CACHE_ROOT}"
MULTIMODAL_CACHE_ROOT="${MULTIMODAL_CACHE_ROOT:?Set MULTIMODAL_CACHE_ROOT}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS}"
PHYSICAL_GPU="${PHYSICAL_GPU:?Set PHYSICAL_GPU}"
SEED="${SEED:-2022}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-1}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-512}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

DATASETS=(
  PADS_09_task06_DrinkGlas
  PADS_10_task07_CrossArms
  Shimmer_11_session11_DRINK
  Shimmer_12_session12_PICK
)
CONDITIONS=(
  vision_line_plot
  vision_activity_graph
  timeseries_mantis
  multimodal_concat
)
COMMON_ARGS=(
  --aggregation mean
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
)

if [[ "$SEED" != "2022" ]]; then
  echo "AAAI27 binary ablation requires SEED=2022; got $SEED" >&2
  exit 1
fi
if ! [[ "$WORKER_ID" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  (( WORKER_ID >= NUM_WORKERS )); then
  echo "Invalid worker configuration: $WORKER_ID/$NUM_WORKERS" >&2
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

result_complete() {
  local task_result=$1
  local dataset=$2
  local condition=$3
  local args_file result_csv
  for args_file in "$task_result"/*/args.json; do
    result_csv="${args_file%/args.json}/train_val.csv"
    if [[ -s "$result_csv" ]] &&
      grep -Fq '"datasets": "aaai27"' "$args_file" &&
      grep -Fq "\"random_seed\": $SEED" "$args_file" &&
      grep -Fq '"aaai27_label_mode": "zero_vs_rest"' "$args_file" &&
      grep -Fq '"modal_interaction": "concat"' "$args_file" &&
      grep -Fq '"mlp_class_weight": "balanced"' "$args_file" &&
      grep -Fq "\"$dataset\"" "$args_file" &&
      awk -F, -v expected="$dataset" \
        'NR > 1 && $1 == expected { found = 1 } END { exit !found }' \
        "$result_csv"; then
      echo "Completed result found for $dataset/$condition: $args_file"
      return 0
    fi
  done
  return 1
}

run_task() {
  local dataset=$1
  local condition=$2
  local image_mode=activity_graph
  local cache_root="$FEATURE_CACHE_ROOT/$condition"
  local task_result="$RESULT_ROOT/$dataset/$condition"
  local -a branch_args

  case "$condition" in
    vision_line_plot)
      image_mode=multichannel_line_plot
      branch_args=(--vit_1_name "$MODEL_DIR" --vit_1_layer 14)
      ;;
    vision_activity_graph)
      branch_args=(--vit_1_name "$MODEL_DIR" --vit_1_layer 14)
      ;;
    timeseries_mantis)
      branch_args=(--mantis --mantis_name "$MANTIS_DIR")
      ;;
    multimodal_concat)
      branch_args=(
        --vit_1_name "$MODEL_DIR" --vit_1_layer 14
        --mantis --mantis_name "$MANTIS_DIR"
      )
      cache_root="$MULTIMODAL_CACHE_ROOT"
      ;;
    *)
      echo "Unsupported condition: $condition" >&2
      exit 1
      ;;
  esac

  if result_complete "$task_result" "$dataset" "$condition"; then
    echo "Skip completed | dataset=$dataset | condition=$condition"
    return 0
  fi
  echo "Run | worker=$WORKER_ID/$NUM_WORKERS | gpu=$PHYSICAL_GPU | seed=$SEED | dataset=$dataset | condition=$condition"
  [[ "$DRY_RUN" == "1" ]] && return 0

  mkdir -p "$task_result" "$cache_root"
  "$PYTHON_BIN" "$PROJECT_DIR/main.py" \
    "${COMMON_ARGS[@]}" \
    "${branch_args[@]}" \
    --datasets aaai27 \
    --dataset_names "$dataset" \
    --image_mode "$image_mode" \
    --modal_interaction concat \
    --random_seed "$SEED" \
    --feature_cache_dir "$cache_root" \
    --result_dir "$task_result"
}

if [[ "$DRY_RUN" != "1" ]]; then
  for path in "$PROJECT_DIR" "$MODEL_DIR" "$MANTIS_DIR" "$DATA_DIR" "$PYTHON_BIN"; do
    [[ -e "$path" ]] || { echo "Missing path: $path" >&2; exit 1; }
  done
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tivit_aaai27_remaining_mpl_gpu_${PHYSICAL_GPU}}"
export HF_HOME="${HF_HOME:-/tmp/tivit_aaai27_remaining_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

wait_for_gpu
task_index=0
for dataset in "${DATASETS[@]}"; do
  for condition in "${CONDITIONS[@]}"; do
    if (( task_index % NUM_WORKERS != WORKER_ID )); then
      ((task_index += 1))
      continue
    fi
    run_task "$dataset" "$condition"
    ((task_index += 1))
  done
done
