#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=ablation_config.sh
source "$SCRIPT_DIR/ablation_config.sh"

RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT.}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID.}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS.}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
DRY_RUN="${DRY_RUN:-0}"

if ! [[ "$WORKER_ID" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  (( WORKER_ID >= NUM_WORKERS )); then
  echo "Invalid worker configuration: $WORKER_ID/$NUM_WORKERS" >&2
  exit 1
fi
if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer: $REPEATS" >&2
  exit 1
fi
case "$PROPOSED_FUSION" in
  concat_attn|cross_attn_gate|masked_pretrain) ;;
  *)
    echo "Invalid PROPOSED_FUSION: $PROPOSED_FUSION" >&2
    exit 1
    ;;
esac

read -r -a SEED_VALUES <<< "$SEEDS"
read -r -a DATASET_VALUES <<< "$DATASET_GROUPS"
CONDITIONS=(
  vision_line_plot
  vision_activity_graph
  timeseries_mantis
  multimodal_concat
  multimodal_proposed
)

for seed in "${SEED_VALUES[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 1; }
done
for dataset_group in "${DATASET_VALUES[@]}"; do
  case "$dataset_group" in
    feng|falltl|uci) ;;
    *) echo "Invalid ablation dataset group: $dataset_group" >&2; exit 1 ;;
  esac
done

if [[ "$DRY_RUN" != "1" ]]; then
  for path in "$PROJECT_DIR" "$MODEL_DIR" "$MANTIS_DIR" "$DATA_DIR" "$PYTHON_BIN"; do
    [[ -e "$path" ]] || { echo "Missing path: $path" >&2; exit 1; }
  done
fi

COMMON_ARGS=(
  --classifier_type mlp
  --fusion_dim "$FUSION_DIM"
  --fusion_heads "$FUSION_HEADS"
  --cross_attn_query "$CROSS_ATTN_QUERY"
  --mask_prob "$MASK_PROB"
  --pretrain_epochs "$PRETRAIN_EPOCHS"
  --mlp_hidden_dim "$MLP_HIDDEN_DIM"
  --mlp_num_layers "$MLP_NUM_LAYERS"
  --mlp_dropout "$MLP_DROPOUT"
  --mlp_lr "$MLP_LR"
  --mlp_weight_decay "$MLP_WEIGHT_DECAY"
  --mlp_epochs "$MLP_EPOCHS"
  --mlp_early_stop_patience "$MLP_PATIENCE"
  --batch_size "$BATCH_SIZE"
  --data_dir "$DATA_DIR"
  --val_ratio "$VAL_RATIO"
)
VISION_ARGS=(
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
)
TIMESERIES_ARGS=(--mantis --mantis_name "$MANTIS_DIR")

result_complete() {
  local task_result=$1
  local result_csv
  for result_csv in "$task_result"/*/train_val.csv; do
    if [[ -s "$result_csv" ]] && (( $(wc -l < "$result_csv") >= 2 )); then
      return 0
    fi
  done
  return 1
}

run_task() {
  local repeat=$1
  local seed=$2
  local dataset_group=$3
  local condition=$4
  local task_result="$RESULT_ROOT/$condition/seed_${seed}/repeat_${repeat}"
  local image_mode=activity_graph
  local fusion=concat
  local -a branch_args dataset_args

  case "$condition" in
    vision_line_plot)
      image_mode=multichannel_line_plot
      branch_args=("${VISION_ARGS[@]}")
      ;;
    vision_activity_graph)
      branch_args=("${VISION_ARGS[@]}")
      ;;
    timeseries_mantis)
      branch_args=("${TIMESERIES_ARGS[@]}")
      ;;
    multimodal_concat)
      branch_args=("${VISION_ARGS[@]}" "${TIMESERIES_ARGS[@]}")
      ;;
    multimodal_proposed)
      fusion=$PROPOSED_FUSION
      branch_args=("${VISION_ARGS[@]}" "${TIMESERIES_ARGS[@]}")
      ;;
    *) echo "Unknown ablation condition: $condition" >&2; exit 1 ;;
  esac

  case "$dataset_group" in
    feng)
      dataset_args=(
        --datasets feng
        --window_size "$WINDOW_SIZE"
        --window_stride "$WINDOW_STRIDE"
        --custom_test_ratio "$CUSTOM_TEST_RATIO"
        --max_windows_per_file "$FENG_MAX_WINDOWS"
      )
      ;;
    falltl)
      dataset_args=(
        --datasets falltl
        --window_size "$WINDOW_SIZE"
        --window_stride "$WINDOW_STRIDE"
        --custom_test_ratio "$CUSTOM_TEST_RATIO"
        --max_windows_per_file "$FALLTL_MAX_WINDOWS"
      )
      ;;
    uci)
      dataset_args=(--datasets uci --custom_test_ratio "$CUSTOM_TEST_RATIO")
      ;;
  esac

  if result_complete "$task_result"; then
    echo "Skip completed | repeat=$repeat | seed=$seed | $dataset_group | $condition"
    return
  fi

  echo "Run | repeat=$repeat/$REPEATS | seed=$seed | dataset=$dataset_group | condition=$condition | gpu=$CUDA_VISIBLE_DEVICES"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  mkdir -p "$task_result"
  "$PYTHON_BIN" "$PROJECT_DIR/main.py" \
    "${COMMON_ARGS[@]}" \
    "${branch_args[@]}" \
    "${dataset_args[@]}" \
    --image_mode "$image_mode" \
    --modal_interaction "$fusion" \
    --random_seed "$seed" \
    --result_dir "$task_result"
}

export CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tivit_ablation_mpl}"
export HF_HOME="${HF_HOME:-/tmp/tivit_ablation_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

task_index=0
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  for seed in "${SEED_VALUES[@]}"; do
    for dataset_group in "${DATASET_VALUES[@]}"; do
      for condition in "${CONDITIONS[@]}"; do
        current_index=$task_index
        task_index=$((task_index + 1))
        if (( current_index % NUM_WORKERS != WORKER_ID )); then
          continue
        fi
        run_task "$repeat" "$seed" "$dataset_group" "$condition"
      done
    done
  done
done
