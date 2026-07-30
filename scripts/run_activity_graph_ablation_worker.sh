#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT.}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID.}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS.}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
REPEATS="${REPEATS:-3}"
FIXED_ABLATION_SEED=2022
SEED="${SEED:-$FIXED_ABLATION_SEED}"
DRY_RUN="${DRY_RUN:-0}"
ABLATION_KIND="${ABLATION_KIND:-image}"

if [[ "$SEED" != "$FIXED_ABLATION_SEED" ]]; then
  echo "Activity-graph ablation requires SEED=$FIXED_ABLATION_SEED; got: $SEED" >&2
  exit 1
fi

if ! [[ "$WORKER_ID" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  (( WORKER_ID >= NUM_WORKERS )); then
  echo "Invalid worker configuration: $WORKER_ID/$NUM_WORKERS" >&2
  exit 1
fi
case "$ABLATION_KIND" in
  image) CONDITIONS=(vision_line_plot vision_activity_graph) ;;
  fusion) CONDITIONS=(multimodal_concat multimodal_proposed) ;;
  *) echo "Invalid ABLATION_KIND: $ABLATION_KIND" >&2; exit 1 ;;
esac

SMALL_TRAINING_ARGS=(
  --fusion_dim 128 --fusion_heads 2 --cross_attn_query ts
  --mask_prob 0.2 --pretrain_epochs 3
  --mlp_hidden_dim 128 --mlp_num_layers 1 --mlp_dropout 0.3
  --mlp_lr 3e-4 --mlp_weight_decay 1e-3
  --mlp_epochs 40 --mlp_early_stop_patience 8 --batch_size 16
)
MAIN_TRAINING_ARGS=(
  --fusion_dim 512 --fusion_heads 4 --cross_attn_query ts
  --mask_prob 0.3 --pretrain_epochs 5
  --mlp_hidden_dim 512 --mlp_num_layers 2 --mlp_dropout 0.1
  --mlp_lr 1e-4 --mlp_weight_decay 1e-4
  --mlp_epochs 20 --mlp_early_stop_patience 3 --batch_size 32
)
COMMON_ARGS=(
  --vit_1_name "$MODEL_DIR" --vit_1_layer 14 --aggregation mean
  --classifier_type mlp
  --data_dir "$DATA_DIR" --val_ratio 0.25 --random_seed "$SEED"
)

result_complete() {
  local task_result=$1
  local expected_rows=$2
  local result_csv
  for result_csv in "$task_result"/*/train_val.csv; do
    if [[ -s "$result_csv" ]] && (( $(wc -l < "$result_csv") >= expected_rows + 1 )); then
      return 0
    fi
  done
  return 1
}

run_task() {
  local repeat=$1 suite=$2 condition=$3 dataset_key=$4 expected_rows=$5
  shift 5
  local image_mode=activity_graph
  local modal_interaction=concat
  local task_result="$RESULT_ROOT/$suite/$condition/$dataset_key/repeat_$repeat"
  local -a training_args branch_args

  [[ "$condition" == "vision_line_plot" ]] && image_mode=multichannel_line_plot
  case "$condition" in
    vision_line_plot|vision_activity_graph)
      branch_args=()
      ;;
    multimodal_concat)
      branch_args=(--mantis --mantis_name "$MANTIS_DIR")
      ;;
    multimodal_proposed)
      branch_args=(--mantis --mantis_name "$MANTIS_DIR")
      modal_interaction=cross_attn_gate
      ;;
    *) echo "Invalid condition: $condition" >&2; exit 1 ;;
  esac
  if [[ "$suite" == "small" ]]; then
    training_args=("${SMALL_TRAINING_ARGS[@]}")
  else
    training_args=("${MAIN_TRAINING_ARGS[@]}")
  fi

  if result_complete "$task_result" "$expected_rows"; then
    echo "Skip completed | suite=$suite | repeat=$repeat | $dataset_key | $condition"
    return
  fi
  echo "Run | worker=$WORKER_ID/$NUM_WORKERS | gpu=$CUDA_VISIBLE_DEVICES | suite=$suite | repeat=$repeat/$REPEATS | seed=$SEED | dataset=$dataset_key | condition=$condition"
  [[ "$DRY_RUN" == "1" ]] && return

  mkdir -p "$task_result"
  "$PYTHON_BIN" "$PROJECT_DIR/main.py" \
    "${COMMON_ARGS[@]}" "${training_args[@]}" "${branch_args[@]}" "$@" \
    --image_mode "$image_mode" --modal_interaction "$modal_interaction" \
    --result_dir "$task_result"
}

dispatch() {
  local index=$TASK_INDEX
  TASK_INDEX=$((TASK_INDEX + 1))
  (( index % NUM_WORKERS == WORKER_ID )) || return 0
  run_task "$@"
}

if [[ "$DRY_RUN" != "1" ]]; then
  required_paths=("$PROJECT_DIR" "$MODEL_DIR" "$DATA_DIR" "$PYTHON_BIN")
  [[ "$ABLATION_KIND" == "fusion" ]] && required_paths+=("$MANTIS_DIR")
  for path in "${required_paths[@]}"; do
    [[ -e "$path" ]] || { echo "Missing path: $path" >&2; exit 1; }
  done
fi

export CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tivit_activity_ablation_mpl}"
export HF_HOME="${HF_HOME:-/tmp/tivit_activity_ablation_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

TASK_INDEX=0
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  for condition in "${CONDITIONS[@]}"; do
    dispatch "$repeat" small "$condition" uea 1 --datasets uea --dataset_names BasicMotions
    dispatch "$repeat" small "$condition" ucr 1 --datasets ucr --dataset_names ECG200
    dispatch "$repeat" small "$condition" feng 1 --window_size 200 --window_stride 100 --custom_test_ratio 0.2 --max_windows_per_file 20 --datasets feng
    dispatch "$repeat" small "$condition" uci 1 --custom_test_ratio 0.2 --datasets uci

    dispatch "$repeat" main "$condition" uea 2 --datasets uea --dataset_names BasicMotions SelfRegulationSCP1
    dispatch "$repeat" main "$condition" ucr 2 --datasets ucr --dataset_names ECG200 FordA
    dispatch "$repeat" main "$condition" feng 1 --window_size 200 --window_stride 100 --custom_test_ratio 0.2 --max_windows_per_file 2 --datasets feng
    dispatch "$repeat" main "$condition" falltl 1 --window_size 200 --window_stride 100 --custom_test_ratio 0.2 --max_windows_per_file 2 --datasets falltl
    dispatch "$repeat" main "$condition" uci 1 --custom_test_ratio 0.2 --datasets uci
  done
done
