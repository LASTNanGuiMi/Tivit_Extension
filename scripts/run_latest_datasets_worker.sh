#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT}"
STATUS_DIR="${STATUS_DIR:?Set STATUS_DIR}"
UNIMODAL_CACHE_ROOT="${UNIMODAL_CACHE_ROOT:?Set UNIMODAL_CACHE_ROOT}"
MULTIMODAL_CACHE_ROOT="${MULTIMODAL_CACHE_ROOT:?Set MULTIMODAL_CACHE_ROOT}"
FALLTL_CACHE_ROOT="${FALLTL_CACHE_ROOT:?Set FALLTL_CACHE_ROOT}"
FALLTL_CACHE_STATUS_FILE="${FALLTL_CACHE_STATUS_FILE:?Set FALLTL_CACHE_STATUS_FILE}"
SPLIT_REFERENCE="${SPLIT_REFERENCE:?Set SPLIT_REFERENCE}"
TARGETS="${TARGETS:-pads shimmer falltl ucihar}"
OUTER_SEEDS="${OUTER_SEEDS:-2020 2021 2022 2023 2024}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-2022}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS}"
PHYSICAL_GPU="${PHYSICAL_GPU:?Set PHYSICAL_GPU}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-2048}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"

AAAI27_CONDITIONS=(
  vision_line_plot
  vision_activity_graph
  timeseries_mantis
  multimodal_concat
  concat_attn
  cross_attn_gate
  masked_pretrain
)
FUSIONS=(concat concat_attn cross_attn_gate masked_pretrain)

read -r -a TARGET_VALUES <<< "$TARGETS"
read -r -a SEED_VALUES <<< "$OUTER_SEEDS"

target_enabled() {
  local wanted=$1
  local target
  for target in "${TARGET_VALUES[@]}"; do
    [[ "$target" == "$wanted" ]] && return 0
  done
  return 1
}

if ! [[ "$WORKER_ID" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] ||
  (( WORKER_ID >= NUM_WORKERS )); then
  echo "Invalid worker configuration: $WORKER_ID/$NUM_WORKERS" >&2
  exit 1
fi
if target_enabled ucihar && [[ "$SEED" != "2022" || "$REPEATS" != "3" ]]; then
  echo "The latest UCIHAR protocol requires SEED=2022 and REPEATS=3." >&2
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

aaai27_result_complete() {
  local task_result=$1 dataset=$2 outer_seed=$3
  local args_file result_csv audit_csv
  for args_file in "$task_result"/*/args.json; do
    result_csv="${args_file%/args.json}/train_val.csv"
    audit_csv="${args_file%/args.json}/splits/${dataset}_subject_split.csv"
    if [[ -s "$result_csv" && -s "$audit_csv" ]] &&
      grep -Fq '"datasets": "aaai27"' "$args_file" &&
      grep -Fq "\"random_seed\": $outer_seed" "$args_file" &&
      grep -Fq '"aaai27_label_mode": "zero_vs_rest"' "$args_file" &&
      grep -Fq '"mlp_class_weight": "balanced"' "$args_file" &&
      grep -Fq "\"$dataset\"" "$args_file" &&
      awk -F, -v expected="$dataset" \
        'NR > 1 && $1 == expected { found = 1 } END { exit !found }' \
        "$result_csv"; then
      return 0
    fi
  done
  return 1
}

run_aaai27_task() {
  local outer_seed=$1 dataset=$2 condition=$3 task_index=$4
  (( task_index % NUM_WORKERS == WORKER_ID )) || return 0

  local image_mode=activity_graph modal_interaction=concat
  local cache_root="$MULTIMODAL_CACHE_ROOT"
  local task_result="$RESULT_ROOT/aaai27/seed_${outer_seed}/$dataset/$condition"
  local -a branch_args

  case "$condition" in
    vision_line_plot)
      image_mode=multichannel_line_plot
      cache_root="$UNIMODAL_CACHE_ROOT/vision_line_plot"
      branch_args=(--vit_1_name "$MODEL_DIR" --vit_1_layer 14)
      ;;
    vision_activity_graph)
      cache_root="$UNIMODAL_CACHE_ROOT/vision_activity_graph"
      branch_args=(--vit_1_name "$MODEL_DIR" --vit_1_layer 14)
      ;;
    timeseries_mantis)
      cache_root="$UNIMODAL_CACHE_ROOT/timeseries_mantis"
      branch_args=(--mantis --mantis_name "$MANTIS_DIR")
      ;;
    multimodal_concat)
      branch_args=(
        --vit_1_name "$MODEL_DIR" --vit_1_layer 14
        --mantis --mantis_name "$MANTIS_DIR"
      )
      ;;
    concat_attn|cross_attn_gate|masked_pretrain)
      modal_interaction="$condition"
      branch_args=(
        --vit_1_name "$MODEL_DIR" --vit_1_layer 14
        --mantis --mantis_name "$MANTIS_DIR"
      )
      ;;
    *) echo "Unsupported AAAI27 condition: $condition" >&2; exit 1 ;;
  esac

  if aaai27_result_complete "$task_result" "$dataset" "$outer_seed"; then
    echo "Skip completed | protocol=aaai27 | outer_seed=$outer_seed | dataset=$dataset | condition=$condition"
    return 0
  fi
  echo "Run | worker=$WORKER_ID/$NUM_WORKERS | gpu=$PHYSICAL_GPU | protocol=aaai27 | split_seed=42 | outer_seed=$outer_seed | dataset=$dataset | condition=$condition"
  [[ "$DRY_RUN" == "1" ]] && return 0

  mkdir -p "$task_result" "$cache_root"
  "$PYTHON_BIN" "$PROJECT_DIR/main.py" \
    --aggregation mean \
    --classifier_type mlp \
    --fusion_dim 128 \
    --fusion_heads 2 \
    --cross_attn_query ts \
    --mask_prob 0.2 \
    --pretrain_epochs 3 \
    --mlp_hidden_dim 128 \
    --mlp_num_layers 1 \
    --mlp_dropout 0.3 \
    --mlp_lr 3e-4 \
    --mlp_weight_decay 1e-3 \
    --mlp_class_weight balanced \
    --mlp_epochs 40 \
    --mlp_early_stop_patience 8 \
    --batch_size 16 \
    --data_dir "$DATA_DIR" \
    --val_ratio 0.25 \
    --aaai27_label_mode zero_vs_rest \
    "${branch_args[@]}" \
    --datasets aaai27 \
    --dataset_names "$dataset" \
    --image_mode "$image_mode" \
    --modal_interaction "$modal_interaction" \
    --random_seed "$outer_seed" \
    --feature_cache_dir "$cache_root" \
    --result_dir "$task_result"
}

falltl_result_complete() {
  local task_result=$1 fusion=$2
  local args_file result_csv audit_csv
  for args_file in "$task_result"/*/args.json; do
    result_csv="${args_file%/args.json}/train_val.csv"
    audit_csv="${args_file%/args.json}/splits/FallTL_comparison_binary_split.csv"
    if [[ -s "$result_csv" && -s "$audit_csv" ]] &&
      grep -Fq '"datasets": "falltl"' "$args_file" &&
      grep -Fq '"falltl_protocol": "comparison_binary"' "$args_file" &&
      grep -Fq '"random_seed": 2022' "$args_file" &&
      grep -Fq "\"modal_interaction\": \"$fusion\"" "$args_file" &&
      awk -F, 'NR > 1 && $1 == "FallTL" { found = 1 } END { exit !found }' \
        "$result_csv"; then
      return 0
    fi
  done
  return 1
}

wait_for_falltl_cache() {
  local status worker_status
  while true; do
    if [[ -f "$FALLTL_CACHE_STATUS_FILE" ]]; then
      status=$(tr -d '\n' < "$FALLTL_CACHE_STATUS_FILE")
      [[ "$status" == "0" ]] && return 0
      if [[ "$status" =~ ^[1-9][0-9]*$ ]]; then
        echo "FallTL concat/cache producer failed with status $status." >&2
        return "$status"
      fi
    fi
    for worker_status in "$STATUS_DIR"/worker_*.status; do
      if [[ -f "$worker_status" ]]; then
        status=$(tr -d '\n' < "$worker_status")
        if [[ "$status" =~ ^[1-9][0-9]*$ ]]; then
          echo "A worker failed before the FallTL cache became ready." >&2
          return 1
        fi
      fi
    done
    echo "Waiting for the FallTL concat/cache producer."
    sleep "$GPU_POLL_SECONDS"
  done
}

run_falltl_task() {
  local fusion=$1 task_index=$2
  (( task_index % NUM_WORKERS == WORKER_ID )) || return 0

  local task_result="$RESULT_ROOT/falltl/$fusion"
  local status=0
  if falltl_result_complete "$task_result" "$fusion"; then
    echo "Skip completed | protocol=falltl | fusion=$fusion"
    if [[ "$fusion" == "concat" && "$DRY_RUN" != "1" ]]; then
      printf '0\n' > "$FALLTL_CACHE_STATUS_FILE"
    fi
    return 0
  fi

  if [[ "$fusion" != "concat" && "$DRY_RUN" != "1" ]]; then
    wait_for_falltl_cache
  fi
  echo "Run | worker=$WORKER_ID/$NUM_WORKERS | gpu=$PHYSICAL_GPU | protocol=falltl | seed=2022 | fusion=$fusion"
  [[ "$DRY_RUN" == "1" ]] && return 0

  mkdir -p "$task_result" "$FALLTL_CACHE_ROOT"
  set +e
  "$PYTHON_BIN" "$PROJECT_DIR/main.py" \
    --vit_1_name "$MODEL_DIR" \
    --vit_1_layer 14 \
    --aggregation mean \
    --image_mode activity_graph \
    --mantis \
    --mantis_name "$MANTIS_DIR" \
    --classifier_type mlp \
    --fusion_dim 512 \
    --fusion_heads 4 \
    --cross_attn_query ts \
    --mask_prob 0.3 \
    --pretrain_epochs 5 \
    --mlp_hidden_dim 512 \
    --mlp_num_layers 2 \
    --mlp_dropout 0.1 \
    --mlp_lr 1e-4 \
    --mlp_weight_decay 1e-4 \
    --mlp_epochs 20 \
    --mlp_early_stop_patience 3 \
    --batch_size 32 \
    --data_dir "$DATA_DIR" \
    --val_ratio 0.25 \
    --datasets falltl \
    --falltl_protocol comparison_binary \
    --modal_interaction "$fusion" \
    --random_seed 2022 \
    --feature_cache_dir "$FALLTL_CACHE_ROOT" \
    --result_dir "$task_result"
  status=$?
  set -e
  if [[ "$fusion" == "concat" ]]; then
    printf '%s\n' "$status" > "$FALLTL_CACHE_STATUS_FILE"
  fi
  return "$status"
}

ucihar_result_complete() {
  local task_result=$1 fusion=$2
  local args_file result_csv
  for args_file in "$task_result"/*/args.json; do
    result_csv="${args_file%/args.json}/train_val.csv"
    if [[ -s "$result_csv" ]] &&
      grep -Fq '"datasets": "uci"' "$args_file" &&
      grep -Fq '"random_seed": 2022' "$args_file" &&
      grep -Fq '"har_channels": "acc_gyro"' "$args_file" &&
      grep -Fq "\"modal_interaction\": \"$fusion\"" "$args_file" &&
      awk -F, 'NR > 1 && $1 == "UCIHAR" { found = 1 } END { exit !found }' \
        "$result_csv"; then
      return 0
    fi
  done
  return 1
}

run_ucihar_task() {
  local repeat=$1 fusion=$2 task_index=$3
  (( task_index % NUM_WORKERS == WORKER_ID )) || return 0

  local task_result="$RESULT_ROOT/har6/uci/repeat_${repeat}/$fusion"
  if ucihar_result_complete "$task_result" "$fusion"; then
    echo "Skip completed | protocol=har6 | dataset=ucihar | repeat=$repeat | fusion=$fusion"
    return 0
  fi
  echo "Run | worker=$WORKER_ID/$NUM_WORKERS | gpu=$PHYSICAL_GPU | protocol=har6 | dataset=ucihar | repeat=$repeat/$REPEATS | seed=$SEED | fusion=$fusion"
  [[ "$DRY_RUN" == "1" ]] && return 0

  mkdir -p "$task_result"
  "$PYTHON_BIN" "$PROJECT_DIR/main.py" \
    --vit_1_name "$MODEL_DIR" \
    --vit_1_layer 14 \
    --aggregation mean \
    --image_mode activity_graph \
    --mantis \
    --mantis_name "$MANTIS_DIR" \
    --classifier_type mlp \
    --fusion_dim 512 \
    --fusion_heads 4 \
    --cross_attn_query ts \
    --mask_prob 0.3 \
    --pretrain_epochs 5 \
    --mlp_hidden_dim 512 \
    --mlp_num_layers 2 \
    --mlp_dropout 0.1 \
    --mlp_lr 1e-4 \
    --mlp_weight_decay 1e-4 \
    --mlp_epochs 20 \
    --mlp_early_stop_patience 3 \
    --batch_size 32 \
    --data_dir "$DATA_DIR/med_data" \
    --val_ratio 0.25 \
    --custom_test_ratio 0.2 \
    --har_channels acc_gyro \
    --datasets uci \
    --modal_interaction "$fusion" \
    --random_seed "$SEED" \
    --result_dir "$task_result"
}

if [[ "$DRY_RUN" != "1" ]]; then
  for path in "$PROJECT_DIR" "$MODEL_DIR" "$MANTIS_DIR" "$DATA_DIR" "$PYTHON_BIN"; do
    [[ -e "$path" ]] || { echo "Missing path: $path" >&2; exit 1; }
  done
  if target_enabled pads || target_enabled shimmer; then
    [[ -f "$SPLIT_REFERENCE" ]] || { echo "Missing split reference: $SPLIT_REFERENCE" >&2; exit 1; }
  fi
  if target_enabled ucihar; then
    [[ -d "$DATA_DIR/med_data" ]] || { echo "Missing UCIHAR data root: $DATA_DIR/med_data" >&2; exit 1; }
  fi
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tivit_latest_suite_mpl_gpu_${PHYSICAL_GPU}}"
export HF_HOME="${HF_HOME:-/tmp/tivit_latest_suite_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

wait_for_gpu
task_index=0
for target in "${TARGET_VALUES[@]}"; do
  case "$target" in
    pads)
      datasets=(PADS_09_task06_DrinkGlas PADS_10_task07_CrossArms)
      for outer_seed in "${SEED_VALUES[@]}"; do
        for dataset in "${datasets[@]}"; do
          for condition in "${AAAI27_CONDITIONS[@]}"; do
            run_aaai27_task "$outer_seed" "$dataset" "$condition" "$task_index"
            task_index=$((task_index + 1))
          done
        done
      done
      ;;
    shimmer)
      datasets=(Shimmer_11_session11_DRINK Shimmer_12_session12_PICK)
      for outer_seed in "${SEED_VALUES[@]}"; do
        for dataset in "${datasets[@]}"; do
          for condition in "${AAAI27_CONDITIONS[@]}"; do
            run_aaai27_task "$outer_seed" "$dataset" "$condition" "$task_index"
            task_index=$((task_index + 1))
          done
        done
      done
      ;;
    falltl)
      for fusion in "${FUSIONS[@]}"; do
        run_falltl_task "$fusion" "$task_index"
        task_index=$((task_index + 1))
      done
      ;;
    ucihar)
      for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        for fusion in "${FUSIONS[@]}"; do
          run_ucihar_task "$repeat" "$fusion" "$task_index"
          task_index=$((task_index + 1))
        done
      done
      ;;
    *) echo "Unsupported target: $target" >&2; exit 1 ;;
  esac
done
