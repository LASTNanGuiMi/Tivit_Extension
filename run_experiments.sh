#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
RESULT_DIR="${RESULT_DIR:-$SERVER_ROOT/results/main_gpu}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WORKER_ID="${WORKER_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
DATASET_GROUPS="${DATASET_GROUPS:-uea ucr feng falltl uci}"
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

dataset_enabled() {
  local requested
  for requested in $DATASET_GROUPS; do
    case "$requested" in
      uea|ucr|feng|falltl|uci) ;;
      *)
        echo "Invalid dataset group in DATASET_GROUPS: $requested" >&2
        exit 1
        ;;
    esac
    if [[ "$requested" == "$1" ]]; then
      return 0
    fi
  done
  return 1
}

mkdir -p "$RESULT_DIR"
COMMON_ARGS=(
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
  --image_mode activity_graph
  --mantis
  --mantis_name "$MANTIS_DIR"
  --classifier_type mlp
  --fusion_dim 512
  --fusion_heads 4
  --cross_attn_query ts
  --mask_prob 0.3
  --pretrain_epochs 5
  --mlp_hidden_dim 512
  --mlp_num_layers 2
  --mlp_dropout 0.1
  --mlp_lr 1e-4
  --mlp_weight_decay 1e-4
  --mlp_epochs 20
  --mlp_early_stop_patience 3
  --batch_size 32
  --data_dir "$DATA_DIR"
  --result_dir "$RESULT_DIR"
  --val_ratio 0.25
)
CUSTOM_DATA_ARGS=(
  --window_size 200
  --window_stride 100
  --custom_test_ratio 0.2
)
FAST_CUSTOM_DATA_ARGS=(
  "${CUSTOM_DATA_ARGS[@]}"
  --max_windows_per_file 2
)
expected_result_rows() {
  case "$1" in
    uea|ucr) echo 2 ;;
    feng|falltl) echo 1 ;;
    *) echo 1 ;;
  esac
}
result_row_count() {
  local file=$1
  awk 'NR > 1 {count++} END {print count + 0}' "$file"
}
result_completed() {
  local dataset=$1
  local fusion=$2
  local dir
  local rows
  local expected_rows

  for dir in "$RESULT_DIR"/*; do
    if [[ -f "$dir/args.json" && -f "$dir/train_val.csv" ]] &&
      grep -q "\"datasets\": \"${dataset}\"" "$dir/args.json" &&
      grep -q "\"modal_interaction\": \"${fusion}\"" "$dir/args.json" &&
      grep -q "\"random_seed\": ${SEED}" "$dir/args.json" &&
      grep -q '"val_ratio": 0.25' "$dir/args.json" &&
      grep -q '"custom_test_ratio": 0.2' "$dir/args.json"; then
      rows=$(result_row_count "$dir/train_val.csv")
      expected_rows=$(expected_result_rows "$dataset")
      if (( rows >= expected_rows )); then
        return 0
      fi
    fi
  done

  return 1
}
run_experiment() {
  local dataset=$1
  local fusion=$2
  local task_index=$TASK_INDEX
  shift 2

  TASK_INDEX=$((TASK_INDEX + 1))
  if (( task_index % NUM_WORKERS != WORKER_ID )); then
    return
  fi

  if result_completed "$dataset" "$fusion"; then
    echo "Skip completed | seed=${SEED} | ${dataset} | fusion=${fusion}"
    return
  fi

  echo "Run worker=${WORKER_ID}/${NUM_WORKERS} | gpu=${CUDA_VISIBLE_DEVICES} | seed=${SEED} | ${dataset} | fusion=${fusion}"
  "$PYTHON_BIN" main.py \
    "${COMMON_ARGS[@]}" \
    --random_seed "$SEED" \
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
RUNS=(1)
TOTAL_RUNS=${#RUNS[@]}
TASK_INDEX=0
for RUN in "${RUNS[@]}"; do
  SEED=2022
  if dataset_enabled uea; then
    for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
      run_experiment uea "$FUSION" \
        --datasets uea \
        --dataset_names BasicMotions SelfRegulationSCP1
    done
  fi

  if dataset_enabled ucr; then
    for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
      run_experiment ucr "$FUSION" \
        --datasets ucr \
        --dataset_names ECG200 FordA
    done
  fi

  if dataset_enabled feng; then
    for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
      run_experiment feng "$FUSION" \
        "${FAST_CUSTOM_DATA_ARGS[@]}" \
        --datasets feng
    done
  fi

  if dataset_enabled falltl; then
    for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
      run_experiment falltl "$FUSION" \
        "${FAST_CUSTOM_DATA_ARGS[@]}" \
        --datasets falltl
    done
  fi

  if dataset_enabled uci; then
    for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
      run_experiment uci "$FUSION" \
        --custom_test_ratio 0.2 \
        --datasets uci
    done
  fi
done
