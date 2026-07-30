#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data/med_data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT.}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID.}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS.}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-2022}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$SEED" != "2022" || "$REPEATS" != "3" ]]; then
  echo "This queue requires SEED=2022 and REPEATS=3; got $SEED/$REPEATS." >&2
  exit 1
fi
if ! [[ "$WORKER_ID" =~ ^[0-9]+$ && "$NUM_WORKERS" =~ ^[1-5]$ ]] ||
  (( WORKER_ID >= NUM_WORKERS )); then
  echo "Invalid worker configuration: $WORKER_ID/$NUM_WORKERS" >&2
  exit 1
fi

for path in "$PROJECT_DIR" "$MODEL_DIR" "$MANTIS_DIR" "$DATA_DIR" "$PYTHON_BIN"; do
  [[ -e "$path" ]] || { echo "Missing path: $path" >&2; exit 1; }
done

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
  --val_ratio 0.25
  --custom_test_ratio 0.2
  --har_channels acc_gyro
)
DATASETS=(flaap uci)
FUSIONS=(concat concat_attn cross_attn_gate masked_pretrain)

result_complete() {
  local task_result=$1
  local expected_dataset=$2
  local fusion=$3
  local args_file
  local csv_file

  for args_file in "$task_result"/*/args.json; do
    csv_file="${args_file%/args.json}/train_val.csv"
    if [[ -s "$csv_file" ]] &&
      grep -q "\"random_seed\": $SEED" "$args_file" &&
      grep -q '"har_channels": "acc_gyro"' "$args_file" &&
      grep -q "\"modal_interaction\": \"$fusion\"" "$args_file" &&
      awk -F, -v expected="$expected_dataset" \
        'NR > 1 && $1 == expected { found = 1 } END { exit !found }' "$csv_file"; then
      return 0
    fi
  done
  return 1
}

run_task() {
  local repeat=$1
  local dataset=$2
  local fusion=$3
  local task_index=$4
  local expected_dataset
  local task_result="$RESULT_ROOT/$dataset/repeat_${repeat}/$fusion"

  if (( task_index % NUM_WORKERS != WORKER_ID )); then
    return
  fi
  case "$dataset" in
    flaap) expected_dataset=FLAAP ;;
    uci) expected_dataset=UCIHAR ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
  esac

  if result_complete "$task_result" "$expected_dataset" "$fusion"; then
    echo "Skip completed | repeat=$repeat | dataset=$dataset | fusion=$fusion"
    return
  fi

  echo "Run | repeat=$repeat/$REPEATS | seed=$SEED | dataset=$dataset | fusion=$fusion | gpu=$CUDA_VISIBLE_DEVICES"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi

  mkdir -p "$task_result"
  "$PYTHON_BIN" "$PROJECT_DIR/main.py" \
    "${COMMON_ARGS[@]}" \
    --datasets "$dataset" \
    --modal_interaction "$fusion" \
    --random_seed "$SEED" \
    --result_dir "$task_result"
}

export CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tivit_har6_mpl_gpu_${CUDA_VISIBLE_DEVICES}}"
export HF_HOME="${HF_HOME:-/tmp/tivit_har6_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

task_index=0
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  for dataset in "${DATASETS[@]}"; do
    for fusion in "${FUSIONS[@]}"; do
      run_task "$repeat" "$dataset" "$fusion" "$task_index"
      task_index=$((task_index + 1))
    done
  done
done
