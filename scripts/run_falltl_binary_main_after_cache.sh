#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT}"
LOG_DIR="${LOG_DIR:?Set LOG_DIR}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:?Set FEATURE_CACHE_ROOT}"
CACHE_STATUS_FILE="${CACHE_STATUS_FILE:?Set CACHE_STATUS_FILE}"
GPUS="${GPUS:-3 4 7}"
POLL_SECONDS="${POLL_SECONDS:-30}"

FUSIONS=(concat_attn cross_attn_gate masked_pretrain)
read -r -a GPU_VALUES <<< "$GPUS"
if (( ${#GPU_VALUES[@]} != ${#FUSIONS[@]} )); then
  echo "GPUS must provide exactly three devices: $GPUS" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT" "$LOG_DIR" "$FEATURE_CACHE_ROOT"
echo "Waiting for concat/cache producer: $CACHE_STATUS_FILE"
while [[ ! -f "$CACHE_STATUS_FILE" ]]; do
  sleep "$POLL_SECONDS"
done
cache_status=$(tr -d '\n' < "$CACHE_STATUS_FILE")
if [[ "$cache_status" != "0" ]]; then
  echo "Concat/cache producer failed with status $cache_status" >&2
  exit 1
fi
echo "Shared feature cache is ready; starting three fusion workers."

run_fusion() {
  local fusion=$1
  local gpu=$2
  local result_dir="$RESULT_ROOT/$fusion"
  local log_file="$LOG_DIR/${fusion}_gpu${gpu}.log"
  local status_file="$LOG_DIR/${fusion}.status"

  set +e
  CUDA_VISIBLE_DEVICES="$gpu" \
    MPLCONFIGDIR="/tmp/tivit_falltl_mpl_gpu${gpu}" \
    HF_HOME=/tmp/tivit_falltl_hf \
    TRANSFORMERS_CACHE=/tmp/tivit_falltl_hf \
    PYTHONWARNINGS=ignore \
    "$PYTHON_BIN" -u "$PROJECT_DIR/main.py" \
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
      --feature_cache_dir "$FEATURE_CACHE_ROOT" \
      --result_dir "$result_dir" > "$log_file" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$status_file"
  return "$status"
}

pids=()
for index in "${!FUSIONS[@]}"; do
  run_fusion "${FUSIONS[$index]}" "${GPU_VALUES[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
(( status == 0 )) || exit "$status"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/aggregate_falltl_binary_main.py" \
  --result-root "$RESULT_ROOT" \
  --output "$RESULT_ROOT/summary.csv"
