#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$PROJECT_DIR/checkpoints/Mantis-8M}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
GPU="${GPU:-0}"
SEED="${SEED:-2022}"
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
DATASET="Shimmer_11_session11_DRINK"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/results/shimmer_example_seed${SEED}}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-$PROJECT_DIR/feature_cache/shimmer_example}"
DRY_RUN="${DRY_RUN:-0}"

for path in \
  "$MODEL_DIR" \
  "$MANTIS_DIR" \
  "$DATA_DIR/med_data/AAAI_Data/$DATASET" \
  "$PROJECT_DIR/data_loading/split_reference_seed42.csv"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || {
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer" >&2; exit 1; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 1; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive" >&2; exit 1; }
[[ "$PATIENCE" =~ ^[1-9][0-9]*$ ]] || { echo "PATIENCE must be positive" >&2; exit 1; }

command=(
  "$PYTHON_BIN" "$PROJECT_DIR/main.py"
  --vit_1_name "$MODEL_DIR"
  --vit_1_layer 14
  --aggregation mean
  --image_mode activity_graph
  --mantis
  --mantis_name "$MANTIS_DIR"
  --classifier_type mlp
  --modal_interaction concat_attn
  --fusion_dim 128
  --fusion_heads 2
  --mlp_hidden_dim 128
  --mlp_num_layers 2
  --mlp_dropout 0.1
  --mlp_lr 3e-4
  --mlp_weight_decay 1e-3
  --mlp_class_weight balanced
  --mlp_epochs "$EPOCHS"
  --mlp_early_stop_patience "$PATIENCE"
  --batch_size 16
  --data_dir "$DATA_DIR"
  --aaai27_label_mode shimmer_hc_vs_pd
  --datasets aaai27
  --dataset_names "$DATASET"
  --random_seed "$SEED"
  --feature_cache_dir "$FEATURE_CACHE_DIR"
  --result_dir "$RESULT_DIR"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$RESULT_DIR" "$FEATURE_CACHE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/neurosigvit_shimmer_mpl_gpu_${GPU}}"
export HF_HOME="${HF_HOME:-/tmp/neurosigvit_shimmer_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
"${command[@]}"
