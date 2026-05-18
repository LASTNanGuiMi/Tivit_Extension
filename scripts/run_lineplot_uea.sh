#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/guoyin/TiViT-main}"
MODEL_DIR="${MODEL_DIR:-/home/guoyin/hf_models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
DATA_DIR="${DATA_DIR:-/home/guoyin/dmmv_extension/dmmv/dmmv/dataset}"
RESULT_DIR="${RESULT_DIR:-/home/guoyin/TiViT-main/results}"
DATASETS="${DATASETS:-BasicMotions SelfRegulationSCP1}"

cd "$PROJECT_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

python main.py \
  --vit_1_name "$MODEL_DIR" \
  --vit_1_layer 14 \
  --aggregation mean \
  --patch_size sqrt \
  --stride 0.1 \
  --classifier_type logistic_regression \
  --datasets uea \
  --dataset_names $DATASETS \
  --data_dir "$DATA_DIR" \
  --result_dir "$RESULT_DIR" \
  --random_seed 2021 \
  --val_ratio 0.2 \
  --image_mode line_plot
