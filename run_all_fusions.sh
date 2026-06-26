#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="/home/guoyin/guoyin/hf_models/CLIP-ViT-H-14-laion2B-s32B-b79K"
MANTIS_DIR="$HOME/guoyin/hf_models/Mantis-8M/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a"
DATA_DIR="$HOME/guoyin/dataset"
RESULT_DIR="/tmp/dl/tivit_results"
COMMON_ARGS="
  --vit_1_name ${MODEL_DIR}
  --vit_1_layer 14
  --aggregation mean
  --image_mode activity_graph
  --mantis
  --mantis_name ${MANTIS_DIR}
  --classifier_type mlp
  --fusion_dim 512
  --fusion_heads 4
  --cross_attn_query ts
  --mask_prob 0.3
  --pretrain_epochs 10
  --mlp_hidden_dim 512
  --mlp_num_layers 2
  --mlp_dropout 0.1
  --mlp_lr 1e-4
  --mlp_weight_decay 1e-4
  --mlp_epochs 20
  --batch_size 16
  --data_dir ${DATA_DIR}
  --result_dir ${RESULT_DIR}
  --random_seed 2021
  --val_ratio 0.2
"
CUSTOM_DATA_ARGS="
  --window_size 200
  --window_stride 100
  --custom_test_ratio 0.2
"
for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
    echo "Running UEA with fusion=${FUSION}"
    python3 main.py \
      ${COMMON_ARGS} \
      --modal_interaction "${FUSION}" \
      --datasets uea \
      --dataset_names BasicMotions SelfRegulationSCP1

    echo "Running UCR with fusion=${FUSION}"
    python3 main.py \
      ${COMMON_ARGS} \
      --modal_interaction "${FUSION}" \
      --datasets ucr \
      --dataset_names ECG200 FordA

    echo "Running FallTL with fusion=${FUSION}"
    python3 main.py \
      ${COMMON_ARGS} \
      ${CUSTOM_DATA_ARGS} \
      --modal_interaction "${FUSION}" \
      --datasets falltl

    echo "Running Feng with fusion=${FUSION}"
    python3 main.py \
      ${COMMON_ARGS} \
      ${CUSTOM_DATA_ARGS} \
      --max_windows_per_file 1000 \
      --modal_interaction "${FUSION}" \
      --datasets feng
done
