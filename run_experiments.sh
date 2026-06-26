#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="/home/guoyin/guoyin/hf_models/CLIP-ViT-H-14-laion2B-s32B-b79K"
MANTIS_DIR="$HOME/guoyin/hf_models/Mantis-8M/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a"
DATA_DIR="$HOME/guoyin/dataset"
RESULT_DIR="/tmp/dl/tivit_results"
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
  --val_ratio 0.2
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
result_completed() {
  local dataset=$1
  local fusion=$2
  local dir

  for dir in "$RESULT_DIR"/*; do
    if [[ -f "$dir/args.json" && -f "$dir/train_val.csv" ]] &&
      grep -q "\"datasets\": \"${dataset}\"" "$dir/args.json" &&
      grep -q "\"modal_interaction\": \"${fusion}\"" "$dir/args.json" &&
      grep -q "\"random_seed\": 2021" "$dir/args.json"; then
      return 0
    fi
  done

  return 1
}
run_experiment() {
  local dataset=$1
  local fusion=$2
  shift 2

  if result_completed "$dataset" "$fusion"; then
    echo "Skip completed | seed=2021 | ${dataset} | fusion=${fusion}"
    return
  fi

  echo "Run ${RUN}/${TOTAL_RUNS} | seed=${SEED} | ${dataset} | fusion=${fusion}"
  python3 main.py \
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
RUNS=(1)
TOTAL_RUNS=${#RUNS[@]}
for RUN in "${RUNS[@]}"; do
  SEED=2021
  for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
    run_experiment uea "$FUSION" \
      --datasets uea \
      --dataset_names BasicMotions SelfRegulationSCP1
  done

  for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
    run_experiment ucr "$FUSION" \
      --datasets ucr \
      --dataset_names ECG200 FordA
  done

  for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
    run_experiment feng "$FUSION" \
      "${FAST_CUSTOM_DATA_ARGS[@]}" \
      --datasets feng
  done

  for FUSION in concat concat_attn cross_attn_gate masked_pretrain; do
    run_experiment falltl "$FUSION" \
      "${FAST_CUSTOM_DATA_ARGS[@]}" \
      --datasets falltl
  done
done
