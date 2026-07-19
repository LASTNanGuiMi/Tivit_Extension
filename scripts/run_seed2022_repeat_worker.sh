#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension/Tivit_Extension-main}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT.}"
REPEATS="${REPEATS:-3}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID.}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS.}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES.}"

if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer: $REPEATS" >&2
  exit 1
fi

cd "$PROJECT_DIR"
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  repeat_dir="$RESULT_ROOT/repeat_${repeat}"
  mkdir -p "$repeat_dir"
  echo "Repeat $repeat/$REPEATS | seed=2022 | worker=$WORKER_ID/$NUM_WORKERS | gpu=$CUDA_VISIBLE_DEVICES"
  DATASET_GROUPS="feng falltl uci" \
    RESULT_DIR="$repeat_dir" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    WORKER_ID="$WORKER_ID" \
    NUM_WORKERS="$NUM_WORKERS" \
    bash ./run_experiments.sh
done
