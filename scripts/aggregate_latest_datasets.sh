#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:?Set PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN}"
RESULT_ROOT="${RESULT_ROOT:?Set RESULT_ROOT}"
STATUS_DIR="${STATUS_DIR:?Set STATUS_DIR}"
NUM_WORKERS="${NUM_WORKERS:?Set NUM_WORKERS}"
TARGETS="${TARGETS:-pads shimmer falltl ucihar}"
OUTER_SEEDS="${OUTER_SEEDS:-2020 2021 2022 2023 2024}"
SPLIT_REFERENCE="${SPLIT_REFERENCE:?Set SPLIT_REFERENCE}"
REPEATS="${REPEATS:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"

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

wait_for_workers() {
  local worker status_file status completed
  while true; do
    completed=0
    for ((worker = 0; worker < NUM_WORKERS; worker++)); do
      status_file="$STATUS_DIR/worker_${worker}.status"
      if [[ -f "$status_file" ]]; then
        status=$(tr -d '\n' < "$status_file")
        if [[ "$status" =~ ^[1-9][0-9]*$ ]]; then
          echo "Worker $worker failed with status $status." >&2
          exit 1
        fi
        if [[ "$status" == "0" ]]; then
          completed=$((completed + 1))
        fi
      fi
    done
    (( completed == NUM_WORKERS )) && return 0
    echo "Workers complete: $completed/$NUM_WORKERS"
    sleep "$POLL_SECONDS"
  done
}

wait_for_workers

aaai27_datasets=()
if target_enabled pads; then
  aaai27_datasets+=(PADS_09_task06_DrinkGlas PADS_10_task07_CrossArms)
fi
if target_enabled shimmer; then
  aaai27_datasets+=(Shimmer_11_session11_DRINK Shimmer_12_session12_PICK)
fi

if (( ${#aaai27_datasets[@]} > 0 )); then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/aggregate_aaai27_outer_seeds.py" \
    --result-root "$RESULT_ROOT/aaai27" \
    --status-dir "$STATUS_DIR" \
    --workers "$NUM_WORKERS" \
    --outer-seeds "${SEED_VALUES[@]}" \
    --split-reference "$SPLIT_REFERENCE" \
    --datasets "${aaai27_datasets[@]}" \
    --poll-seconds 1
fi

if target_enabled falltl; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/aggregate_falltl_binary_main.py" \
    --result-root "$RESULT_ROOT/falltl" \
    --output "$RESULT_ROOT/falltl/summary.csv"
fi

if target_enabled ucihar; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/aggregate_har6_repeats.py" \
    --result-root "$RESULT_ROOT/har6" \
    --repeats "$REPEATS" \
    --status-dir "$STATUS_DIR" \
    --workers "$NUM_WORKERS" \
    --datasets uci \
    --output "$RESULT_ROOT/har6/ucihar_three_repeat_average.csv" \
    --poll-seconds 1
fi

echo "All requested summaries are complete: $RESULT_ROOT"
