#!/usr/bin/env bash
set -euo pipefail

SERVER_ROOT="${SERVER_ROOT:-/home/xuzheyuan/guoyin/Tivit/Tivit_extension}"
PROJECT_DIR="${PROJECT_DIR:-$SERVER_ROOT/Tivit_Extension-main}"
MODEL_DIR="${MODEL_DIR:-$SERVER_ROOT/models/CLIP-ViT-H-14-laion2B-s32B-b79K}"
MANTIS_DIR="${MANTIS_DIR:-$SERVER_ROOT/Checkpoint/Checkpoint/models--paris-noah--Mantis-8M/snapshots/93a16a52a5e2e6d76c0b823533b5836dd83ca10a}"
DATA_DIR="${DATA_DIR:-/home/xuzheyuan/guoyin/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/xuzheyuan/miniconda3/envs/tivit_env/bin/python}"
GPUS="${GPUS:?Set GPUS to space-separated physical GPU indices}"
TARGETS="${TARGETS:-pads shimmer falltl ucihar}"
OUTER_SEEDS="${OUTER_SEEDS:-2020 2021 2022 2023 2024}"
SEED="${SEED:-2022}"
REPEATS="${REPEATS:-3}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-tivit_latest_datasets_${TIMESTAMP}}"
RESULT_ROOT="${RESULT_ROOT:-$SERVER_ROOT/results/tivit_latest_datasets_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-$SERVER_ROOT/logs/tivit_latest_datasets_${TIMESTAMP}}"
STATUS_DIR="$LOG_DIR/status"
UNIMODAL_CACHE_ROOT="${UNIMODAL_CACHE_ROOT:-$SERVER_ROOT/feature_cache/aaai27_binary_remaining_ablation_seed2022}"
MULTIMODAL_CACHE_ROOT="${MULTIMODAL_CACHE_ROOT:-$SERVER_ROOT/feature_cache/aaai27_binary_activity_graph_clip_h14_mantis8m}"
FALLTL_CACHE_ROOT="${FALLTL_CACHE_ROOT:-$SERVER_ROOT/feature_cache/falltl_comparison_binary_seed42_main}"
FALLTL_CACHE_STATUS_FILE="$STATUS_DIR/falltl_concat_cache.status"
SPLIT_REFERENCE="${SPLIT_REFERENCE:-$DATA_DIR/med_data/AAAI_Data/data_loading/split_reference_seed42.csv}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
GPU_FREE_MEMORY_MAX_MB="${GPU_FREE_MEMORY_MAX_MB:-2048}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a GPU_VALUES <<< "$GPUS"
read -r -a TARGET_VALUES <<< "$TARGETS"
read -r -a SEED_VALUES <<< "$OUTER_SEEDS"
NUM_WORKERS=${#GPU_VALUES[@]}

if (( NUM_WORKERS < 1 || NUM_WORKERS > 8 )); then
  echo "GPUS must contain between one and eight GPU indices: $GPUS" >&2
  exit 1
fi
if (( ${#TARGET_VALUES[@]} < 1 )); then
  echo "Select at least one target from: pads shimmer falltl ucihar" >&2
  exit 1
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_VALUES[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $gpu" >&2; exit 1; }
  [[ -z "${SEEN_GPUS[$gpu]+x}" ]] || { echo "Duplicate GPU index: $gpu" >&2; exit 1; }
  SEEN_GPUS[$gpu]=1
done

declare -A SEEN_TARGETS=()
has_aaai27=0
has_falltl=0
has_ucihar=0
total_tasks=0
for target in "${TARGET_VALUES[@]}"; do
  [[ -z "${SEEN_TARGETS[$target]+x}" ]] || { echo "Duplicate target: $target" >&2; exit 1; }
  SEEN_TARGETS[$target]=1
  case "$target" in
    pads|shimmer)
      has_aaai27=1
      total_tasks=$((total_tasks + ${#SEED_VALUES[@]} * 2 * 7))
      ;;
    falltl)
      has_falltl=1
      total_tasks=$((total_tasks + 4))
      ;;
    ucihar)
      has_ucihar=1
      total_tasks=$((total_tasks + REPEATS * 4))
      ;;
    *) echo "Unsupported target: $target" >&2; exit 1 ;;
  esac
done

if (( has_aaai27 == 1 )); then
  if (( ${#SEED_VALUES[@]} < 2 )); then
    echo "PADS/Shimmer require at least two outer seeds." >&2
    exit 1
  fi
  declare -A SEEN_SEEDS=()
  for outer_seed in "${SEED_VALUES[@]}"; do
    [[ "$outer_seed" =~ ^[0-9]+$ ]] || { echo "Invalid outer seed: $outer_seed" >&2; exit 1; }
    [[ -z "${SEEN_SEEDS[$outer_seed]+x}" ]] || { echo "Duplicate outer seed: $outer_seed" >&2; exit 1; }
    SEEN_SEEDS[$outer_seed]=1
  done
  [[ "$(basename "$SPLIT_REFERENCE")" == "split_reference_seed42.csv" ]] || {
    echo "AAAI27 split reference must be split_reference_seed42.csv: $SPLIT_REFERENCE" >&2
    exit 1
  }
fi
if (( has_ucihar == 1 )) && [[ "$SEED" != "2022" || "$REPEATS" != "3" ]]; then
  echo "The latest UCIHAR protocol requires SEED=2022 and REPEATS=3." >&2
  exit 1
fi

worker_env=(
  SERVER_ROOT="$SERVER_ROOT"
  PROJECT_DIR="$PROJECT_DIR"
  MODEL_DIR="$MODEL_DIR"
  MANTIS_DIR="$MANTIS_DIR"
  DATA_DIR="$DATA_DIR"
  PYTHON_BIN="$PYTHON_BIN"
  RESULT_ROOT="$RESULT_ROOT"
  STATUS_DIR="$STATUS_DIR"
  UNIMODAL_CACHE_ROOT="$UNIMODAL_CACHE_ROOT"
  MULTIMODAL_CACHE_ROOT="$MULTIMODAL_CACHE_ROOT"
  FALLTL_CACHE_ROOT="$FALLTL_CACHE_ROOT"
  FALLTL_CACHE_STATUS_FILE="$FALLTL_CACHE_STATUS_FILE"
  SPLIT_REFERENCE="$SPLIT_REFERENCE"
  TARGETS="$TARGETS"
  OUTER_SEEDS="$OUTER_SEEDS"
  SEED="$SEED"
  REPEATS="$REPEATS"
  WAIT_FOR_GPU_FREE="$WAIT_FOR_GPU_FREE"
  GPU_FREE_MEMORY_MAX_MB="$GPU_FREE_MEMORY_MAX_MB"
  GPU_POLL_SECONDS="$GPU_POLL_SECONDS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  env "${worker_env[@]}" \
    WORKER_ID=0 \
    NUM_WORKERS=1 \
    PHYSICAL_GPU="${GPU_VALUES[0]}" \
    DRY_RUN=1 \
    bash "$PROJECT_DIR/scripts/run_latest_datasets_worker.sh"
  echo "Dry run complete: $total_tasks tasks across targets: $TARGETS"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p \
  "$RESULT_ROOT" \
  "$LOG_DIR" \
  "$STATUS_DIR"
if (( has_aaai27 == 1 )); then
  mkdir -p "$UNIMODAL_CACHE_ROOT" "$MULTIMODAL_CACHE_ROOT"
fi
if (( has_falltl == 1 )); then
  mkdir -p "$FALLTL_CACHE_ROOT"
fi

for worker in "${!GPU_VALUES[@]}"; do
  printf 'pending\n' > "$STATUS_DIR/worker_${worker}.status"
done
if (( has_falltl == 1 )); then
  printf 'pending\n' > "$FALLTL_CACHE_STATUS_FILE"
fi

for worker in "${!GPU_VALUES[@]}"; do
  gpu=${GPU_VALUES[$worker]}
  log_file="$LOG_DIR/gpu${gpu}.log"
  status_file="$STATUS_DIR/worker_${worker}.status"
  worker_command="cd '$PROJECT_DIR'; set -o pipefail; SERVER_ROOT='$SERVER_ROOT' PROJECT_DIR='$PROJECT_DIR' MODEL_DIR='$MODEL_DIR' MANTIS_DIR='$MANTIS_DIR' DATA_DIR='$DATA_DIR' PYTHON_BIN='$PYTHON_BIN' RESULT_ROOT='$RESULT_ROOT' STATUS_DIR='$STATUS_DIR' UNIMODAL_CACHE_ROOT='$UNIMODAL_CACHE_ROOT' MULTIMODAL_CACHE_ROOT='$MULTIMODAL_CACHE_ROOT' FALLTL_CACHE_ROOT='$FALLTL_CACHE_ROOT' FALLTL_CACHE_STATUS_FILE='$FALLTL_CACHE_STATUS_FILE' SPLIT_REFERENCE='$SPLIT_REFERENCE' TARGETS='$TARGETS' OUTER_SEEDS='$OUTER_SEEDS' SEED='$SEED' REPEATS='$REPEATS' WORKER_ID='$worker' NUM_WORKERS='$NUM_WORKERS' PHYSICAL_GPU='$gpu' WAIT_FOR_GPU_FREE='$WAIT_FOR_GPU_FREE' GPU_FREE_MEMORY_MAX_MB='$GPU_FREE_MEMORY_MAX_MB' GPU_POLL_SECONDS='$GPU_POLL_SECONDS' bash scripts/run_latest_datasets_worker.sh 2>&1 | tee -a '$log_file'; status=\${PIPESTATUS[0]}; printf '%s\n' \"\$status\" > '$status_file'; exit \"\$status\""
  if (( worker == 0 )); then
    tmux new-session -d -s "$SESSION" -n "gpu${gpu}" "$worker_command"
  else
    tmux new-window -d -t "$SESSION" -n "gpu${gpu}" "$worker_command"
  fi
done

summary_command="cd '$PROJECT_DIR'; PROJECT_DIR='$PROJECT_DIR' PYTHON_BIN='$PYTHON_BIN' RESULT_ROOT='$RESULT_ROOT' STATUS_DIR='$STATUS_DIR' NUM_WORKERS='$NUM_WORKERS' TARGETS='$TARGETS' OUTER_SEEDS='$OUTER_SEEDS' SPLIT_REFERENCE='$SPLIT_REFERENCE' REPEATS='$REPEATS' POLL_SECONDS='$GPU_POLL_SECONDS' bash scripts/aggregate_latest_datasets.sh"
tmux new-window -d -t "$SESSION" -n summary "$summary_command"

cat <<EOF
Started tmux session: $SESSION
Scope: TiVit main.py experiments only (external baselines are excluded)
Targets: $TARGETS
GPUs: $GPUS
Total tasks: $total_tasks
PADS/Shimmer: fixed split seed 42; outer seeds $OUTER_SEEDS; seven conditions
FallTL: comparison_binary; seed 2022; four fusion modes
UCIHAR: six channels; seed 2022; three repeats; four fusion modes
Results: $RESULT_ROOT
Logs: $LOG_DIR
Summaries are generated automatically after all workers finish.
EOF
