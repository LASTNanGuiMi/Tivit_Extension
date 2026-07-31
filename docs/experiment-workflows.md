# Experiment Workflows

This document indexes the maintained experiment scripts. Launchers create output
directories, distribute tasks, skip validated completed runs, and invoke a
matching aggregator when applicable. Worker scripts are implementation details
unless a workflow explicitly says otherwise.

## Common Configuration

Most launchers accept these environment variables:

| Variable | Purpose |
| --- | --- |
| `SERVER_ROOT` | Parent directory for the project, models, results, and logs |
| `PROJECT_DIR` or `PROJECT_ROOT` | Repository checkout |
| `MODEL_DIR` | Vision checkpoint |
| `MANTIS_DIR` | Mantis checkpoint |
| `DATA_DIR` | Dataset root |
| `PYTHON_BIN` | Python interpreter for the intended environment |
| `GPUS` | Space-separated physical GPU indices |
| `RESULT_ROOT` or `RESULT_DIR` | Result destination |
| `LOG_DIR` or `LOG_ROOT` | Log destination |
| `TIMESTAMP`, `SESSION` | Optional deterministic names for resuming a queue |

The scripts contain research-server defaults for convenience. Always override
them on another machine. Generated results, logs, model files, and feature caches
must remain outside version control.

## Latest TiVit Four-Dataset Suite

Use `scripts/run_latest_datasets_tmux.sh` as the single entry point for the
TiVit submission experiments on PADS, Shimmer, FallTL, and UCIHAR. This suite
only invokes the local `main.py`; Medformer and all other external baselines are
intentionally excluded. The defaults were copied from the most recent completed
TiVit experiment for each dataset family, rather than normalizing the four
families to one hyperparameter set:

| Target | Reference experiment | Reproduced protocol |
| --- | --- | --- |
| `pads` | `aaai27_outer_seed5_split42_rngreset_20260729` | PADS 9/10, fixed subject split seed 42, outer seeds 2020-2024, seven ablation conditions |
| `shimmer` | `aaai27_outer_seed5_split42_rngreset_20260729` | Shimmer 11/12, fixed subject split seed 42, outer seeds 2020-2024, seven ablation conditions |
| `falltl` | `falltl_comparison_binary_main_seed2022_20260729` | Comparison-binary labels, seed 2022, four fusion modes, shared frozen-feature cache |
| `ucihar` | `har6_seed2022_3repeat_20260725_152033` | Total acceleration XYZ plus body gyroscope XYZ, seed 2022, three repeats, four fusion modes |

Run the complete 156-task suite with one GPU pool:

```bash
GPUS="0 1 2 3 4" \
TARGETS="pads shimmer falltl ucihar" \
WAIT_FOR_GPU_FREE=1 \
bash scripts/run_latest_datasets_tmux.sh
```

`TARGETS` can select any subset without changing its protocol. For example,
`TARGETS="falltl ucihar"` runs only those two families. Use `DRY_RUN=1` to print
the exact task queue without creating a tmux session or starting training:

```bash
GPUS="0" TARGETS="pads" DRY_RUN=1 \
bash scripts/run_latest_datasets_tmux.sh
```

Results are grouped under `aaai27/`, `falltl/`, and `har6/` in one result root.
The matching aggregators run automatically after every worker succeeds. FallTL's
concat task is the cache producer; the other three fusion tasks wait for it.

## General Benchmarks

| Launcher | Scope | Notes |
| --- | --- | --- |
| `run_experiments.sh` | UEA, UCR, Feng, FallTL, and UCI-HAR | Main 512-dimensional Activity Graph + Mantis configuration; seed 2022 |
| `run_small_data_experiments.sh` | BasicMotions, ECG200, Feng, and UCI-HAR | Smaller 128-dimensional configuration; seed 2022 |
| `scripts/run_seed2022_repeats_tmux.sh` | Feng, FallTL, and UCI-HAR | Repeats the main queue across GPUs and runs `aggregate_seed2022_repeats.py` |

`run_experiments.sh` can be partitioned manually with `WORKER_ID` and
`NUM_WORKERS`, or limited with `DATASET_GROUPS`, for example:

```bash
DATASET_GROUPS="feng falltl uci" \
WORKER_ID=0 \
NUM_WORKERS=2 \
CUDA_VISIBLE_DEVICES=0 \
bash run_experiments.sh
```

## Ablations

The general feature ablation consists of five unique conditions:

| Condition | Vision input | Time-series branch | Fusion |
| --- | --- | --- | --- |
| `vision_line_plot` | Multichannel line plot | None | Single branch |
| `vision_activity_graph` | Activity Graph | None | Single branch |
| `timeseries_mantis` | None | Mantis | Single branch |
| `multimodal_concat` | Activity Graph | Mantis | Concatenation |
| `multimodal_proposed` | Activity Graph | Mantis | Gated cross-attention by default |

Use `scripts/run_ablation_tmux.sh` for the full configurable queue. Shared
defaults live in `scripts/ablation_config.sh`; `ABLATION_CONDITIONS` can select a
subset. This protocol intentionally requires seed 2022.

`scripts/run_activity_graph_ablation_tmux.sh` is the focused launcher. Set
`ABLATION_KIND=image` for line plot versus Activity Graph, or
`ABLATION_KIND=fusion` for concatenation versus the proposed fusion.

```bash
GPUS="0 1 2" \
ABLATION_KIND=image \
SEED=2022 \
REPEATS=3 \
bash scripts/run_activity_graph_ablation_tmux.sh
```

## Six-Channel HAR

`scripts/run_har6_tmux.sh` evaluates FLAAP and UCI-HAR using six channels:

- FLAAP: acceleration XYZ plus gyroscope XYZ.
- UCI-HAR: total acceleration XYZ plus body gyroscope XYZ.

The fixed queue uses seed 2022, three repeats, and all four fusion modes. It
automatically invokes `scripts/aggregate_har6_repeats.py`.

## AAAI27

| Launcher | Purpose | Aggregator |
| --- | --- | --- |
| `scripts/run_aaai27_binary_tmux.sh` | Four PADS/Shimmer datasets, binary labels, four fusion modes, seed 2022 | `aggregate_aaai27_binary.py` |
| `scripts/run_aaai27_remaining_ablation_tmux.sh` | Vision representation and unimodal/multimodal feature ablations | `aggregate_aaai27_remaining_ablation.py` |
| `scripts/run_aaai27_outer_seed_tmux.sh` | Fixed subject split with multiple outer training seeds and seven conditions | `aggregate_aaai27_outer_seeds.py` |

These workflows use balanced MLP loss and frozen-feature caches. The outer-seed
workflow keeps the subject split fixed at seed 42 while varying classifier/fusion
initialization and training order through `OUTER_SEEDS`.

Example:

```bash
GPUS="0 1 2 3" \
OUTER_SEEDS="2020 2021 2022 2023 2024" \
WAIT_FOR_GPU_FREE=1 \
bash scripts/run_aaai27_outer_seed_tmux.sh
```

`scripts/run_medformer_aaai27_tmux.sh` is a separate baseline workflow. It
requires an external Medformer checkout and a dedicated compatible environment;
the local adapter is `scripts/run_medformer_aaai27_binary.py`.

## FallTL Comparison Protocol

Use `--falltl_protocol comparison_binary` in `main.py` for the fixed binary
protocol. `scripts/run_falltl_binary_main_after_cache.sh` is a continuation helper:
it waits for a completed concat/cache producer, then runs the remaining three
fusion modes and calls `aggregate_falltl_binary_main.py`. It requires
`RESULT_ROOT`, `LOG_DIR`, `FEATURE_CACHE_ROOT`, and `CACHE_STATUS_FILE`.

## Aggregation Tools

Aggregation scripts validate configurations before accepting results. They reject
missing, duplicate, or protocol-incompatible runs rather than averaging whatever
files happen to exist.

| Script | Input |
| --- | --- |
| `aggregate_ablation.py` | General ablation result tree |
| `aggregate_seed2022_repeats.py` | Repeated main benchmark runs |
| `aggregate_har6_repeats.py` | Six-channel HAR repeats |
| `aggregate_aaai27_binary.py` | AAAI27 binary fusion runs |
| `aggregate_aaai27_remaining_ablation.py` | AAAI27 representation/feature ablations |
| `aggregate_aaai27_outer_seeds.py` | AAAI27 fixed-split outer seeds |
| `aggregate_falltl_binary_main.py` | FallTL comparison-binary fusion runs |
| `aggregate_medformer_aaai27.py` | External Medformer baseline runs |

## Validation Tools

The following tests are standalone scripts rather than a pytest suite:

```bash
python scripts/test_feature_cache.py
python scripts/test_mlp_fixed_split.py
python scripts/test_ablation_image_modes.py
python scripts/test_custom_datasets.py --data_dir /path/to/data
python scripts/test_har_channels.py --data-dir /path/to/med_data
python scripts/test_aaai27_datasets.py --data-dir /path/to/AAAI_Data
```

Supporting data utilities:

- `scripts/check_ts_file.py` inspects custom time-series files.
- `scripts/repair_ts_labels.py` repairs malformed class-label separators.

Before starting a long GPU queue, run the relevant dataset validation script and
the launcher's Shell syntax check.
