# TiViT Extension

TiViT Extension is a multimodal time-series classification research codebase. It
turns multichannel sensor sequences into visual representations, extracts frozen
features from vision and time-series foundation models, and trains classical or
MLP fusion classifiers.

![Activity Graph methodology](assets/methodology.svg)

## Features

- Visual inputs: per-channel line plots, multichannel line plots, Activity Graph,
  activity matrices, and segmented views.
- Feature backbones: CLIP/ViT-compatible vision models, Mantis-8M, and MOMENT.
- Fusion: direct concatenation, attention after concatenation, gated
  cross-attention, and masked-branch pretraining.
- Data adapters: UCR, UEA, UCI-HAR, FLAAP, Feng, FallTL, and seven fixed-split
  AAAI27 datasets.
- Reproducibility: split audit files, frozen-feature caches, class-balanced MLP
  loss, fixed-protocol launchers, and result aggregators.

## Repository Layout

```text
.
|-- main.py                       # Main experiment entry point
|-- src/                          # Models, data adapters, training, and metrics
|-- scripts/                      # Validation, launch, worker, and aggregation tools
|-- docs/experiment-workflows.md  # Script catalog and reproducible workflows
|-- assets/methodology.svg        # Method overview
|-- run_experiments.sh            # Main seed-2022 benchmark queue
|-- run_small_data_experiments.sh # Smaller seed-2022 benchmark queue
|-- requirements.txt              # Full environment, including Git dependencies
`-- requirements_no_git.txt       # PyPI-only subset
```

Generated data, checkpoints, logs, feature caches, and results are intentionally
excluded from Git.

## Installation

Python 3.11 is recommended.

```bash
conda create -n tivit_env python=3.11
conda activate tivit_env
python -m pip install -r requirements.txt
```

`requirements.txt` installs Mantis and MOMENT from their Git repositories. Use
`requirements_no_git.txt` only when compatible copies of those packages are
already available in the environment.

The experiment launchers default to paths from the original research server.
Override them instead of editing the scripts:

```bash
export SERVER_ROOT=/path/to/tivit-workspace
export MODEL_DIR=/path/to/CLIP-ViT-H-14-laion2B-s32B-b79K
export MANTIS_DIR=/path/to/Mantis-8M
export DATA_DIR=/path/to/data
export PYTHON_BIN=/path/to/tivit_env/bin/python
```

## Quick Start

At least one feature branch is required. The following example combines an
Activity Graph vision branch with Mantis and trains a gated fusion head:

```bash
python main.py \
  --vit_1_name /path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --mantis \
  --mantis_name /path/to/Mantis-8M \
  --classifier_type mlp \
  --modal_interaction cross_attn_gate \
  --datasets uea \
  --dataset_names BasicMotions \
  --data_dir /path/to/data \
  --result_dir /path/to/results \
  --random_seed 2022 \
  --val_ratio 0.25
```

Inspect the full command-line interface with:

```bash
python main.py --help
```

## Data Adapters

All model batches use `(batch, channels, time)`.

| `--datasets` | Input | Split behavior |
| --- | --- | --- |
| `ucr`, `uea` | aeon classification archives | Combine the source splits and re-split with `--custom_test_ratio` |
| `uci` | Raw UCI-HAR files or preprocessed `Feature/feature.npy` and `Label/label.npy` | Re-split; `--har_channels acc_gyro` selects total acceleration and body gyroscope XYZ |
| `flaap` | Preprocessed FLAAP NumPy arrays | Re-split; uses its six acceleration/gyroscope channels |
| `feng` | Sensor CSV files | Build windows from contiguous activity segments |
| `falltl` | FallTL CSV files | Use legacy windows or the fixed comparison protocol |
| `aaai27` | AAAI27 dataset directories and reference loader | Preserve fixed subject-level train/validation/test splits |

Feng and legacy FallTL support `--window_size`, `--window_stride`, and
`--max_windows_per_file`. The `comparison_binary` FallTL protocol creates one
length-256 sequence per CSV, maps `D=0` and `F=1`, and uses the fixed seed-42
60/20/20 split.

AAAI27 supports these dataset names:

```text
mPowerRest
mPowerReturn
mPowerOutbound
PADS_09_task06_DrinkGlas
PADS_10_task07_CrossArms
Shimmer_11_session11_DRINK
Shimmer_12_session12_PICK
```

Use `--aaai27_label_mode zero_vs_rest` for the binary `0` versus `1/2` mapping.
The adapter validates the reference loader and expected split sizes before
training.

## Training and Caching

The MLP path freezes the selected feature backbones, extracts each data split
once, and trains the fusion module and classifier head. Relevant controls include:

- `--modal_interaction`: `concat`, `concat_attn`, `cross_attn_gate`, or
  `masked_pretrain`.
- `--mlp_class_weight balanced`: inverse-frequency training weights.
- `--mlp_early_stop_patience`: validation macro-F1 early stopping.
- `--feature_cache_dir`: reuse frozen split features when model, data, labels,
  and branch configuration match.

Cache metadata is validated before reuse. A mismatched cache fails explicitly
instead of silently loading incompatible features.

## Reproducible Workflows

The repository contains several GPU/tmux launchers for the maintained experiment
protocols. Start with [the workflow catalog](docs/experiment-workflows.md) before
running anything under `scripts/`; worker scripts are generally launched by their
matching queue script and expect environment variables from it.

For the general queue:

```bash
SERVER_ROOT=/path/to/workspace \
DATA_DIR=/path/to/data \
MODEL_DIR=/path/to/clip \
MANTIS_DIR=/path/to/mantis \
PYTHON_BIN=/path/to/python \
bash run_experiments.sh
```

For multi-GPU workflows, provide physical GPU indices as a space-separated list,
for example `GPUS="0 1 2 3"`.

## Outputs

Each `main.py` invocation creates a timestamped directory under `--result_dir`.

| File | Content |
| --- | --- |
| `args.json` | Complete run configuration |
| `train_val.csv` | Validation and test metrics |
| `splits/*.npz` | Random train/validation indices for re-split datasets |
| `splits/*_split.csv` | Fixed-protocol subject/file split audit |
| `activity_graph_samples/*.png` | Optional Activity Graph samples |
| `activity_lineplot_samples/*.png` | Optional line-plot samples |

Reported metrics include accuracy, macro precision, macro recall, macro F1,
macro AUROC, and macro AUPRC.

## Validation

Lightweight checks that do not require downloaded datasets:

```bash
python -m compileall -q main.py src scripts
bash -n run_experiments.sh run_small_data_experiments.sh scripts/*.sh
python scripts/test_feature_cache.py
python scripts/test_mlp_fixed_split.py
python scripts/test_ablation_image_modes.py
```

Dataset-backed checks are documented in
[the workflow catalog](docs/experiment-workflows.md#validation-tools).

## License

See [LICENSE](LICENSE).
