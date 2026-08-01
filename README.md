# NeuroSigViT

Anonymous implementation of **NeuroSigViT**, the frozen visual-temporal model
described in *Representations for Wearable Sensor-Based Parkinson's Disease
Assessment Lie Hidden in Pretrained Vision Transformers*.

The repository provides one checked, self-contained experiment example:
`Shimmer_11_session11_DRINK`. The processed Shimmer example and its fixed
subject-disjoint split are included. CLIP ViT-H/14 and Mantis-8M checkpoints
must be downloaded separately.

## Reviewer quick start

After cloning the repository on a CUDA-capable Linux machine, the following
commands create the environment, download the two public checkpoints to the
paths expected by the Shell launcher, validate the artifact, and start the
Shimmer run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_dir='checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='paris-noah/Mantis-8M', revision='93a16a52a5e2e6d76c0b823533b5836dd83ca10a', local_dir='checkpoints/Mantis-8M')"

python scripts/test_paper_protocol.py
python scripts/test_aaai27_datasets.py --data-dir data
bash scripts/run_shimmer_example.sh
```

Once the environment and checkpoints exist, the **one-command reproduction
entry point** is simply:

```bash
bash scripts/run_shimmer_example.sh
```

The launcher contains the complete paper-aligned Python command. Reviewers do
not need to reconstruct arguments manually; the expanded command is also shown
later for transparency.

## Method overview

<p align="center">
  <img src="./assets/neurosigvit_method.jpg" alt="NeuroSigViT architecture" width="100%">
</p>

If the embedded preview is disabled by a Markdown editor, open the
[full-resolution method figure](./assets/neurosigvit_method.jpg) directly.

NeuroSigViT represents each six-channel wearable sample through two frozen
branches:

1. **Activity-graph visual branch.** A deterministic pair-covering channel walk
   renders one multicolumn activity graph. A frozen CLIP ViT-H/14 is read at
   transformer layer 14. Patch tokens are mean-pooled without the CLS token,
   passed through the frozen OpenCLIP post-norm and projection, and
   L2-normalized to a 1024-dimensional vector.
2. **Numerical temporal branch.** Frozen Mantis-8M encodes each original signal
   channel. The six channel embeddings are mean-pooled and L2-normalized to one
   512-dimensional vector.
3. **Learned interaction.** Both branch vectors are projected to 128 dimensions
   and treated as two modality tokens. Two-head self-attention updates the two
   tokens, which are flattened and classified by a two-layer MLP with dropout
   0.1.

Only the projection, two-token attention, and classification layers are
trained. Both pretrained encoders remain frozen.

> **Implementation contract.** The supplied manuscript artwork contains legacy
> `R^1028`/`R^256` annotations and an all-token averaging expression. The code
> and experiment command in this repository use the corrected contract above:
> 1024 projected visual dimensions, 512 channel-pooled temporal dimensions, and
> patch-only visual pooling with the CLS token excluded.

## Repository contents

```text
.
|-- main.py
|-- src/                              # rendering, encoders, fusion, training
|-- assets/
|   `-- neurosigvit_method.jpg        # method overview
|-- data/
|   `-- med_data/AAAI_Data/
|       `-- Shimmer_11_session11_DRINK/
|           |-- Feature/              # 130 anonymous float32 samples
|           |-- Label/label.npy
|           `-- Meta/                 # channels and anonymous subject map
|-- data_loading/
|   |-- datasets.py                   # Shimmer-only subject-aware loader
|   `-- split_reference_seed42.csv    # fixed 77/25/28 assignment
|-- scripts/
|   |-- run_shimmer_example.sh        # convenient launcher
|   `-- test_*.py
`-- requirements.txt
```

Generated results, logs, and feature caches are ignored by Git.

## Checkpoints

| Component | Required checkpoint | Official link |
| --- | --- | --- |
| Visual encoder | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | [CLIP ViT-H/14 checkpoint](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) |
| Temporal encoder | `paris-noah/Mantis-8M` | [Mantis-8M checkpoint](https://huggingface.co/paris-noah/Mantis-8M) |

The visual model used by NeuroSigViT is therefore **LAION OpenCLIP ViT-H/14,
trained on LAION-2B with the `s32B-b79K` recipe**. It is not the smaller ViT-B
or ViT-L checkpoint.

Download both repositories with `huggingface_hub`:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_dir='checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K')"

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='paris-noah/Mantis-8M', revision='93a16a52a5e2e6d76c0b823533b5836dd83ca10a', local_dir='checkpoints/Mantis-8M')"
```

The pinned Mantis revision above is the revision used by the validated server
environment. A Hugging Face account is normally not required for these public
repositories, but large-file downloads need working Git LFS/Xet or
`huggingface_hub` support.

The download commands deliberately create the launcher's default paths:

```text
checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/
checkpoints/Mantis-8M/
```

## Dataset availability

Only the processed Shimmer DRINK example is used by the maintained experiment.
The remaining links are provided for reference to the datasets discussed in the
paper; this repository does not claim that those external downloads are already
converted to the required tensor layout.

| Dataset | Availability | Used by the included example? |
| --- | --- | --- |
| Shimmer / PDWearML | [IEEE DataPort dataset page](https://ieee-dataport.org/documents/pdwearml-leveraging-daily-activities-rapid-free-living-parkinsons-disease-severity) and [PDWearML code](https://github.com/wang-xulong/PDWearML) | Yes; the processed DRINK subset is bundled |
| FallTL | [Zenodo record 17552449](https://zenodo.org/records/17552449) | No |
| PADS | [PhysioNet PADS v1.0.0](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) | No |
| UCI-HAR | [Google Drive download](https://drive.google.com/file/d/13HA6l3dnOm46dN4EgzS_YRwHUGIEruKD/view?usp=drive_link) · [official UCI page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | No |

Users must follow the original dataset licenses and terms. Bundling the processed
Shimmer example does not transfer ownership of or relicense the source dataset.

## Included Shimmer task

The bundled sample root is:

```text
data/med_data/AAAI_Data/Shimmer_11_session11_DRINK/
```

Each sample contains six right-wrist inertial channels and 4096 time steps. The
maintained binary endpoint is:

- class 0: healthy controls (`HC`);
- class 1: Parkinson's disease (`MildPD` and `ModeratePD` merged).

This is an HC-versus-PD-status task on a severity-annotated cohort, not a
MildPD-versus-ModeratePD classifier. The fixed split contains 77 training, 25
validation, and 28 test subjects. Channel standardization statistics are fitted
on the selected training split only and then applied to validation and test
data.

## Environment setup

The code was tested end to end in the following environment:

| Component | Validated value |
| --- | --- |
| Operating system | Ubuntu 22.04 / Linux kernel 5.15, glibc 2.35 |
| Python | 3.11.15 |
| GPU | NVIDIA GeForce RTX 4090 |
| NVIDIA driver | 535.309.01 |
| PyTorch | 2.7.1 (`torch.version.cuda == 12.6`) |
| torchvision | 0.22.1 |
| OpenCLIP | 2.32.0 |
| Mantis TSFM | 1.0.0 |
| Transformers | 4.33.3 |

`requirements.txt` is exported from this validated environment and pins every
top-level project dependency, including the checkpoint download client. It
contains no floating Git branch or unpinned Git URL. Install it in a clean
Python 3.11 environment rather than mixing it into an existing research
environment.

Create an isolated environment and install the packaged dependency file:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda is also supported:

```bash
conda create -n neurosigvit python=3.11 -y
conda activate neurosigvit
python -m pip install -r requirements.txt
```

Confirm that PyTorch sees the GPU:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run with the shell launcher

If the checkpoints were downloaded to the default `checkpoints/` paths above,
no path variables are required. The data path also defaults to the bundled
`data/` directory. The complete reproduction command is:

```bash
bash scripts/run_shimmer_example.sh
```

To use checkpoints or a Python environment stored elsewhere, override them:

```bash
export MODEL_DIR="$PWD/checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K"
export MANTIS_DIR="$PWD/checkpoints/Mantis-8M"
export PYTHON_BIN="$PWD/.venv/bin/python"
export GPU=0
```

Inspect the command without running training:

```bash
DRY_RUN=1 bash scripts/run_shimmer_example.sh
```

The default command runs 40 epochs with seed 2022 and early-stopping patience
8. It writes results and feature caches to ignored repository-local folders.

For a short end-to-end smoke test:

```bash
EPOCHS=1 PATIENCE=1 \
RESULT_DIR=/tmp/neurosigvit_smoke_result \
FEATURE_CACHE_DIR=/tmp/neurosigvit_smoke_cache \
bash scripts/run_shimmer_example.sh
```

Supported launcher overrides are `DATA_DIR`, `GPU`, `SEED`, `EPOCHS`,
`PATIENCE`, `PYTHON_BIN`, `RESULT_DIR`, and `FEATURE_CACHE_DIR`.

## Run directly with Python

The following is the complete command executed by the launcher. It can be
copied directly into a Bash terminal after setting `MODEL_DIR` and `MANTIS_DIR`:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --vit_1_name "$MODEL_DIR" \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode activity_graph \
  --mantis \
  --mantis_name "$MANTIS_DIR" \
  --classifier_type mlp \
  --modal_interaction concat_attn \
  --fusion_dim 128 \
  --fusion_heads 2 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 2 \
  --mlp_dropout 0.1 \
  --mlp_lr 3e-4 \
  --mlp_weight_decay 1e-3 \
  --mlp_class_weight balanced \
  --mlp_epochs 40 \
  --mlp_early_stop_patience 8 \
  --batch_size 16 \
  --data_dir data \
  --aaai27_label_mode shimmer_hc_vs_pd \
  --datasets aaai27 \
  --dataset_names Shimmer_11_session11_DRINK \
  --random_seed 2022 \
  --feature_cache_dir feature_cache/shimmer_example \
  --result_dir results/shimmer_example_seed2022
```

Important settings:

| Argument | Meaning |
| --- | --- |
| `--vit_1_layer 14` | Reads the frozen ViT-H/14 representation at layer 14 |
| `--aggregation mean` | Mean-pools patch tokens while excluding CLS |
| `--image_mode activity_graph` | Uses the pair-covering activity-graph rendering |
| `--mantis` | Enables the frozen numerical time-series branch |
| `--modal_interaction concat_attn` | Applies learned attention to the two modality tokens |
| `--aaai27_label_mode shimmer_hc_vs_pd` | Maps HC to 0 and Mild/Moderate PD to 1 |
| `--mlp_class_weight balanced` | Computes class weights from the training split |

## Validation

Run these checks before starting a long job:

```bash
python -m compileall -q main.py src data_loading scripts
bash -n scripts/run_shimmer_example.sh
python scripts/test_feature_cache.py
python scripts/test_mlp_fixed_split.py
python scripts/test_paper_protocol.py
python scripts/test_aaai27_datasets.py --data-dir data
```

The checks cover:

- fixed subject split and HC-vs-PD label mapping;
- six-channel `(B, 6, 4096)` input layout;
- training-only standardization statistics;
- feature-cache consistency;
- CLS exclusion and the 1024-dimensional visual projection;
- Mantis cross-channel pooling to 512 dimensions;
- fixed-split classifier behavior.

## Outputs

Each run creates a timestamped experiment directory below `--result_dir`. The
main artifacts include:

- `args.json`: complete command-line configuration;
- `train_val.csv`: validation and test metrics;
- `splits/Shimmer_11_session11_DRINK_subject_split.csv`: split audit;
- cached `.npz` branch features below `--feature_cache_dir`.

Feature caches are keyed by the model and preprocessing configuration. Delete a
cache only when intentionally forcing feature re-extraction.

## Troubleshooting

- **`Missing required path`**: check that `MODEL_DIR` and `MANTIS_DIR` point to
  downloaded checkpoint directories, not to their parent directory.
- **CUDA out of memory**: select another GPU with `GPU=<index>` or reduce
  `--batch_size` in the direct Python command.
- **Dataset not found**: run from the repository root or explicitly set
  `DATA_DIR=/absolute/path/to/repository/data`.
- **Stale feature dimensions**: remove the relevant ignored feature-cache
  directory after changing checkpoint, layer, pooling, or image settings.
- **Hugging Face download failure**: update `huggingface_hub`, confirm network
  access, and ensure sufficient disk space for the approximately one-billion-
  parameter CLIP model.

## License

See `LICENSE`. Dataset use is additionally subject to the original dataset
terms linked above.
