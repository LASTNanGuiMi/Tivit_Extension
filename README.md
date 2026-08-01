# NeuroSigViT

This repository contains the implementation of **NeuroSigViT** for wearable
sensor-based Parkinson's disease assessment. The maintained example uses the
Shimmer DRINK subset and combines frozen CLIP ViT-H/14 and Mantis-8M features
with a lightweight trainable fusion head.

The accompanying paper is *Representations for Wearable Sensor-Based
Parkinson's Disease Assessment Lie Hidden in Pretrained Vision Transformers*.

## Artifact scope

The repository is intentionally limited to one complete, runnable example.

| Item | Setting |
| --- | --- |
| Dataset | `Shimmer_11_session11_DRINK` |
| Task | Healthy control vs. Parkinson's disease |
| Input | Six right-wrist inertial channels, 4096 time steps |
| Split | Fixed subject-disjoint train/validation/test split |
| Split size | 77 / 25 / 28 subjects and samples |
| Default seed | 2022 |
| Frozen encoders | CLIP ViT-H/14 and Mantis-8M |
| Trainable modules | Branch projections, two-token attention, and MLP classifier |
| Entry point | `bash scripts/run_shimmer_example.sh` |

The processed Shimmer arrays and split reference are included. The two model
checkpoints must be downloaded separately. Commands for FallTL, PADS, and
UCI-HAR are not part of this release.

## Method

![NeuroSigViT architecture](assets/neurosigvit_method.jpg)

The visual branch arranges the six sensor channels as an activity graph and
extracts the layer-14 representation from frozen CLIP ViT-H/14. Patch tokens
are averaged without the CLS token and projected to a 1024-dimensional visual
feature. The temporal branch applies frozen Mantis-8M to each channel and
averages the channel embeddings to obtain one 512-dimensional temporal
feature. Both features are projected to 128 dimensions, updated by two-token
self-attention, and classified by a two-layer MLP.

Only the projection, attention, and classifier layers are optimized. The code
uses 1024 visual dimensions, 512 temporal dimensions, and patch-only visual
pooling. These implementation dimensions supersede the legacy dimension labels
in the method artwork.

## 1. Environment

The experiment was checked with the following configuration:

| Component | Version |
| --- | --- |
| Operating system | Ubuntu 22.04 |
| Python | 3.11.15 |
| GPU | NVIDIA GeForce RTX 4090 |
| NVIDIA driver | 535.309.01 |
| PyTorch | 2.7.1, CUDA 12.6 build |
| torchvision | 0.22.1 |
| open_clip_torch | 2.32.0 |
| mantis-tsfm | 1.0.0 |
| transformers | 4.33.3 |

Create a clean Python environment from the pinned dependency file:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The equivalent Conda setup is:

```bash
conda create -n neurosigvit python=3.11 -y
conda activate neurosigvit
python -m pip install -r requirements.txt
```

Confirm that PyTorch can access the GPU:

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Checkpoints

| Branch | Model | Source |
| --- | --- | --- |
| Visual | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | [Hugging Face](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) |
| Temporal | `paris-noah/Mantis-8M` | [Hugging Face](https://huggingface.co/paris-noah/Mantis-8M) |

Download both checkpoints to the default locations used by the launcher:

```bash
mkdir -p checkpoints

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_dir='checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K')"

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='paris-noah/Mantis-8M', revision='93a16a52a5e2e6d76c0b823533b5836dd83ca10a', local_dir='checkpoints/Mantis-8M')"
```

The Mantis command fixes the revision used in the checked server environment.
After downloading, the expected layout is:

```text
checkpoints/
|-- CLIP-ViT-H-14-laion2B-s32B-b79K/
`-- Mantis-8M/
```

## 3. Data

The bundled example is stored at:

```text
data/med_data/AAAI_Data/Shimmer_11_session11_DRINK/
|-- Feature/              # 130 float32 arrays
|-- Label/label.npy
`-- Meta/                 # channel information and anonymous subject map
```

Labels are defined as:

- `0`: healthy control (`HC`)
- `1`: Parkinson's disease (`MildPD` and `ModeratePD`)

The fixed split contains 77 training, 25 validation, and 28 test subjects. Each
subject contributes one processed sample in this example. Channel
standardization is fitted on the training split and reused for validation and
test data.

Original dataset sources:

| Dataset | Download or project page |
| --- | --- |
| Shimmer / PDWearML | [IEEE DataPort](https://ieee-dataport.org/documents/pdwearml-leveraging-daily-activities-rapid-free-living-parkinsons-disease-severity), [PDWearML repository](https://github.com/wang-xulong/PDWearML) |
| FallTL | [Zenodo record 17552449](https://zenodo.org/records/17552449) |
| PADS | [PhysioNet PADS v1.0.0](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) |
| UCI-HAR | [Google Drive](https://drive.google.com/file/d/13HA6l3dnOm46dN4EgzS_YRwHUGIEruKD/view?usp=drive_link), [UCI repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |

The original licenses and access terms apply to all datasets.

## 4. One-command run

Run the maintained experiment from the repository root:

```bash
bash scripts/run_shimmer_example.sh
```

The default configuration uses seed 2022, batch size 16, at most 40 epochs,
and early-stopping patience 8. Results and feature caches are written below
`results/` and `feature_cache/`.

Use environment variables when checkpoints, data, or Python are stored in
different locations:

```bash
MODEL_DIR=/path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
MANTIS_DIR=/path/to/Mantis-8M \
DATA_DIR=/path/to/data \
PYTHON_BIN=/path/to/python \
GPU=0 \
bash scripts/run_shimmer_example.sh
```

The launcher also accepts `SEED`, `EPOCHS`, `PATIENCE`, `RESULT_DIR`, and
`FEATURE_CACHE_DIR`.

Check path resolution and print the command without starting training:

```bash
DRY_RUN=1 bash scripts/run_shimmer_example.sh
```

For a one-epoch smoke run:

```bash
EPOCHS=1 PATIENCE=1 \
RESULT_DIR=/tmp/neurosigvit_smoke_result \
FEATURE_CACHE_DIR=/tmp/neurosigvit_smoke_cache \
bash scripts/run_shimmer_example.sh
```

## 5. Direct Python command

The shell launcher expands to the following command. Set `MODEL_DIR` and
`MANTIS_DIR` before using it directly.

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

## 6. Outputs

A run creates a timestamped directory under the selected result path:

```text
results/shimmer_example_seed2022/
`-- <timestamp>/
    |-- args.json
    |-- train_val.csv
    `-- splits/
        `-- Shimmer_11_session11_DRINK_subject_split.csv
```

`args.json` records the complete configuration, `train_val.csv` contains the
reported metrics, and the split CSV records the subject assignment. Frozen
branch features are stored as `.npz` files below the selected feature-cache
directory. Generated results and caches are excluded from Git.

## 7. Verification

Run the following checks before a full experiment:

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
- `(B, 6, 4096)` input shape;
- training-only normalization statistics;
- feature-cache signatures and reuse;
- CLIP patch-only pooling and 1024-dimensional projection;
- Mantis channel pooling to 512 dimensions;
- fixed-split classifier behavior.

## Repository layout

```text
.
|-- main.py
|-- src/                         # model and training code
|-- assets/
|   `-- neurosigvit_method.jpg
|-- data/                        # bundled Shimmer example
|-- data_loading/                # loader and fixed split reference
|-- scripts/
|   |-- run_shimmer_example.sh
|   `-- test_*.py
|-- requirements.txt
`-- LICENSE
```

## License

Code is released under the terms in [LICENSE](LICENSE). Dataset use remains
subject to the terms of the original sources listed above.
