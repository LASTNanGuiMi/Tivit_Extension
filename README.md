# NeuroSigViT

Code for **NeuroSigViT**, the model described in *Representations for Wearable
Sensor-Based Parkinson's Disease Assessment Lie Hidden in Pretrained Vision
Transformers*.

The repository includes a complete Shimmer example
(`Shimmer_11_session11_DRINK`), its fixed subject split, and the processed input
arrays. The CLIP ViT-H/14 and Mantis-8M weights are downloaded separately.

## Method

![NeuroSigViT architecture](assets/neurosigvit_method.jpg)

NeuroSigViT has two frozen feature branches. The visual branch converts the six
sensor channels into an activity graph and extracts patch-token features from
layer 14 of CLIP ViT-H/14. The temporal branch applies Mantis-8M to each sensor
channel and averages the six channel embeddings. The resulting 1024-dimensional
visual feature and 512-dimensional temporal feature are projected to 128
dimensions, updated by two-token self-attention, and passed to an MLP
classifier. Only the projection, attention, and classifier layers are trained.

The implementation excludes the CLIP CLS token before patch-token averaging.
It also pools Mantis features across channels, so the temporal branch produces
one 512-dimensional vector per sample. These dimensions are the ones used by
the code, even though an earlier version of the figure contains legacy
dimension labels.

## Repository layout

```text
.
├── main.py
├── src/                              # model, feature extraction, and training
├── assets/
│   └── neurosigvit_method.jpg
├── data/
│   └── med_data/AAAI_Data/
│       └── Shimmer_11_session11_DRINK/
│           ├── Feature/              # 130 processed samples
│           ├── Label/label.npy
│           └── Meta/                 # channel and anonymous subject metadata
├── data_loading/
│   ├── datasets.py
│   └── split_reference_seed42.csv
├── scripts/
│   ├── run_shimmer_example.sh
│   └── test_*.py
└── requirements.txt
```

## Environment

The supplied `requirements.txt` pins the direct Python dependencies used for
the checked run. The following configuration was tested on the server:

| Component | Version |
| --- | --- |
| Ubuntu | 22.04 |
| Python | 3.11.15 |
| GPU | NVIDIA GeForce RTX 4090 |
| NVIDIA driver | 535.309.01 |
| PyTorch | 2.7.1 (CUDA 12.6 build) |
| torchvision | 0.22.1 |
| open_clip_torch | 2.32.0 |
| mantis-tsfm | 1.0.0 |
| transformers | 4.33.3 |

Create a fresh environment before installing the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda can be used instead:

```bash
conda create -n neurosigvit python=3.11 -y
conda activate neurosigvit
python -m pip install -r requirements.txt
```

Check the CUDA installation with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Checkpoints

NeuroSigViT uses the following public checkpoints:

| Branch | Checkpoint | Download |
| --- | --- | --- |
| Visual | LAION OpenCLIP ViT-H/14, `s32B-b79K` | [Hugging Face](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) |
| Temporal | Mantis-8M | [Hugging Face](https://huggingface.co/paris-noah/Mantis-8M) |

Download them to the default paths used by the example script:

```bash
mkdir -p checkpoints

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_dir='checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K')"

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='paris-noah/Mantis-8M', revision='93a16a52a5e2e6d76c0b823533b5836dd83ca10a', local_dir='checkpoints/Mantis-8M')"
```

The second command fixes the Mantis revision used for the server check.

## Data

The maintained example uses a processed Shimmer DRINK subset included at:

```text
data/med_data/AAAI_Data/Shimmer_11_session11_DRINK/
```

Each sample has six right-wrist inertial channels and 4096 time steps. The task
is binary PD status classification:

- `0`: healthy control (`HC`)
- `1`: Parkinson's disease (`MildPD` and `ModeratePD`)

The fixed subject-disjoint split contains 77 training, 25 validation, and 28
test samples. Standardization statistics are fitted on the training split and
then applied to validation and test data.

Original dataset pages:

| Dataset | Link |
| --- | --- |
| Shimmer / PDWearML | [IEEE DataPort](https://ieee-dataport.org/documents/pdwearml-leveraging-daily-activities-rapid-free-living-parkinsons-disease-severity), [PDWearML code](https://github.com/wang-xulong/PDWearML) |
| FallTL | [Zenodo record 17552449](https://zenodo.org/records/17552449) |
| PADS | [PhysioNet PADS v1.0.0](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) |
| UCI-HAR | [Google Drive](https://drive.google.com/file/d/13HA6l3dnOm46dN4EgzS_YRwHUGIEruKD/view?usp=drive_link), [UCI repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |

Only the bundled Shimmer example is covered by the commands below. Use of the
source datasets remains subject to their original licenses and terms.

## Run the Shimmer example

After installing the environment and downloading both checkpoints, run:

```bash
bash scripts/run_shimmer_example.sh
```

This is the one-command experiment entry point. It uses the bundled data,
trains for up to 40 epochs with early-stopping patience 8, and writes outputs to
`results/shimmer_example_seed2022/`.

The script accepts environment-variable overrides when the data, checkpoints,
or Python executable are stored elsewhere:

```bash
MODEL_DIR=/path/to/CLIP-ViT-H-14-laion2B-s32B-b79K \
MANTIS_DIR=/path/to/Mantis-8M \
DATA_DIR=/path/to/data \
PYTHON_BIN=/path/to/python \
GPU=0 \
bash scripts/run_shimmer_example.sh
```

To print the command and verify all paths without starting training:

```bash
DRY_RUN=1 bash scripts/run_shimmer_example.sh
```

A short smoke run can be launched with:

```bash
EPOCHS=1 PATIENCE=1 \
RESULT_DIR=/tmp/neurosigvit_smoke_result \
FEATURE_CACHE_DIR=/tmp/neurosigvit_smoke_cache \
bash scripts/run_shimmer_example.sh
```

## Direct Python command

The shell script wraps the following command. Set `MODEL_DIR` and `MANTIS_DIR`
before running it directly.

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

## Checks

The lightweight checks do not require a full training run:

```bash
python -m compileall -q main.py src data_loading scripts
bash -n scripts/run_shimmer_example.sh
python scripts/test_feature_cache.py
python scripts/test_mlp_fixed_split.py
python scripts/test_paper_protocol.py
python scripts/test_aaai27_datasets.py --data-dir data
```

They cover the fixed split, label mapping, `(B, 6, 4096)` input shape,
training-only normalization, feature-cache consistency, CLIP patch pooling,
visual projection size, and Mantis channel pooling.

## Outputs

Each run records the command-line configuration, split information, metrics,
and cached branch features. The main files are:

```text
results/shimmer_example_seed2022/
├── <timestamp>/args.json
├── <timestamp>/train_val.csv
└── <timestamp>/splits/Shimmer_11_session11_DRINK_subject_split.csv

feature_cache/shimmer_example/
└── ... .npz
```

`results/` and `feature_cache/` are excluded from Git.

## License

See [LICENSE](LICENSE). Dataset use is also subject to the terms on the source
pages listed above.
