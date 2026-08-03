# Selected dataset processing protocols

All four selected datasets are exposed below the project `data/` directory and
launched through `scripts/run_selected_dataset.sh`. Historical protocols remain
available through their explicit command-line switches, but the wrappers use the
leakage-safe protocols listed here.

| Key | Dataset | Input tensor | Split | Labels | Normalization |
|---|---|---|---|---|---|
| `falltl` | FallTL | one variable-length CSV per sample, 9 channels | stratified file-level 60/20/20, seed 42 | filename prefix `D=0`, `F=1` | interpolate non-finite values; channel mean/std from training sequences only; zero-pad each split |
| `ucihar` | UCI HAR | `(N,6,128)`: total acceleration XYZ + body gyroscope XYZ | official test subjects; official training subjects split into train/validation at seed 42 | official six activities, converted from 1-6 to 0-5 | use the official preprocessed inertial signals; no additional scaling |
| `shimmer10` | `Shimmer_10_session10_AFC` | `(N,6,4096)` | label-stratified subject-level 60/20/20, seed 42 | `HC=0`, `MildPD/ModeratePD=1` | channel mean/std from the training split only |
| `pads11` | `PADS_11_task08_TouchIndex` | `(N,6,976)` | label-stratified subject-level 60/20/20, seed 42 | retain Healthy and Parkinson only: `Healthy=0`, `Parkinson=1`; remove OMD | channel mean/std from the training split only |

The UCI HAR validation split is subject-disjoint from training and from the
official test partition. FallTL splitting happens after treating each CSV as one
sample, so windows from one recording cannot cross splits. Shimmer and PADS use
the subject IDs in `Meta/subject_map.csv`; the split audit written with each run
records the original and mapped labels.

`comparison_binary` is the selected, leakage-safe FallTL protocol; the historical
compatibility path is named `legacy_windows`. The fixed data-split seed is 42 in
all four loaders. The wrapper's `SEED` variable maps to `--random_seed` (default
2022) for model initialization, training randomness, and result naming; it does
not replace the fixed split seed.

Run examples:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neurosigvit
cd NeuroSigViT-main
DRY_RUN=1 bash scripts/run_selected_dataset.sh falltl
DRY_RUN=1 bash scripts/run_selected_dataset.sh ucihar
DRY_RUN=1 bash scripts/run_selected_dataset.sh shimmer10
DRY_RUN=1 bash scripts/run_selected_dataset.sh pads11
```

Remove `DRY_RUN=1` only after checking GPU availability. `GPU`, `SEED`,
`EPOCHS`, `PATIENCE`, `RESULT_DIR`, and `FEATURE_CACHE_DIR` can be overridden as
environment variables. Additional `main.py` arguments can be appended after the
dataset key.

The commands assume the `neurosigvit` Conda environment described in
`README.md`. If Miniconda is installed elsewhere, adjust only the first
`source` command locally; do not put that machine-specific path into a
submitted script or document.

Before creating an anonymous artifact, run `python scripts/check_anonymity.py`
in the activated environment and follow `ANONYMITY.md`. Local data links,
results, caches, checkpoints, and Git metadata are runtime-only and must not be
included in the submission archive.
