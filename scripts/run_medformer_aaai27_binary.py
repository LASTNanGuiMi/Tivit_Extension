#!/usr/bin/env python3
"""Run one Medformer classification baseline on an AAAI27 binary task.

The Medformer repository has a useful classification experiment loop, but its
data provider does not know about the AAAI27 subject-split datasets. This
adapter registers a small in-process dataset provider and keeps the original
Medformer model/training code unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEDFORMER_ROOT = Path("/home/xuzheyuan/guoyin/Medformer")
DEFAULT_DATA_DIR = Path("/home/xuzheyuan/guoyin/data")
DATASET_NAMES = (
    "PADS_09_task06_DrinkGlas",
    "Shimmer_11_session11_DRINK",
)
MODEL_NAMES = (
    "Autoformer",
    "Transformer",
    "PatchTST",
    "Medformer",
    "Crossformer",
    "FEDformer",
)
EXPECTED_REFERENCE_SHA256 = (
    "c11d4678274b9c66d36d08f4fc08890c99fe8e79879f008d43b19b1faaf535a0"
)
EXPECTED_SPLIT_SAMPLES = {
    "PADS_09_task06_DrinkGlas": (280, 92, 97),
    "Shimmer_11_session11_DRINK": (77, 25, 28),
}

_BUNDLE_CACHE = {}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _resize_samples(samples: np.ndarray, target_length: int) -> np.ndarray:
    """Resize [N, T, C] samples using TiViT's linear interpolation."""
    if samples.shape[1] == target_length:
        return samples.astype(np.float32, copy=False)
    tensor = torch.from_numpy(samples.transpose(0, 2, 1).astype(np.float32))
    resized = F.interpolate(
        tensor, size=target_length, mode="linear", align_corners=True
    )
    return resized.transpose(1, 2).contiguous().numpy()


def _load_bundle(dataset_name: str, data_dir: str, target_length: int):
    key = (dataset_name, str(Path(data_dir).expanduser().resolve()), target_length)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]

    base = Path(data_dir).expanduser()
    candidates = (
        base,
        base / "AAAI_Data",
        base / "med_data" / "AAAI_Data",
        base / "med_data",
    )
    data_root = None
    for candidate in candidates:
        if (
            (candidate / "data_loading" / "datasets.py").is_file()
            and (candidate / "data_loading" / "split_reference_seed42.csv").is_file()
            and (candidate / dataset_name / "Meta" / "subject_map.csv").is_file()
        ):
            data_root = candidate.resolve()
            break
    if data_root is None:
        raise FileNotFoundError(
            f"Could not find AAAI_Data/{dataset_name} below {data_dir}"
        )

    module_path = data_root / "data_loading" / "datasets.py"
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if digest != EXPECTED_REFERENCE_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {module_path}: {digest}; "
            f"expected {EXPECTED_REFERENCE_SHA256}"
        )
    module_name = f"_aaai27_reference_{digest[:12]}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    dataset_class = next(
        cls for cls in module.DATASET_CLASSES if cls.dataset_name == dataset_name
    )
    datasets = {
        split: dataset_class(
            data_root=data_root, split=split, normalize=True, verbose=False
        )
        for split in ("train", "vali", "test")
    }
    reference = module._read_reference_csv(
        data_root / "data_loading" / "split_reference_seed42.csv"
    )
    module._verify_reference_assignment(datasets["train"], reference)
    module._validate_label_file(datasets["train"])
    actual_counts = tuple(len(datasets[split]) for split in ("train", "vali", "test"))
    if actual_counts != EXPECTED_SPLIT_SAMPLES[dataset_name]:
        raise AssertionError(
            f"{dataset_name}: expected split samples "
            f"{EXPECTED_SPLIT_SAMPLES[dataset_name]}, got {actual_counts}"
        )

    bundle = SimpleNamespace(
        dataset_name=dataset_name,
        data_root=data_root,
        train_dataset=datasets["train"],
        vali_dataset=datasets["vali"],
        test_dataset=datasets["test"],
        label_mode="zero_vs_rest",
    )
    split_data = {}
    for split, source_dataset, labels in (
        ("TRAIN", bundle.train_dataset, bundle.train_dataset.y),
        ("VAL", bundle.vali_dataset, bundle.vali_dataset.y),
        ("TEST", bundle.test_dataset, bundle.test_dataset.y),
    ):
        split_data[split] = (
            _resize_samples(source_dataset.X, target_length),
            (np.asarray(labels) != 0).astype(np.int64),
        )
    _BUNDLE_CACHE[key] = (bundle, split_data)
    return bundle, split_data


def _write_split_audit(bundle, result_dir: Path) -> Path:
    split_dir = result_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    output = split_dir / f"{bundle.dataset_name}_subject_split.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "numeric_subject_id",
                "original_label_id",
                "label_id",
                "split",
            ]
        )
        rows = []
        for split, dataset in (
            ("train", bundle.train_dataset),
            ("vali", bundle.vali_dataset),
            ("test", bundle.test_dataset),
        ):
            for subject_id in dataset.selected_subject_ids:
                original_label = int(dataset.subject_label[int(subject_id)])
                rows.append(
                    (
                        bundle.dataset_name,
                        int(subject_id),
                        original_label,
                        int(original_label != 0),
                        split,
                    )
                )
        writer.writerows(sorted(rows, key=lambda row: row[1]))
    return output


class AAAI27Loader(Dataset):
    """Medformer Dataset interface backed by TiViT's AAAI27 reference loader."""

    def __init__(self, args, root_path=None, flag=None):
        del root_path
        dataset_name = args.aaai27_dataset
        data_dir = args.aaai27_data_dir
        target_length = int(args.aaai27_target_seq_len)
        if dataset_name not in DATASET_NAMES:
            raise ValueError(f"Unsupported AAAI27 dataset: {dataset_name}")
        if flag is None:
            raise ValueError("AAAI27Loader requires TRAIN, VAL, or TEST flag")

        bundle, split_data = _load_bundle(dataset_name, data_dir, target_length)
        split = flag.upper()
        if split not in split_data:
            raise ValueError(f"Unsupported split: {flag}")
        self.X, self.y = split_data[split]
        self.max_seq_len = int(self.X.shape[1])
        self.class_names = ["zero", "nonzero"]
        self.dataset_name = dataset_name
        self.original_sequence_length = int(bundle.train_dataset.sequence_length)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.tensor(
            int(self.y[index]), dtype=torch.long
        )

    def __len__(self):
        return int(self.y.shape[0])


class DeviceAwareCrossEntropy(nn.Module):
    """Cross entropy whose class weights follow logits across CPU/GPU eval."""

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, logits, labels):
        return F.cross_entropy(
            logits, labels, weight=self.weights.to(device=logits.device)
        )


def _build_args(parsed: argparse.Namespace):
    return SimpleNamespace(
        task_name="classification",
        is_training=1,
        model_id=f"AAAI27-{parsed.dataset}",
        model=parsed.model,
        data="AAAI27",
        root_path=str(parsed.data_dir),
        data_path="",
        features="M",
        target="OT",
        freq="h",
        seq_len=parsed.target_seq_len,
        label_len=48,
        pred_len=0,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=6,
        dec_in=6,
        c_out=6,
        d_model=128,
        n_heads=8,
        e_layers=6,
        d_layers=1,
        d_ff=256,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        no_inter_attn=False,
        chunk_size=16,
        patch_len=16,
        stride=8,
        sampling_rate=256,
        patch_len_list="2,4,8",
        single_channel=False,
        augmentations="none",
        num_workers=0,
        itr=1,
        train_epochs=parsed.train_epochs,
        batch_size=parsed.batch_size,
        patience=parsed.patience,
        learning_rate=1e-4,
        des="Exp",
        loss="CE",
        lradj="type1",
        use_amp=False,
        swa=True,
        use_gpu=bool(torch.cuda.is_available() and not parsed.cpu),
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        device_ids=[0],
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        seed=parsed.seed,
        aaai27_dataset=parsed.dataset,
        aaai27_data_dir=str(parsed.data_dir),
        aaai27_target_seq_len=parsed.target_seq_len,
        aaai27_label_mode="zero_vs_rest",
        class_weight=parsed.class_weight,
    )


def _metrics_row(
    dataset,
    model,
    seed,
    target_length,
    val_loss,
    val_metrics,
    test_loss,
    test_metrics,
):
    row = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "target_seq_len": target_length,
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),
    }
    for name, value in val_metrics.items():
        row[f"val_{name}"] = float(value)
    for name, value in test_metrics.items():
        row[f"test_{name}"] = float(value)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-seq-len", type=int, default=512)
    parser.add_argument("--train-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--class-weight", choices=("balanced", "none"), default="balanced"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--medformer-root", type=Path, default=DEFAULT_MEDFORMER_ROOT)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parsed = parser.parse_args()

    if parsed.seed != 42:
        raise SystemExit("This comparison runner is fixed to seed=42")
    if parsed.target_seq_len < 16:
        raise SystemExit("target sequence length must be at least 16")

    # Put Medformer first so its models and layers resolve without modifying it.
    sys.path.insert(0, str(parsed.medformer_root))
    sys.path.insert(1, str(PROJECT_ROOT))
    from data_provider import data_factory
    from exp.exp_classification import Exp_Classification

    data_factory.data_dict["AAAI27"] = AAAI27Loader

    class LocalExp(Exp_Classification):
        def _acquire_device(self):
            # The queue constrains CUDA_VISIBLE_DEVICES to one physical card.
            # Exp_Basic otherwise rewrites the mask and makes cuda:0 ambiguous.
            if self.args.use_gpu and torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")

        def _select_criterion(self):
            if self.args.class_weight == "none":
                return nn.CrossEntropyLoss()
            _, split_data = _load_bundle(
                self.args.aaai27_dataset,
                self.args.aaai27_data_dir,
                self.args.aaai27_target_seq_len,
            )
            counts = np.bincount(split_data["TRAIN"][1], minlength=2)
            weights = counts.sum() / (len(counts) * counts.astype(np.float64))
            return DeviceAwareCrossEntropy(weights)

    run_dir = parsed.result_root / parsed.dataset / parsed.model
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    _seed_everything(parsed.seed)
    args = _build_args(parsed)

    bundle, _ = _load_bundle(
        parsed.dataset, str(parsed.data_dir), parsed.target_seq_len
    )
    args.original_seq_len = int(bundle.train_dataset.sequence_length)
    split_audit = _write_split_audit(bundle, run_dir)
    print(f"Subject split audit: {split_audit}")
    with (run_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    print(
        f"Medformer AAAI27 | dataset={parsed.dataset} model={parsed.model} "
        f"seed={parsed.seed} seq_len={parsed.target_seq_len} "
        f"batch_size={parsed.batch_size} class_weight={parsed.class_weight} "
        f"device={'cuda:0' if args.use_gpu else 'cpu'}"
    )
    exp = LocalExp(args)
    setting = (
        f"classification_AAAI27_{parsed.dataset}_{parsed.model}"
        f"_sl{parsed.target_seq_len}_dm128_el6_df256_seed{parsed.seed}"
    )
    exp.train(setting)

    criterion = exp._select_criterion()
    val_data, val_loader = exp._get_data(flag="VAL")
    test_data, test_loader = exp._get_data(flag="TEST")
    val_loss, val_metrics = exp.vali(val_data, val_loader, criterion)
    test_loss, test_metrics = exp.vali(test_data, test_loader, criterion)
    row = _metrics_row(
        parsed.dataset,
        parsed.model,
        parsed.seed,
        parsed.target_seq_len,
        val_loss,
        val_metrics,
        test_loss,
        test_metrics,
    )
    row["class_weight"] = parsed.class_weight
    row["label_mapping"] = "0_vs_1_or_2"
    row["original_seq_len"] = args.original_seq_len
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True, allow_nan=True)
    fieldnames = list(row.keys())
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print("Test metrics:", json.dumps(test_metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
