#!/usr/bin/env python3
import argparse
import csv
import gc
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datautils import (
    AAAI27_DATASET_NAMES,
    get_aaai27_dataloaders,
    write_aaai27_split_audit,
)


EXPECTED = {
    "mPowerRest": {
        "time": 2688,
        "subjects": (624, 207, 211),
        "samples": (16124, 5399, 5368),
    },
    "mPowerReturn": {
        "time": 960,
        "subjects": (552, 183, 187),
        "samples": (9132, 2619, 2733),
    },
    "mPowerOutbound": {
        "time": 1088,
        "subjects": (622, 207, 210),
        "samples": (15289, 5278, 5193),
    },
    "PADS_09_task06_DrinkGlas": {
        "time": 976,
        "subjects": (280, 92, 97),
        "samples": (280, 92, 97),
    },
    "PADS_10_task07_CrossArms": {
        "time": 976,
        "subjects": (280, 92, 97),
        "samples": (280, 92, 97),
    },
    "Shimmer_11_session11_DRINK": {
        "time": 4096,
        "subjects": (77, 25, 28),
        "samples": (77, 25, 28),
    },
    "Shimmer_12_session12_PICK": {
        "time": 4096,
        "subjects": (65, 21, 25),
        "samples": (65, 21, 25),
    },
}


def validate_dataset(dataset_name, data_dir, batch_size, label_mode):
    args = SimpleNamespace(
        data_dir=data_dir,
        batch_size=batch_size,
        aaai27_label_mode=label_mode,
    )
    bundle = get_aaai27_dataloaders(dataset_name, args)
    expected = EXPECTED[dataset_name]
    split_items = (
        ("train", bundle.train_dataset, bundle.train_loader, bundle.train_labels),
        ("vali", bundle.vali_dataset, bundle.vali_loader, bundle.vali_labels),
        ("test", bundle.test_dataset, bundle.test_loader, bundle.test_labels),
    )

    subject_counts = tuple(
        len(source_dataset.selected_subject_ids)
        for _, source_dataset, _, _ in split_items
    )
    sample_counts = tuple(len(labels) for _, _, _, labels in split_items)
    if subject_counts != expected["subjects"]:
        raise AssertionError(
            f"{dataset_name}: subjects={subject_counts}, expected={expected['subjects']}"
        )
    if sample_counts != expected["samples"]:
        raise AssertionError(
            f"{dataset_name}: samples={sample_counts}, expected={expected['samples']}"
        )

    for split, source_dataset, loader, labels in split_items:
        original_labels = np.asarray(source_dataset.y, dtype=np.int64)
        expected_labels = (
            (original_labels != 0).astype(np.int64)
            if label_mode == "shimmer_hc_vs_pd"
            else original_labels
        )
        np.testing.assert_array_equal(labels, expected_labels)
        if label_mode == "shimmer_hc_vs_pd" and set(np.unique(labels)) != {0, 1}:
            raise AssertionError(
                f"{dataset_name} {split}: expected binary labels, got "
                f"{np.unique(labels).tolist()}"
            )

        expected_shape = (len(labels), 6, expected["time"])
        tensor = loader.dataset.tensors[0]
        if tuple(tensor.shape) != expected_shape:
            raise AssertionError(
                f"{dataset_name} {split}: shape={tuple(tensor.shape)}, "
                f"expected={expected_shape}"
            )
        if tensor.dtype != torch.float32 or labels.dtype != np.int64:
            raise AssertionError(
                f"{dataset_name} {split}: dtype={tensor.dtype}/{labels.dtype}"
            )

        (batch,) = next(iter(loader))
        expected_batch = min(batch_size, len(labels))
        if tuple(batch.shape) != (expected_batch, 6, expected["time"]):
            raise AssertionError(
                f"{dataset_name} {split}: batch shape={tuple(batch.shape)}"
            )
        if not torch.isfinite(batch).all():
            raise AssertionError(f"{dataset_name} {split}: non-finite batch")

    train_tensor = bundle.train_loader.dataset.tensors[0]
    channel_means = train_tensor.mean(dim=(0, 2))
    channel_stds = train_tensor.std(dim=(0, 2), unbiased=False)
    if not torch.allclose(channel_means, torch.zeros_like(channel_means), atol=1e-5):
        raise AssertionError(f"{dataset_name}: training channel means are not zero")
    if not torch.allclose(channel_stds, torch.ones_like(channel_stds), atol=1e-4):
        raise AssertionError(f"{dataset_name}: training channel stds are not one")

    with tempfile.TemporaryDirectory(prefix="aaai27_audit_") as temp_dir:
        audit_path = write_aaai27_split_audit(bundle, temp_dir)
        with Path(audit_path).open(encoding="utf-8", newline="") as handle:
            audit_rows = list(csv.DictReader(handle))
        if len(audit_rows) != sum(expected["subjects"]):
            raise AssertionError(
                f"{dataset_name}: audit rows={len(audit_rows)}, "
                f"expected={sum(expected['subjects'])}"
            )
        for row in audit_rows:
            original_label = int(row["original_label_id"])
            expected_label = (
                int(original_label != 0)
                if label_mode == "shimmer_hc_vs_pd"
                else original_label
            )
            if int(row["label_id"]) != expected_label:
                raise AssertionError(
                    f"{dataset_name}: invalid audit mapping in row {row}"
                )

    print(
        f"PASS {dataset_name}: subjects={subject_counts}, samples={sample_counts}, "
        f"batch=(B,6,{expected['time']}), label_mode={label_mode}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate the fixed subject-level NeuroSigViT Shimmer adapter."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--label-mode",
        choices=["original", "shimmer_hc_vs_pd"],
        default="shimmer_hc_vs_pd",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        choices=AAAI27_DATASET_NAMES,
        default=["Shimmer_11_session11_DRINK"],
    )
    args = parser.parse_args()

    for dataset_name in args.dataset_names:
        validate_dataset(
            dataset_name,
            args.data_dir,
            args.batch_size,
            args.label_mode,
        )
        gc.collect()

    print(f"VALIDATION PASSED: {len(args.dataset_names)}/{len(args.dataset_names)} datasets")


if __name__ == "__main__":
    main()
