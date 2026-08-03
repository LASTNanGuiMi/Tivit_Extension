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

from src.datautils import get_aaai27_dataloaders, write_aaai27_split_audit


SELECTED_DATASETS = {
    "Shimmer_10_session10_AFC": {
        "time": 4096,
        "subjects": (69, 23, 25),
        "label_mode": "shimmer_hc_vs_pd",
        "mapping": {0: 0, 1: 1, 2: 1},
    },
    "PADS_11_task08_TouchIndex": {
        "time": 976,
        "subjects": (280, 92, 97),
        "label_mode": "pads_pd_vs_hc",
        "mapping": {0: 0, 1: 1},
    },
}


def validate_dataset(dataset_name, data_dir, batch_size):
    expected = SELECTED_DATASETS[dataset_name]
    args = SimpleNamespace(
        data_dir=data_dir,
        batch_size=batch_size,
        aaai27_label_mode=expected["label_mode"],
    )
    bundle = get_aaai27_dataloaders(dataset_name, args)
    split_items = (
        ("train", bundle.train_dataset, bundle.train_loader, bundle.train_labels),
        ("vali", bundle.vali_dataset, bundle.vali_loader, bundle.vali_labels),
        ("test", bundle.test_dataset, bundle.test_loader, bundle.test_labels),
    )

    subject_counts = tuple(
        len(source_dataset.selected_subject_ids)
        for _, source_dataset, _, _ in split_items
    )
    if subject_counts != expected["subjects"]:
        raise AssertionError(
            f"{dataset_name}: subjects={subject_counts}, "
            f"expected={expected['subjects']}"
        )

    expected_sample_counts = []
    for split, source_dataset, loader, labels in split_items:
        original_labels = np.asarray(source_dataset.y, dtype=np.int64)
        keep = np.isin(
            original_labels,
            np.asarray(list(expected["mapping"]), dtype=np.int64),
        )
        expected_labels = np.asarray(
            [
                expected["mapping"][int(label)]
                for label in original_labels[keep]
            ],
            dtype=np.int64,
        )
        expected_sample_counts.append(len(expected_labels))
        np.testing.assert_array_equal(labels, expected_labels)
        if set(np.unique(labels)) != {0, 1}:
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

    sample_counts = tuple(len(labels) for _, _, _, labels in split_items)
    if sample_counts != tuple(expected_sample_counts):
        raise AssertionError(
            f"{dataset_name}: samples={sample_counts}, "
            f"expected={tuple(expected_sample_counts)}"
        )

    train_tensor = bundle.train_loader.dataset.tensors[0]
    channel_means = train_tensor.mean(dim=(0, 2))
    channel_stds = train_tensor.std(dim=(0, 2), unbiased=False)
    if not torch.allclose(channel_means, torch.zeros_like(channel_means), atol=1e-5):
        raise AssertionError(f"{dataset_name}: training channel means are not zero")
    if not torch.allclose(channel_stds, torch.ones_like(channel_stds), atol=1e-4):
        raise AssertionError(f"{dataset_name}: training channel stds are not one")

    with tempfile.TemporaryDirectory(prefix="neurosigvit_audit_") as temp_dir:
        audit_path = write_aaai27_split_audit(bundle, temp_dir)
        with Path(audit_path).open(encoding="utf-8", newline="") as handle:
            audit_rows = list(csv.DictReader(handle))
        if len(audit_rows) != sum(expected_sample_counts):
            raise AssertionError(
                f"{dataset_name}: audit rows={len(audit_rows)}, "
                f"expected={sum(expected_sample_counts)}"
            )
        for row in audit_rows:
            original_label = int(row["original_label_id"])
            expected_label = expected["mapping"][original_label]
            if int(row["label_id"]) != expected_label:
                raise AssertionError(
                    f"{dataset_name}: invalid audit mapping in row {row}"
                )

    print(
        f"PASS {dataset_name}: source_subjects={subject_counts}, "
        f"selected_samples={sample_counts}, batch=(B,6,{expected['time']}), "
        f"label_mode={expected['label_mode']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate the selected NeuroSigViT clinical datasets."
    )
    parser.add_argument("--data-dir", default="data/Neuro")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        choices=tuple(SELECTED_DATASETS),
        default=list(SELECTED_DATASETS),
    )
    args = parser.parse_args()

    for dataset_name in args.dataset_names:
        validate_dataset(dataset_name, args.data_dir, args.batch_size)
        gc.collect()

    print(f"VALIDATION PASSED: {len(args.dataset_names)}/{len(args.dataset_names)}")


if __name__ == "__main__":
    main()
