#!/usr/bin/env python3
"""Anonymous subject-aware loader for the bundled Shimmer DRINK example."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_SEED = 42
SPLIT_RATIOS = (0.6, 0.2, 0.2)
REFERENCE_COLUMNS = {
    "dataset_name",
    "numeric_subject_id",
    "label_id",
    "split",
}


def _canonical_split(split: str) -> str:
    value = {"val": "vali", "validation": "vali"}.get(
        split.strip().lower(), split.strip().lower()
    )
    if value not in {"train", "vali", "test"}:
        raise ValueError(f"split must be train, vali, or test; received {split!r}")
    return value


class SubjectMapDataset(Dataset):
    """Load features after constructing a deterministic subject-level split."""

    dataset_name: ClassVar[str]
    relative_directories: ClassVar[tuple[str, ...]]
    sequence_length: ClassVar[int]
    label_names: ClassVar[dict[int, str]]
    expected_subject_counts: ClassVar[tuple[int, int, int]]
    expected_sample_count: ClassVar[int]
    num_channels_expected: ClassVar[int] = 6

    def __init__(
        self,
        data_root: str | Path = ".",
        split: str = "train",
        normalize: bool = False,
        verbose: bool = False,
        *,
        _load_samples: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = _canonical_split(split)
        self.normalize = bool(normalize)
        self.verbose = bool(verbose)
        self.split_seed = SPLIT_SEED
        self.root_path = self._resolve_dataset_root(self.data_root)
        self.data_path = self.root_path / "Feature"
        self.subject_map_path = self.root_path / "Meta" / "subject_map.csv"

        subject_rows = self._read_subject_map()
        self.subject_label = {
            row["numeric_subject_id"]: row["label_id"] for row in subject_rows
        }
        self.train_ids, self.val_ids, self.test_ids = self._build_subject_level_split(
            subject_rows
        )
        self.split_subject_ids = {
            "train": self.train_ids,
            "vali": self.val_ids,
            "test": self.test_ids,
        }

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.sample_subject_ids: np.ndarray | None = None
        if _load_samples:
            self.X, self.y, self.sample_subject_ids = self._load_split_samples()
        if self.verbose:
            print(
                f"{self.dataset_name} split={self.split}: "
                f"subjects={len(self.selected_subject_ids)}, samples={len(self)}"
            )

    def _resolve_dataset_root(self, data_root: Path) -> Path:
        if (data_root / "Meta" / "subject_map.csv").is_file():
            return data_root
        for relative_directory in self.relative_directories:
            candidate = data_root / relative_directory
            if (
                (candidate / "Meta" / "subject_map.csv").is_file()
                and (candidate / "Feature").is_dir()
            ):
                return candidate
        expected = ", ".join(str(data_root / name) for name in self.relative_directories)
        raise FileNotFoundError(f"Could not locate {self.dataset_name}; expected {expected}")

    def _read_subject_map(self) -> list[dict[str, int]]:
        rows: list[dict[str, int]] = []
        with self.subject_map_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"numeric_subject_id", "label_id"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{self.subject_map_path} is missing {sorted(missing)}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    rows.append(
                        {
                            "numeric_subject_id": int(row["numeric_subject_id"]),
                            "label_id": int(row["label_id"]),
                        }
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid subject or label at {self.subject_map_path}:{line_number}"
                    ) from exc

        subject_ids = [row["numeric_subject_id"] for row in rows]
        if not rows or len(subject_ids) != len(set(subject_ids)):
            raise ValueError("subject_map.csv must contain unique, non-empty subject rows")
        unknown_labels = {row["label_id"] for row in rows} - set(self.label_names)
        if unknown_labels:
            raise ValueError(f"Unknown labels: {sorted(unknown_labels)}")
        return sorted(rows, key=lambda row: row["numeric_subject_id"])

    def _build_subject_level_split(
        self, subject_rows: Sequence[dict[str, int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        split_ids: list[list[int]] = [[], [], []]
        for label_id in sorted({row["label_id"] for row in subject_rows}):
            ids = [
                row["numeric_subject_id"]
                for row in subject_rows
                if row["label_id"] == label_id
            ]
            random.Random(self.split_seed).shuffle(ids)
            train_end = int(np.floor(SPLIT_RATIOS[0] * len(ids)))
            vali_end = train_end + int(np.floor(SPLIT_RATIOS[1] * len(ids)))
            split_ids[0].extend(ids[:train_end])
            split_ids[1].extend(ids[train_end:vali_end])
            split_ids[2].extend(ids[vali_end:])

        arrays = tuple(
            np.asarray(sorted(values), dtype=np.int64) for values in split_ids
        )
        if tuple(len(values) for values in arrays) != self.expected_subject_counts:
            raise ValueError(
                f"Unexpected subject split counts for {self.dataset_name}: "
                f"{tuple(len(values) for values in arrays)}"
            )
        sets = [set(map(int, values)) for values in arrays]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise AssertionError("Subject leakage detected between splits")
        if set.union(*sets) != set(self.subject_label):
            raise AssertionError("Subject split does not cover the subject map")
        return arrays  # type: ignore[return-value]

    @property
    def selected_subject_ids(self) -> np.ndarray:
        return self.split_subject_ids[self.split]

    def _feature_path(self, subject_id: int) -> Path:
        return self.data_path / f"feature_{subject_id:03d}.npy"

    def _read_subject_samples(self, subject_id: int) -> np.ndarray:
        feature_path = self._feature_path(subject_id)
        samples = np.load(feature_path, allow_pickle=False)
        if samples.ndim == 2:
            samples = samples[None, ...]
        expected_tail = (self.sequence_length, self.num_channels_expected)
        if samples.ndim != 3 or tuple(samples.shape[1:]) != expected_tail:
            raise ValueError(f"Unexpected feature shape in {feature_path}: {samples.shape}")
        if samples.dtype != np.float32 or not np.isfinite(samples).all():
            raise ValueError(f"Features must be finite float32 values: {feature_path}")
        return samples

    def _load_split_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        subjects: list[np.ndarray] = []
        for subject_value in self.selected_subject_ids:
            subject_id = int(subject_value)
            samples = self._read_subject_samples(subject_id)
            count = len(samples)
            features.append(samples)
            labels.append(np.full(count, self.subject_label[subject_id], dtype=np.int64))
            subjects.append(np.full(count, subject_id, dtype=np.int64))
        x = np.concatenate(features).astype(np.float32, copy=False)
        y = np.concatenate(labels)
        sample_subject_ids = np.concatenate(subjects)
        order = np.random.default_rng(self.split_seed).permutation(len(y))
        return x[order], y[order], sample_subject_ids[order]

    def __len__(self) -> int:
        if self.y is None:
            raise RuntimeError("Dataset samples were not loaded")
        return int(len(self.y))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset samples were not loaded")
        return torch.from_numpy(self.X[index]), torch.tensor(self.y[index]).long()

    @property
    def num_channels(self) -> int:
        return self.num_channels_expected

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def is_multilabel(self) -> bool:
        return False


class ShimmerDrinkDataset(SubjectMapDataset):
    dataset_name = "Shimmer_11_session11_DRINK"
    relative_directories = ("Shimmer_11_session11_DRINK",)
    sequence_length = 4096
    label_names = {0: "HC", 1: "MildPD", 2: "ModeratePD"}
    expected_subject_counts = (77, 25, 28)
    expected_sample_count = 130


DATASET_CLASSES: tuple[type[SubjectMapDataset], ...] = (ShimmerDrinkDataset,)


def _read_reference_csv(
    reference_csv: Path,
) -> dict[str, dict[int, tuple[int, str]]]:
    result = {ShimmerDrinkDataset.dataset_name: {}}
    with reference_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REFERENCE_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{reference_csv} is missing {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if row["dataset_name"].strip() != ShimmerDrinkDataset.dataset_name:
                raise ValueError(f"Unsupported dataset at {reference_csv}:{line_number}")
            subject_id = int(row["numeric_subject_id"])
            result[ShimmerDrinkDataset.dataset_name][subject_id] = (
                int(row["label_id"]),
                _canonical_split(row["split"]),
            )
    return result


def _verify_reference_assignment(
    dataset: SubjectMapDataset,
    reference: dict[str, dict[int, tuple[int, str]]],
) -> None:
    actual: dict[int, tuple[int, str]] = {}
    for split, subject_ids in dataset.split_subject_ids.items():
        for subject_value in subject_ids:
            subject_id = int(subject_value)
            actual[subject_id] = (dataset.subject_label[subject_id], split)
    expected = reference.get(dataset.dataset_name, {})
    if actual != expected:
        raise ValueError(f"{dataset.dataset_name} does not match the fixed split reference")


def _validate_label_file(dataset: SubjectMapDataset) -> None:
    label_path = dataset.root_path / "Label" / "label.npy"
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    if labels.dtype != np.int64 or labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"Expected int64 [N,2] labels in {label_path}")
    mapping = {int(subject_id): int(label_id) for label_id, subject_id in labels}
    if len(mapping) != len(labels) or mapping != dataset.subject_label:
        raise ValueError("label.npy does not match Meta/subject_map.csv")
