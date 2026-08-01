import csv
import hashlib
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aeon.datasets import load_classification
from torch.utils.data import DataLoader, TensorDataset


FALLTL_FEATURE_COLUMNS = [
    "AccX",
    "AccY",
    "AccZ",
    "GyrX",
    "GyrY",
    "GyrZ",
    "EulerX",
    "EulerY",
    "EulerZ",
]

FENG_PREFERRED_SENSORS = [
    "LowerBack",
    "RightThigh",
    "LeftThigh",
]

FENG_PREFERRED_SIGNALS = [
    "Acc",
    "Gyr",
]


UCI_HAR_SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

UCI_HAR_ACC_GYRO_SIGNAL_FILES = [
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

UCI_HAR_ACC_GYRO_INDICES = [6, 7, 8, 3, 4, 5]


AAAI27_DATASET_NAMES = (
    "mPowerRest",
    "mPowerReturn",
    "mPowerOutbound",
    "PADS_09_task06_DrinkGlas",
    "PADS_10_task07_CrossArms",
    "Shimmer_11_session11_DRINK",
    "Shimmer_12_session12_PICK",
)
AAAI27_EXPECTED_SPLIT_SAMPLES = {
    "mPowerRest": (16124, 5399, 5368),
    "mPowerReturn": (9132, 2619, 2733),
    "mPowerOutbound": (15289, 5278, 5193),
    "PADS_09_task06_DrinkGlas": (280, 92, 97),
    "PADS_10_task07_CrossArms": (280, 92, 97),
    "Shimmer_11_session11_DRINK": (77, 25, 28),
    "Shimmer_12_session12_PICK": (65, 21, 25),
}
AAAI27_DYNAMIC_DATASET_SPECS = {
    "PADS_10_task07_CrossArms": {
        "sequence_length": 976,
        "label_names": {
            0: "Healthy",
            1: "Parkinson",
            2: "OtherMovementDisorders",
        },
        "expected_subject_counts": (280, 92, 97),
        "expected_sample_count": 469,
    },
    "Shimmer_12_session12_PICK": {
        "sequence_length": 4096,
        "label_names": {0: "HC", 1: "MildPD", 2: "ModeratePD"},
        "expected_subject_counts": (65, 21, 25),
        "expected_sample_count": 111,
    },
}
AAAI27_REFERENCE_DATASETS_SHA256 = (
    "0113d69736e9678a43b8e2c62b344bb34e6776c023085f1a80e7e81b0a512092"
)
_AAAI27_MODULE_CACHE = {}


@dataclass
class AAAI27DataBundle:
    dataset_name: str
    data_root: Path
    reference_root: Path
    train_loader: DataLoader
    train_labels: np.ndarray
    vali_loader: DataLoader
    vali_labels: np.ndarray
    test_loader: DataLoader
    test_labels: np.ndarray
    train_dataset: Any
    vali_dataset: Any
    test_dataset: Any
    label_mode: str = "original"
    label_mapping: dict[int, int] | None = None


@dataclass
class FallTLComparisonBundle:
    train_loader: DataLoader
    train_labels: np.ndarray
    vali_loader: DataLoader
    vali_labels: np.ndarray
    test_loader: DataLoader
    test_labels: np.ndarray
    train_files: list[str]
    vali_files: list[str]
    test_files: list[str]


def linear_interpolation(data):
    n, d, l = data.shape
    result = data.copy()
    x = np.arange(l)

    for i in range(n):
        for j in range(d):
            y = data[i, j, :]
            nan_mask = np.isnan(y)
            if np.all(nan_mask):
                continue
            result[i, j, nan_mask] = np.interp(x[nan_mask], x[~nan_mask], y[~nan_mask])

    return result


def pad_samples(samples, padding_value=0, to_length=None):
    # Step 1: Find the maximum size of the second dimension
    if to_length is None:
        to_length = max([sample.shape[1] for sample in samples])

    output = np.zeros((len(samples), samples[0].shape[0], to_length))
    # Step 2: Pad each sample's second dimension using numpy.pad

    for i, sample in enumerate(samples):
        second_dim_len = sample.shape[1]

        # Pad the second dimension with the padding_value
        padded_sample = np.pad(
            sample,
            ((0, 0), (0, to_length - second_dim_len)),
            constant_values=padding_value,
        )

        # Stack the first dimension with the padded second dimension
        output[i] = padded_sample

    return output


def sample_equal_classes(train_data, train_labels, num_samples=1000):
    # Step 1: Get unique classes
    classes = np.unique(train_labels)

    # Step 2: Calculate the number of samples to be selected from each class
    num_classes = len(classes)
    samples_per_class = (
        num_samples // num_classes
    )  # Ensure total number of samples is exactly `num_samples`

    # Step 3: Sample equally from each class
    sampled_data = []
    sampled_labels = []

    for cls in classes:
        # Get indices of samples belonging to class `cls`
        class_indices = np.flatnonzero(train_labels == cls)

        # Randomly sample `samples_per_class` samples
        sampled_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )

        # Ensure that `sampled_indices` is a flat array of integers for proper indexing
        sampled_indices = sampled_indices.astype(int)

        # Append the sampled data and labels
        sampled_data.append(train_data[sampled_indices])
        sampled_labels.append(train_labels[sampled_indices])

    # Combine the data and labels into single arrays
    sampled_data = np.vstack(sampled_data)
    sampled_labels = np.hstack(sampled_labels)

    return sampled_data, sampled_labels


def find_uci_har_dir(data_dir):
    candidates = [
        data_dir,
        os.path.join(data_dir, "UCI HAR Dataset"),
    ]

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "train", "Inertial Signals")):
            return candidate

    raise FileNotFoundError(
        "Could not find UCI HAR Dataset. Expected either data_dir itself or "
        "data_dir/'UCI HAR Dataset' to contain train/Inertial Signals."
    )


def load_uci_har_split(data_dir, split, signal_files=UCI_HAR_SIGNAL_FILES):
    uci_dir = find_uci_har_dir(data_dir)
    signal_dir = os.path.join(uci_dir, split, "Inertial Signals")

    signals = []
    for signal_name in signal_files:
        path = os.path.join(signal_dir, f"{signal_name}_{split}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing UCI HAR signal file: {path}")
        signals.append(np.loadtxt(path, dtype=np.float32))

    data = np.stack(signals, axis=1)
    labels_path = os.path.join(uci_dir, split, f"y_{split}.txt")
    labels = np.loadtxt(labels_path, dtype=np.int64) - 1

    return data, labels


def find_preprocessed_har_dir(data_dir, dirname, required=True):
    candidates = [
        data_dir,
        os.path.join(data_dir, dirname),
        os.path.join(data_dir, dirname, dirname),
        os.path.join(data_dir, "med_data", dirname),
        os.path.join(data_dir, "med_data", dirname, dirname),
    ]

    seen = set()
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        feature_path = os.path.join(candidate, "Feature", "feature.npy")
        label_path = os.path.join(candidate, "Label", "label.npy")
        if os.path.isfile(feature_path) and os.path.isfile(label_path):
            return candidate

    if not required:
        return None

    raise FileNotFoundError(
        f"Could not find preprocessed {dirname}. Expected Feature/feature.npy "
        f"and Label/label.npy below {data_dir!r}, optionally under med_data/{dirname}."
    )


def load_preprocessed_har(data_dir, dirname, channel_indices=None):
    dataset_dir = find_preprocessed_har_dir(data_dir, dirname)
    feature_path = os.path.join(dataset_dir, "Feature", "feature.npy")
    label_path = os.path.join(dataset_dir, "Label", "label.npy")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(label_path, mmap_mode="r", allow_pickle=False)

    if features.ndim != 3:
        raise ValueError(
            f"Expected {dirname} features with shape (samples, time, channels), "
            f"got {features.shape}."
        )
    if labels.ndim != 1 or len(labels) != len(features):
        raise ValueError(
            f"Expected one {dirname} label per sample, got features={features.shape}, "
            f"labels={labels.shape}."
        )

    if channel_indices is None:
        channel_indices = list(range(features.shape[2]))
    else:
        channel_indices = list(channel_indices)
    if not channel_indices or len(set(channel_indices)) != len(channel_indices):
        raise ValueError(f"Channel indices must be non-empty and unique: {channel_indices}")
    if min(channel_indices) < 0 or max(channel_indices) >= features.shape[2]:
        raise ValueError(
            f"Channel indices {channel_indices} are invalid for {dirname} shape "
            f"{features.shape}."
        )

    # Downloaded Medformer arrays use (N, T, C); NeuroSigViT expects (N, C, T).
    data = np.asarray(
        features[:, :, channel_indices], dtype=np.float32
    ).transpose(0, 2, 1)
    encoded_labels = _encode_labels(np.asarray(labels))

    return data, encoded_labels


def _split_array_data(data, labels, test_ratio, random_seed):
    train_indices, test_indices = _split_indices(labels, test_ratio, random_seed)
    return (
        data[train_indices],
        labels[train_indices],
        data[test_indices],
        labels[test_indices],
    )


def find_dataset_dir(data_dir, dirname_candidates, required_glob="*.csv"):
    candidates = [data_dir]
    candidates.extend(os.path.join(data_dir, dirname) for dirname in dirname_candidates)

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        if required_glob is None:
            return candidate
        if _glob_csv_files(candidate, required_glob):
            return candidate

    joined = ", ".join(dirname_candidates)
    raise FileNotFoundError(
        f"Could not find dataset CSV files. Expected data_dir itself or one of "
        f"these subdirectories to contain {required_glob}: {joined}."
    )


def _glob_csv_files(data_dir, pattern):
    from glob import glob

    return sorted(
        path for path in glob(os.path.join(data_dir, pattern)) if os.path.isfile(path)
    )


def _validate_window_args(window_size, stride):
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")


def _read_csv(path):
    import pandas as pd

    return pd.read_csv(path)


def _check_columns(df, columns, csv_file):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {missing}")


def _numeric_values(df, feature_cols):
    import pandas as pd

    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    if np.isnan(values).any():
        values = linear_interpolation(values.T[None, :, :])[0].T
        values = np.nan_to_num(values, nan=0.0)

    return values


def _encode_labels(labels):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}

    return np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)


def _standardize_from_train(train_data, test_data):
    mean = np.nanmean(train_data, axis=(0, 2), keepdims=True)
    std = np.nanstd(train_data, axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    return (train_data - mean) / std, (test_data - mean) / std


def _split_indices(labels, test_ratio, random_seed):
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    _, counts = np.unique(labels, return_counts=True)
    test_count = int(np.ceil(len(labels) * test_ratio))
    train_count = len(labels) - test_count
    class_count = len(counts)
    stratify = (
        labels
        if np.all(counts >= 2) and test_count >= class_count and train_count >= class_count
        else None
    )

    if stratify is None:
        print(
            "Warning: at least one class has fewer than two windows; "
            "using a non-stratified train/test split."
        )

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify,
    )

    return train_indices, test_indices


def _resplit_data(
    train_data,
    train_labels,
    test_data,
    test_labels,
    test_ratio,
    random_seed,
):
    labels = np.concatenate((np.asarray(train_labels), np.asarray(test_labels)))
    train_indices, test_indices = _split_indices(labels, test_ratio, random_seed)

    if isinstance(train_data, list):
        data = train_data + test_data
        new_train_data = [data[index] for index in train_indices]
        new_test_data = [data[index] for index in test_indices]
    else:
        data = np.concatenate((train_data, test_data), axis=0)
        new_train_data = data[train_indices]
        new_test_data = data[test_indices]

    return (
        new_train_data,
        labels[train_indices],
        new_test_data,
        labels[test_indices],
    )


def _make_windows_from_segment(values, label, window_size, stride, samples, labels):
    if len(values) < window_size:
        return

    for start in range(0, len(values) - window_size + 1, stride):
        samples.append(values[start : start + window_size].T)
        labels.append(label)


def _iter_contiguous_label_segments(df, label_column):
    label_changes = df[label_column].ne(df[label_column].shift()).cumsum()
    for _, segment in df.groupby(label_changes, sort=False):
        label = segment[label_column].iloc[0]
        if label != label:
            continue
        yield label, segment


def _build_custom_split(samples, labels, test_ratio, random_seed):
    if not samples:
        raise ValueError(
            "No windows were created. Check data paths, labels, window_size, and stride."
        )

    data = np.asarray(samples, dtype=np.float32)
    labels = _encode_labels(labels)
    train_indices, test_indices = _split_indices(labels, test_ratio, random_seed)

    train_data = data[train_indices]
    test_data = data[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    train_data, test_data = _standardize_from_train(train_data, test_data)

    return (
        train_data.astype(np.float32),
        train_labels,
        test_data.astype(np.float32),
        test_labels,
    )


def load_falltl_data(
    data_dir,
    test_ratio=0.2,
    random_seed=None,
    window_size=200,
    stride=100,
    max_windows_per_file=None,
):
    _validate_window_args(window_size, stride)
    falltl_dir = find_dataset_dir(data_dir, ["FallTL", "falltl"], "*.csv")
    csv_files = _glob_csv_files(falltl_dir, "*.csv")

    samples = []
    labels = []

    for csv_file in csv_files:
        df = _read_csv(csv_file)
        _check_columns(df, FALLTL_FEATURE_COLUMNS, csv_file)

        if "Label" in df.columns:
            label_segments = _iter_contiguous_label_segments(df, "Label")
        else:
            filename_parts = os.path.splitext(os.path.basename(csv_file))[0].split("_")
            label = filename_parts[1] if len(filename_parts) >= 2 else filename_parts[0]
            label_segments = [(label, df)]

        created_for_file = 0
        for label, segment in label_segments:
            values = _numeric_values(segment, FALLTL_FEATURE_COLUMNS)
            before = len(samples)
            _make_windows_from_segment(
                values, label, window_size, stride, samples, labels
            )
            created_for_file += len(samples) - before

            if max_windows_per_file and created_for_file >= max_windows_per_file:
                extra = created_for_file - max_windows_per_file
                if extra > 0:
                    del samples[-extra:]
                    del labels[-extra:]
                break

    return _build_custom_split(samples, labels, test_ratio, random_seed)


def _natural_path_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", os.path.basename(path))
    ]


def _interpolate_falltl_sequence(values, source_file):
    values = np.asarray(values, dtype=np.float32)
    positions = np.arange(len(values), dtype=np.float32)
    interpolated = values.copy()
    for channel in range(values.shape[1]):
        observed = np.isfinite(values[:, channel])
        if not np.any(observed):
            raise ValueError(
                f"FallTL file {source_file} has no finite values in channel {channel}."
            )
        interpolated[:, channel] = np.interp(
            positions,
            positions[observed],
            values[observed, channel],
        )
    return interpolated


def _standardize_and_pad_falltl(train_sequences, *other_splits):
    train_points = np.concatenate(train_sequences, axis=0)
    mean = train_points.mean(axis=0, keepdims=True)
    std = train_points.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    padded_splits = []
    for sequences in (train_sequences, *other_splits):
        max_length = max(len(sequence) for sequence in sequences)
        padded = np.zeros(
            (len(sequences), train_points.shape[1], max_length), dtype=np.float32
        )
        for index, sequence in enumerate(sequences):
            standardized = ((sequence - mean) / std).astype(np.float32)
            padded[index, :, : len(sequence)] = standardized.T
        padded_splits.append(padded)
    return tuple(padded_splits)


def _load_falltl_comparison_arrays(data_dir):
    falltl_dir = find_dataset_dir(data_dir, ["FallTL", "falltl"], "*.csv")
    csv_files = sorted(
        _glob_csv_files(falltl_dir, "*.csv"), key=_natural_path_key
    )
    source_files = np.asarray([os.path.basename(path) for path in csv_files])
    labels = np.asarray(
        [1 if filename.startswith("F") else 0 for filename in source_files],
        dtype=np.int64,
    )
    sequences = []
    valid_labels = []
    valid_files = []
    for csv_file, label, source_file in zip(csv_files, labels, source_files):
        values = np.genfromtxt(
            csv_file,
            delimiter=",",
            skip_header=1,
            usecols=range(9),
            dtype=np.float32,
        )
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if len(values) == 0:
            continue
        sequences.append(_interpolate_falltl_sequence(values, source_file))
        valid_labels.append(label)
        valid_files.append(source_file)

    if not sequences:
        raise FileNotFoundError(f"No FallTL CSV files found below {data_dir}")
    return (
        sequences,
        np.asarray(valid_labels, dtype=np.int64),
        np.asarray(valid_files),
    )


def get_falltl_comparison_dataloaders(args):
    from sklearn.model_selection import train_test_split

    sequences, labels, source_files = _load_falltl_comparison_arrays(args.data_dir)
    all_indices = np.arange(len(labels))
    train_indices, remainder_indices = train_test_split(
        all_indices,
        test_size=0.4,
        random_state=42,
        stratify=labels,
    )
    vali_indices, test_indices = train_test_split(
        remainder_indices,
        test_size=0.5,
        random_state=42,
        stratify=labels[remainder_indices],
    )

    train_data, vali_data, test_data = _standardize_and_pad_falltl(
        [sequences[index] for index in train_indices],
        [sequences[index] for index in vali_indices],
        [sequences[index] for index in test_indices],
    )
    train_loader, vali_loader = _make_tensor_loaders(
        train_data, vali_data, args.batch_size
    )
    _, test_loader = _make_tensor_loaders(
        train_data, test_data, args.batch_size
    )
    bundle = FallTLComparisonBundle(
        train_loader=train_loader,
        train_labels=labels[train_indices],
        vali_loader=vali_loader,
        vali_labels=labels[vali_indices],
        test_loader=test_loader,
        test_labels=labels[test_indices],
        train_files=source_files[train_indices].tolist(),
        vali_files=source_files[vali_indices].tolist(),
        test_files=source_files[test_indices].tolist(),
    )
    distributions = []
    for split, split_labels in (
        ("train", bundle.train_labels),
        ("vali", bundle.vali_labels),
        ("test", bundle.test_labels),
    ):
        values, counts = np.unique(split_labels, return_counts=True)
        distributions.append(
            f"{split}=" + "/".join(
                f"{int(value)}:{int(count)}"
                for value, count in zip(values, counts)
            )
        )
    print(
        "FallTL comparison_binary: sequence_length=256; labels=D:0/F:1; "
        "split_seed=42; " + "; ".join(distributions)
    )
    return bundle


def write_falltl_comparison_split_audit(bundle, result_dir):
    split_dir = Path(result_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / "FallTL_comparison_binary_split.csv"
    rows = []
    for split, files, labels in (
        ("train", bundle.train_files, bundle.train_labels),
        ("vali", bundle.vali_files, bundle.vali_labels),
        ("test", bundle.test_files, bundle.test_labels),
    ):
        rows.extend(
            (filename, int(label), split)
            for filename, label in zip(files, labels)
        )
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "label_id", "split"])
        writer.writerows(sorted(rows))
    return output_path


def _find_feng_feature_columns(columns):
    available = set(columns)
    preferred = []

    for sensor in FENG_PREFERRED_SENSORS:
        for signal in FENG_PREFERRED_SIGNALS:
            for axis in ["X", "Y", "Z"]:
                candidates = [
                    f"{signal}_{axis}_{sensor}",
                    f"{signal}{axis}_{sensor}",
                    f"{sensor}_{signal}_{axis}",
                    f"{sensor}_{signal}{axis}",
                ]
                match = next((column for column in candidates if column in available), None)
                if match:
                    preferred.append(match)

    if preferred:
        return preferred

    excluded = {"Activity", "TimeStamp", "Timestamp", "Time", "Subject"}
    numeric_like = []
    for column in columns:
        if column in excluded:
            continue
        if any(token in column.lower() for token in ["acc", "gyr", "gyro", "quat"]):
            numeric_like.append(column)

    return numeric_like


def load_feng_data(
    data_dir,
    test_ratio=0.2,
    random_seed=None,
    window_size=200,
    stride=100,
    max_windows_per_file=None,
):
    _validate_window_args(window_size, stride)
    feng_dir = find_dataset_dir(
        data_dir,
        [
            "Feng et al.",
            os.path.join("Feng", "dataset"),
            "Feng",
            os.path.join("feng", "dataset"),
            "feng",
            "feng_et_al",
            "dataset",
        ],
        "P*.csv",
    )
    csv_files = _glob_csv_files(feng_dir, "P*.csv")

    samples = []
    labels = []

    for csv_file in csv_files:
        df = _read_csv(csv_file)
        _check_columns(df, ["Activity"], csv_file)
        feature_cols = _find_feng_feature_columns(df.columns)
        if not feature_cols:
            raise ValueError(
                f"Could not identify Feng feature columns in {csv_file}. "
                "Expected sensor columns containing Acc, Gyr/Gyro, or Quat."
            )

        created_for_file = 0
        for label, segment in _iter_contiguous_label_segments(df, "Activity"):
            values = _numeric_values(segment, feature_cols)
            before = len(samples)
            _make_windows_from_segment(
                values, label, window_size, stride, samples, labels
            )
            created_for_file += len(samples) - before

            if max_windows_per_file and created_for_file >= max_windows_per_file:
                extra = created_for_file - max_windows_per_file
                if extra > 0:
                    del samples[-extra:]
                    del labels[-extra:]
                break

    return _build_custom_split(samples, labels, test_ratio, random_seed)


def _make_tensor_loaders(train_data, test_data, batch_size):
    train_loader = DataLoader(
        TensorDataset(torch.Tensor(train_data).type(torch.float)),
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.Tensor(test_data).type(torch.float)),
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def find_aaai27_data_root(data_dir, dataset_name):
    if dataset_name not in AAAI27_DATASET_NAMES:
        raise ValueError(f"Unsupported AAAI27 dataset: {dataset_name}")

    base = Path(data_dir).expanduser()
    candidates = (
        base,
        base / "AAAI_Data",
        base / "med_data" / "AAAI_Data",
        base / "med_data",
    )
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if not (
            (normalized / dataset_name / "Meta" / "subject_map.csv").is_file()
            and (normalized / dataset_name / "Feature").is_dir()
        ):
            continue
        return normalized

    raise FileNotFoundError(
        f"Could not find AAAI_Data/{dataset_name} below {data_dir!r}. "
        "Expected the dataset Feature/ and Meta/subject_map.csv files."
    )


def _find_aaai27_reference_root(data_root):
    candidates = (
        Path(data_root) / "data_loading",
        Path(__file__).resolve().parents[1] / "data_loading",
    )
    for candidate in candidates:
        if (
            (candidate / "datasets.py").is_file()
            and (candidate / "split_reference_seed42.csv").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Missing AAAI27 reference loader and split file. Expected "
        "data_loading/datasets.py and data_loading/split_reference_seed42.csv "
        "either beside the dataset root or in the repository."
    )


def _load_aaai27_reference(reference_root):
    module_path = Path(reference_root) / "datasets.py"
    digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if digest != AAAI27_REFERENCE_DATASETS_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {module_path}: {digest}. "
            f"Expected {AAAI27_REFERENCE_DATASETS_SHA256}."
        )

    cache_key = str(module_path.resolve())
    if cache_key in _AAAI27_MODULE_CACHE:
        return _AAAI27_MODULE_CACHE[cache_key]

    module_name = f"_aaai27_reference_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AAAI27 reference module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _AAAI27_MODULE_CACHE[cache_key] = module
    return module


def _make_aaai27_tensor_loader(source_dataset, batch_size):
    if source_dataset.X is None or source_dataset.y is None:
        raise ValueError("AAAI27 source dataset did not load samples")

    # The reference interface is [N,T,6]; NeuroSigViT consumes [N,6,T].
    inputs = torch.from_numpy(source_dataset.X.transpose(0, 2, 1))
    tensor_dataset = TensorDataset(inputs)
    tensor_dataset.source_dataset = source_dataset
    tensor_dataset.sample_subject_ids = source_dataset.sample_subject_ids
    loader = DataLoader(
        tensor_dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=False,
    )
    labels = np.asarray(source_dataset.y, dtype=np.int64)
    return loader, labels


def _standardize_aaai27_bundle_from_train(bundle, batch_size):
    """Standardize selected task channels with training-split statistics only."""
    train_data = bundle.train_loader.dataset.tensors[0]
    mean = train_data.mean(dim=(0, 2), keepdim=True)
    std = train_data.std(dim=(0, 2), unbiased=False, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    for split in ("train", "vali", "test"):
        loader = getattr(bundle, f"{split}_loader")
        standardized = ((loader.dataset.tensors[0] - mean) / std).float()
        tensor_dataset = TensorDataset(standardized)
        tensor_dataset.source_dataset = loader.dataset.source_dataset
        tensor_dataset.sample_subject_ids = loader.dataset.sample_subject_ids
        setattr(
            bundle,
            f"{split}_loader",
            DataLoader(
                tensor_dataset,
                num_workers=0,
                batch_size=batch_size,
                shuffle=False,
            ),
        )


def _apply_aaai27_label_protocol(bundle, label_mode, batch_size):
    family = "shimmer" if bundle.dataset_name.startswith("Shimmer_") else "pads"
    protocols = {
        "shimmer_hc_vs_pd": ("shimmer", {0: 0, 1: 1, 2: 1}),
        "pads_pd_vs_hc": ("pads", {0: 0, 1: 1}),
        "pads_pd_vs_omd": ("pads", {2: 0, 1: 1}),
    }

    if label_mode == "original":
        labels = bundle.train_dataset.label_names
        mapping = {int(label): int(label) for label in labels}
    else:
        if label_mode not in protocols:
            raise ValueError(f"Unsupported AAAI27 label mode: {label_mode}")
        expected_family, mapping = protocols[label_mode]
        if family != expected_family:
            raise ValueError(
                f"Label mode {label_mode} is only valid for {expected_family} "
                f"datasets, not {bundle.dataset_name}."
            )

    for split in ("train", "vali", "test"):
        source_dataset = getattr(bundle, f"{split}_dataset")
        loader = getattr(bundle, f"{split}_loader")
        original_labels = np.asarray(source_dataset.y, dtype=np.int64)
        keep = np.isin(original_labels, np.asarray(list(mapping), dtype=np.int64))
        if not np.any(keep):
            raise ValueError(
                f"{bundle.dataset_name} split={split} has no samples for {label_mode}."
            )

        inputs = loader.dataset.tensors[0][torch.as_tensor(keep)]
        tensor_dataset = TensorDataset(inputs)
        tensor_dataset.source_dataset = source_dataset
        tensor_dataset.sample_subject_ids = np.asarray(
            source_dataset.sample_subject_ids
        )[keep]
        mapped_labels = np.asarray(
            [mapping[int(label)] for label in original_labels[keep]],
            dtype=np.int64,
        )
        setattr(
            bundle,
            f"{split}_loader",
            DataLoader(
                tensor_dataset,
                num_workers=0,
                batch_size=batch_size,
                shuffle=False,
            ),
        )
        setattr(bundle, f"{split}_labels", mapped_labels)

    bundle.label_mode = label_mode
    bundle.label_mapping = mapping


def _validate_aaai27_bundle(bundle, reference_module):
    reference_csv = bundle.reference_root / "split_reference_seed42.csv"
    reference = reference_module._read_reference_csv(reference_csv)
    if bundle.dataset_name in reference:
        reference_module._verify_reference_assignment(bundle.train_dataset, reference)
    reference_module._validate_label_file(bundle.train_dataset)

    split_items = (
        ("train", bundle.train_dataset, bundle.train_loader, bundle.train_labels),
        ("vali", bundle.vali_dataset, bundle.vali_loader, bundle.vali_labels),
        ("test", bundle.test_dataset, bundle.test_loader, bundle.test_labels),
    )
    split_sample_counts = []
    for split, source_dataset, loader, labels in split_items:
        if source_dataset.split != split:
            raise AssertionError(
                f"{bundle.dataset_name}: expected split={split}, got {source_dataset.split}"
            )
        expected_shape = (
            len(source_dataset),
            source_dataset.num_channels,
            source_dataset.sequence_length,
        )
        actual_shape = tuple(loader.dataset.tensors[0].shape)
        if actual_shape != expected_shape:
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: expected NeuroSigViT shape "
                f"{expected_shape}, got {actual_shape}"
            )
        if not np.array_equal(labels, source_dataset.y):
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: adapter labels changed"
            )
        expected_labels = np.asarray(
            [
                source_dataset.subject_label[int(subject_id)]
                for subject_id in source_dataset.sample_subject_ids
            ],
            dtype=np.int64,
        )
        if not np.array_equal(labels, expected_labels):
            raise AssertionError(
                f"{bundle.dataset_name} split={split}: sample labels do not match subjects"
            )
        split_sample_counts.append(len(source_dataset))

    actual_split_samples = tuple(split_sample_counts)
    expected_split_samples = AAAI27_EXPECTED_SPLIT_SAMPLES[bundle.dataset_name]
    if actual_split_samples != expected_split_samples:
        raise AssertionError(
            f"{bundle.dataset_name}: expected train/vali/test samples "
            f"{expected_split_samples}, got {actual_split_samples}"
        )
    total_samples = sum(actual_split_samples)
    if total_samples != bundle.train_dataset.expected_sample_count:
        raise AssertionError(
            f"{bundle.dataset_name}: expected "
            f"{bundle.train_dataset.expected_sample_count} total samples, "
            f"got {total_samples}"
        )


def get_aaai27_dataloaders(dataset_name, args):
    data_root = find_aaai27_data_root(args.data_dir, dataset_name)
    reference_root = _find_aaai27_reference_root(data_root)
    reference_module = _load_aaai27_reference(reference_root)
    dataset_classes = {
        dataset_class.dataset_name: dataset_class
        for dataset_class in reference_module.DATASET_CLASSES
    }
    for dynamic_name, spec in AAAI27_DYNAMIC_DATASET_SPECS.items():
        dataset_classes[dynamic_name] = type(
            f"{dynamic_name}Dataset",
            (reference_module.SubjectMapDataset,),
            {
                "dataset_name": dynamic_name,
                "relative_directories": (dynamic_name,),
                **spec,
            },
        )
    dataset_class = dataset_classes[dataset_name]

    split_datasets = {
        split: dataset_class(
            data_root=data_root,
            split=split,
            normalize=False,
            verbose=False,
        )
        for split in ("train", "vali", "test")
    }
    train_loader, train_labels = _make_aaai27_tensor_loader(
        split_datasets["train"], args.batch_size
    )
    vali_loader, vali_labels = _make_aaai27_tensor_loader(
        split_datasets["vali"], args.batch_size
    )
    test_loader, test_labels = _make_aaai27_tensor_loader(
        split_datasets["test"], args.batch_size
    )
    bundle = AAAI27DataBundle(
        dataset_name=dataset_name,
        data_root=data_root,
        reference_root=reference_root,
        train_loader=train_loader,
        train_labels=train_labels,
        vali_loader=vali_loader,
        vali_labels=vali_labels,
        test_loader=test_loader,
        test_labels=test_labels,
        train_dataset=split_datasets["train"],
        vali_dataset=split_datasets["vali"],
        test_dataset=split_datasets["test"],
    )
    _validate_aaai27_bundle(bundle, reference_module)

    label_mode = getattr(args, "aaai27_label_mode", "original")
    _apply_aaai27_label_protocol(bundle, label_mode, args.batch_size)
    _standardize_aaai27_bundle_from_train(bundle, args.batch_size)

    split_distributions = []
    for split, labels in (
        ("train", bundle.train_labels),
        ("vali", bundle.vali_labels),
        ("test", bundle.test_labels),
    ):
        values, counts = np.unique(labels, return_counts=True)
        distribution = "/".join(
            f"{int(value)}:{int(count)}" for value, count in zip(values, counts)
        )
        split_distributions.append(f"{split}=[{distribution}]")
    print(
        f"AAAI27 {dataset_name}: "
        f"train={len(bundle.train_labels)}, vali={len(bundle.vali_labels)}, "
        f"test={len(bundle.test_labels)}; "
        f"label_mode={label_mode}; {' '.join(split_distributions)}; "
        "subject split/reference=PASS"
    )
    return bundle


def write_aaai27_split_audit(bundle, result_dir):
    split_dir = Path(result_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / f"{bundle.dataset_name}_subject_split.csv"

    assignments = []
    for split, source_dataset in (
        ("train", bundle.train_dataset),
        ("vali", bundle.vali_dataset),
        ("test", bundle.test_dataset),
    ):
        assignments.extend(
            (
                int(subject_id),
                int(source_dataset.subject_label[int(subject_id)]),
                split,
            )
            for subject_id in source_dataset.selected_subject_ids
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset_name",
            "label_protocol",
            "numeric_subject_id",
            "original_label_id",
            "label_id",
            "split",
        ])
        for subject_id, original_label_id, split in sorted(assignments):
            if original_label_id not in bundle.label_mapping:
                continue
            label_id = bundle.label_mapping[original_label_id]
            writer.writerow(
                [
                    bundle.dataset_name,
                    bundle.label_mode,
                    subject_id,
                    original_label_id,
                    label_id,
                    split,
                ]
            )
    return output_path


def get_dataloader(dataset, args):
    har_channels = getattr(args, "har_channels", "all")
    if har_channels not in {"all", "acc_gyro"}:
        raise ValueError(f"Unsupported HAR channel subset: {har_channels}")

    if args.datasets == "flaap":
        data, labels = load_preprocessed_har(
            args.data_dir,
            "FLAAP",
            channel_indices=range(6),
        )
        if data.shape[1] != 6:
            raise ValueError(f"Expected FLAAP to contain 6 channels, got {data.shape}.")
        train_data, train_labels, test_data, test_labels = _split_array_data(
            data,
            labels,
            args.custom_test_ratio,
            args.random_seed,
        )
        train_loader, test_loader = _make_tensor_loaders(
            train_data, test_data, args.batch_size
        )
        return train_loader, train_labels, test_loader, test_labels

    if args.datasets == "uci":
        uci_protocol = getattr(args, "uci_protocol", "official_subject")
        signal_files = UCI_HAR_SIGNAL_FILES
        channel_indices = range(9)
        expected_channels = 9
        if har_channels == "acc_gyro":
            signal_files = UCI_HAR_ACC_GYRO_SIGNAL_FILES
            channel_indices = UCI_HAR_ACC_GYRO_INDICES
            expected_channels = 6

        raw_uci_dir = None
        try:
            raw_uci_dir = find_uci_har_dir(args.data_dir)
        except FileNotFoundError:
            pass

        if uci_protocol == "official_subject":
            if raw_uci_dir is None:
                raise FileNotFoundError(
                    "The official UCI-HAR protocol requires the original "
                    "train/Inertial Signals and test/Inertial Signals directories; "
                    "a combined Feature/feature.npy file cannot recover the "
                    "subject-disjoint split. Use --uci_protocol legacy_resplit "
                    "only for auditing historical results."
                )
            train_data, train_labels = load_uci_har_split(
                raw_uci_dir, "train", signal_files=signal_files
            )
            test_data, test_labels = load_uci_har_split(
                raw_uci_dir, "test", signal_files=signal_files
            )
        elif uci_protocol == "legacy_resplit":
            if raw_uci_dir is not None:
                train_data, train_labels = load_uci_har_split(
                    raw_uci_dir, "train", signal_files=signal_files
                )
                test_data, test_labels = load_uci_har_split(
                    raw_uci_dir, "test", signal_files=signal_files
                )
                train_data, train_labels, test_data, test_labels = _resplit_data(
                    train_data,
                    train_labels,
                    test_data,
                    test_labels,
                    args.custom_test_ratio,
                    args.random_seed,
                )
            else:
                data, labels = load_preprocessed_har(
                    args.data_dir,
                    "UCI-HAR",
                    channel_indices=channel_indices,
                )
                train_data, train_labels, test_data, test_labels = _split_array_data(
                    data,
                    labels,
                    args.custom_test_ratio,
                    args.random_seed,
                )
        else:
            raise ValueError(f"Unsupported UCI-HAR protocol: {uci_protocol}")

        if train_data.shape[1] != expected_channels:
            raise ValueError(
                f"Expected UCI-HAR {har_channels} data to contain "
                f"{expected_channels} channels, got {train_data.shape}."
            )

        train_loader, test_loader = _make_tensor_loaders(
            train_data, test_data, args.batch_size
        )

        return train_loader, train_labels, test_loader, test_labels

    if args.datasets == "falltl":
        train_data, train_labels, test_data, test_labels = load_falltl_data(
            args.data_dir,
            test_ratio=args.custom_test_ratio,
            random_seed=args.random_seed,
            window_size=args.window_size,
            stride=args.window_stride,
            max_windows_per_file=args.max_windows_per_file,
        )

        train_loader, test_loader = _make_tensor_loaders(
            train_data, test_data, args.batch_size
        )

        return train_loader, train_labels, test_loader, test_labels

    if args.datasets == "feng":
        train_data, train_labels, test_data, test_labels = load_feng_data(
            args.data_dir,
            test_ratio=args.custom_test_ratio,
            random_seed=args.random_seed,
            window_size=args.window_size,
            stride=args.window_stride,
            max_windows_per_file=args.max_windows_per_file,
        )

        train_loader, test_loader = _make_tensor_loaders(
            train_data, test_data, args.batch_size
        )

        return train_loader, train_labels, test_loader, test_labels

    data_dir = f"{args.data_dir}/{str(args.datasets).upper()}"

    train_data, train_labels = load_classification(
        dataset,
        split="train",
        extract_path=data_dir,
        load_equal_length=(args.aeon or (args.datasets == "uea")),
        load_no_missing=(args.aeon or (args.datasets == "uea")),
    )
    test_data, test_labels = load_classification(
        dataset,
        split="test",
        extract_path=data_dir,
        load_equal_length=(args.aeon or (args.datasets == "uea")),
        load_no_missing=(args.aeon or (args.datasets == "uea")),
    )

    if dataset == "InsectWingbeat":
        # Downsample before recombining so the final split still follows 60/20/20.
        train_data, train_labels = sample_equal_classes(
            train_data, train_labels, num_samples=1000
        )
        test_data, test_labels = sample_equal_classes(
            test_data, test_labels, num_samples=1000
        )

    train_data, train_labels, test_data, test_labels = _resplit_data(
        train_data,
        train_labels,
        test_data,
        test_labels,
        args.custom_test_ratio,
        args.random_seed,
    )

    # Preprocessing
    if args.datasets == "ucr" and not args.aeon:
        # Padding if time series are of different length
        if isinstance(train_data, list):
            to_length = max(
                np.unique([sample.shape[1] for sample in train_data + test_data])
            )
            train_data = pad_samples(train_data, to_length=to_length)
            test_data = pad_samples(test_data, to_length=to_length)

        # Linear interpolation for missing values
        if np.isnan(train_data).any():
            train_data = linear_interpolation(train_data)

        if np.isnan(test_data).any():
            test_data = linear_interpolation(test_data)

        # Standard normalization
        if (np.abs(train_data.mean()) > 0.01) or (np.abs(train_data.std() - 1) > 0.01):
            mean = np.nanmean(train_data)
            std = np.nanstd(train_data)
            train_data = (train_data - mean) / std
            test_data = (test_data - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.Tensor(train_data).type(torch.float)),
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.Tensor(test_data).type(torch.float)),
        num_workers=0,
        batch_size=args.batch_size,
        shuffle=False,
    )

    return train_loader, train_labels, test_loader, test_labels
