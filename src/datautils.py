import os

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


def load_uci_har_split(data_dir, split):
    uci_dir = find_uci_har_dir(data_dir)
    signal_dir = os.path.join(uci_dir, split, "Inertial Signals")

    signals = []
    for signal_name in UCI_HAR_SIGNAL_FILES:
        path = os.path.join(signal_dir, f"{signal_name}_{split}.txt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing UCI HAR signal file: {path}")
        signals.append(np.loadtxt(path, dtype=np.float32))

    data = np.stack(signals, axis=1)
    labels_path = os.path.join(uci_dir, split, f"y_{split}.txt")
    labels = np.loadtxt(labels_path, dtype=np.int64) - 1

    return data, labels


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

    return sorted(glob(os.path.join(data_dir, pattern)))


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
        ["Feng et al.", "Feng", "feng", "feng_et_al"],
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
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.Tensor(test_data).type(torch.float)),
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def get_dataloader(dataset, args):
    if args.datasets == "uci":
        train_data, train_labels = load_uci_har_split(args.data_dir, "train")
        test_data, test_labels = load_uci_har_split(args.data_dir, "test")

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

    if dataset == "InsectWingbeat":
        # Downsample big dataset with stratification per class
        train_data, train_labels = sample_equal_classes(
            train_data, train_labels, num_samples=1000
        )
        test_data, test_labels = sample_equal_classes(
            test_data, test_labels, num_samples=1000
        )

    train_loader = DataLoader(
        TensorDataset(torch.Tensor(train_data).type(torch.float)),
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.Tensor(test_data).type(torch.float)),
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=False,
    )

    return train_loader, train_labels, test_loader, test_labels
