import numpy as np
import torch
from aeon.datasets import load_classification
from torch.utils.data import DataLoader, TensorDataset


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


def get_dataloader(dataset, args):
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
