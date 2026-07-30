#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datautils import (  # noqa: E402
    UCI_HAR_ACC_GYRO_INDICES,
    find_preprocessed_har_dir,
    get_dataloader,
    load_preprocessed_har,
)


EXPECTED = {
    "flaap": {"dataset": "FLAAP", "samples": 13123, "time": 100, "classes": 10},
    "uci": {"dataset": "UCI-HAR", "samples": 10299, "time": 128, "classes": 6},
}


def loader_args(dataset, data_dir, batch_size):
    return SimpleNamespace(
        datasets=dataset,
        data_dir=data_dir,
        har_channels="acc_gyro",
        custom_test_ratio=0.2,
        random_seed=2022,
        batch_size=batch_size,
    )


def check_raw_mapping(data_dir):
    flaap_dir = Path(find_preprocessed_har_dir(data_dir, "FLAAP"))
    flaap_raw = np.load(flaap_dir / "Feature" / "feature.npy", mmap_mode="r")
    flaap_data, _ = load_preprocessed_har(data_dir, "FLAAP", range(6))
    np.testing.assert_array_equal(flaap_data[0], flaap_raw[0].T.astype(np.float32))

    uci_dir = Path(find_preprocessed_har_dir(data_dir, "UCI-HAR"))
    uci_raw = np.load(uci_dir / "Feature" / "feature.npy", mmap_mode="r")
    uci_data, _ = load_preprocessed_har(
        data_dir,
        "UCI-HAR",
        UCI_HAR_ACC_GYRO_INDICES,
    )
    expected = uci_raw[0][:, UCI_HAR_ACC_GYRO_INDICES].T.astype(np.float32)
    np.testing.assert_array_equal(uci_data[0], expected)


def check_loader(dataset, data_dir, batch_size):
    config = EXPECTED[dataset]
    train_loader, train_labels, test_loader, test_labels = get_dataloader(
        config["dataset"],
        loader_args(dataset, data_dir, batch_size),
    )
    train_shape = tuple(train_loader.dataset.tensors[0].shape)
    test_shape = tuple(test_loader.dataset.tensors[0].shape)

    assert train_shape[1:] == (6, config["time"]), train_shape
    assert test_shape[1:] == (6, config["time"]), test_shape
    assert train_shape[0] + test_shape[0] == config["samples"]
    assert len(np.unique(np.concatenate((train_labels, test_labels)))) == config["classes"]

    print(
        f"{config['dataset']}: train={train_shape}, test={test_shape}, "
        f"classes={config['classes']}"
    )


def main():
    parser = argparse.ArgumentParser(description="Validate FLAAP/UCI-HAR six-channel loading.")
    parser.add_argument(
        "--data-dir",
        default="/home/xuzheyuan/guoyin/data/med_data",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    check_raw_mapping(args.data_dir)
    check_loader("flaap", args.data_dir, args.batch_size)
    check_loader("uci", args.data_dir, args.batch_size)
    print("HAR channel checks passed.")


if __name__ == "__main__":
    main()
