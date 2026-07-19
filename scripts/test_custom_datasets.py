import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse
from collections import Counter
from types import SimpleNamespace

import numpy as np

from src.datautils import get_dataloader


def summarize_labels(labels):
    counts = Counter(np.asarray(labels).tolist())
    return ", ".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def test_dataset(name, args):
    loader_args = SimpleNamespace(
        datasets=name,
        data_dir=args.data_dir,
        custom_test_ratio=args.custom_test_ratio,
        random_seed=args.random_seed,
        window_size=args.window_size,
        window_stride=args.window_stride,
        max_windows_per_file=args.max_windows_per_file,
        batch_size=args.batch_size,
    )

    dataset_name = "FallTL" if name == "falltl" else "Feng"
    train_loader, train_labels, test_loader, test_labels = get_dataloader(
        dataset_name, loader_args
    )

    train_x = train_loader.dataset.tensors[0]
    test_x = test_loader.dataset.tensors[0]
    first_batch = next(iter(train_loader))[0]

    print(f"\n[{name}] OK")
    print(f"train shape: {tuple(train_x.shape)}")
    print(f"test shape:  {tuple(test_x.shape)}")
    print(f"batch shape: {tuple(first_batch.shape)}")
    print(f"train labels: {summarize_labels(train_labels)}")
    print(f"test labels:  {summarize_labels(test_labels)}")
    print(f"mean/std: train={train_x.mean().item():.6f}/{train_x.std().item():.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke test FallTL and Feng custom CSV dataset loaders."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["falltl", "feng"],
        choices=["falltl", "feng"],
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--window_stride", type=int, default=100)
    parser.add_argument("--custom_test_ratio", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument(
        "--max_windows_per_file",
        type=int,
        default=None,
        help="Only applies to Feng.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    for dataset in parsed.datasets:
        test_dataset(dataset, parsed)
