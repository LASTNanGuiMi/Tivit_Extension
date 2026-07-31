#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


DATASETS = {"flaap": "FLAAP", "uci": "UCIHAR"}
FUSIONS = ("concat", "concat_attn", "cross_attn_gate", "masked_pretrain")
METRICS = (
    "test_macro_f1",
    "test_macro_recall",
    "test_macro_auprc",
    "test_macro_precision",
    "test_macro_auroc",
    "test_accuracy",
)
CHANNEL_NAMES = {
    "FLAAP": "Acc_X,Acc_Y,Acc_Z,Gyr_X,Gyr_Y,Gyr_Z",
    "UCIHAR": (
        "total_acc_x,total_acc_y,total_acc_z,"
        "body_gyro_x,body_gyro_y,body_gyro_z"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate repeated six-channel HAR runs.")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="Dataset keys to aggregate (default: flaap uci)",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def wait_for_workers(status_dir, workers, poll_seconds):
    print(f"Waiting for {workers} workers in {status_dir}", flush=True)
    while True:
        statuses = {
            path.name: path.read_text().strip()
            for path in status_dir.glob("worker_*.status")
        }
        if len(statuses) >= workers:
            failures = {name: value for name, value in statuses.items() if value != "0"}
            if failures:
                raise RuntimeError(f"Workers failed: {failures}")
            return
        print(f"Workers complete: {len(statuses)}/{workers}", flush=True)
        time.sleep(poll_seconds)


def load_task_record(task_dir, expected_dataset, expected_fusion):
    matches = []
    for args_path in sorted(task_dir.glob("*/args.json")):
        csv_path = args_path.parent / "train_val.csv"
        if not csv_path.is_file():
            continue
        config = json.loads(args_path.read_text())
        if (
            int(config["random_seed"]) != 2022
            or config.get("har_channels") != "acc_gyro"
            or config["modal_interaction"] != expected_fusion
        ):
            continue
        with csv_path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["dataset"] == expected_dataset]
        if len(rows) == 1:
            matches.append((csv_path, rows[0]))

    if len(matches) != 1:
        paths = [str(path) for path, _ in matches]
        raise ValueError(
            f"Expected one completed result in {task_dir}, found {len(matches)}: {paths}"
        )
    return {metric: float(matches[0][1][metric]) for metric in METRICS}


def load_records(result_root, repeats, datasets=tuple(DATASETS)):
    records = {}
    for repeat in range(1, repeats + 1):
        for dataset_key in datasets:
            dataset_name = DATASETS[dataset_key]
            for fusion in FUSIONS:
                task_dir = result_root / dataset_key / f"repeat_{repeat}" / fusion
                if not task_dir.is_dir():
                    raise FileNotFoundError(f"Missing task directory: {task_dir}")
                records[(repeat, dataset_name, fusion)] = load_task_record(
                    task_dir,
                    dataset_name,
                    fusion,
                )
    return records


def aggregate(records, repeats, datasets=tuple(DATASETS)):
    rows = []
    for dataset_key in datasets:
        dataset_name = DATASETS[dataset_key]
        for fusion in FUSIONS:
            row = {
                "dataset": dataset_name,
                "channels": 6,
                "channel_names": CHANNEL_NAMES[dataset_name],
                "modal_interaction": fusion,
                "random_seed": 2022,
                "n_repeats": repeats,
            }
            for metric in METRICS:
                values = [
                    records[(repeat, dataset_name, fusion)][metric]
                    for repeat in range(1, repeats + 1)
                ]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = (
                    statistics.stdev(values) if repeats > 1 else 0.0
                )
            rows.append(row)
    return rows


def main():
    args = parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Provide each dataset at most once")
    datasets = tuple(args.datasets)
    wait_for_workers(args.status_dir, args.workers, args.poll_seconds)
    rows = aggregate(
        load_records(args.result_root, args.repeats, datasets),
        args.repeats,
        datasets,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} averages to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        raise
