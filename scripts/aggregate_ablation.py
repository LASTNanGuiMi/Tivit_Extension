#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


CONDITIONS = (
    "vision_line_plot",
    "vision_activity_graph",
    "timeseries_mantis",
    "multimodal_concat",
    "multimodal_proposed",
)
DATASET_NAMES = {"feng": "Feng", "falltl": "FallTL", "uci": "UCIHAR"}
METRICS = (
    "test_macro_f1",
    "test_macro_recall",
    "test_macro_auprc",
    "test_macro_precision",
    "test_macro_auroc",
    "test_accuracy",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Tivit ablation results.")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--datasets", required=True, nargs="+", choices=DATASET_NAMES)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def wait_for_workers(status_dir, workers, poll_seconds):
    if not status_dir or workers <= 0:
        return
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


def load_records(root, seeds, repeats, datasets):
    records = {}
    for condition in CONDITIONS:
        for seed in seeds:
            for repeat in range(1, repeats + 1):
                run_dir = root / condition / f"seed_{seed}" / f"repeat_{repeat}"
                for args_path in sorted(run_dir.glob("*/args.json")):
                    csv_path = args_path.parent / "train_val.csv"
                    if not csv_path.is_file():
                        continue
                    config = json.loads(args_path.read_text())
                    if int(config["random_seed"]) != seed:
                        raise ValueError(f"Seed mismatch in {args_path}")
                    with csv_path.open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    if len(rows) != 1:
                        raise ValueError(f"Expected one result row in {csv_path}")
                    dataset = rows[0]["dataset"]
                    key = (condition, dataset, seed, repeat)
                    if key in records:
                        raise ValueError(f"Duplicate result for {key}")
                    records[key] = {metric: float(rows[0][metric]) for metric in METRICS}

    expected = {
        (condition, DATASET_NAMES[dataset], seed, repeat)
        for condition in CONDITIONS
        for dataset in datasets
        for seed in seeds
        for repeat in range(1, repeats + 1)
    }
    missing = sorted(expected - records.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} results: {missing}")
    return records


def aggregate(records, seeds, repeats, datasets):
    rows = []
    for dataset_group in datasets:
        dataset = DATASET_NAMES[dataset_group]
        for condition in CONDITIONS:
            keys = [
                (condition, dataset, seed, repeat)
                for seed in seeds
                for repeat in range(1, repeats + 1)
            ]
            row = {
                "dataset": dataset,
                "condition": condition,
                "seeds": "|".join(map(str, seeds)),
                "n_runs": len(keys),
            }
            for metric in METRICS:
                values = [records[key][metric] for key in keys]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.pstdev(values)
            rows.append(row)
    return rows


def main():
    args = parse_args()
    wait_for_workers(args.status_dir, args.workers, args.poll_seconds)
    records = load_records(args.result_root, args.seeds, args.repeats, args.datasets)
    rows = aggregate(records, args.seeds, args.repeats, args.datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ablation summaries to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        raise
