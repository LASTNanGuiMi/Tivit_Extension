#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


DATASETS = ("Feng", "FallTL", "UCIHAR")
FUSIONS = ("concat", "concat_attn", "cross_attn_gate", "masked_pretrain")
METRICS = (
    "test_macro_f1",
    "test_macro_recall",
    "test_macro_auprc",
    "test_macro_precision",
    "test_macro_auroc",
    "test_accuracy",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate repeated Seed 2022 runs.")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
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


def load_records(result_root, repeats):
    records = {}
    for repeat in range(1, repeats + 1):
        repeat_dir = result_root / f"repeat_{repeat}"
        if not repeat_dir.is_dir():
            raise FileNotFoundError(f"Missing repeat directory: {repeat_dir}")
        for args_path in sorted(repeat_dir.glob("*/args.json")):
            csv_path = args_path.parent / "train_val.csv"
            if not csv_path.is_file():
                continue
            config = json.loads(args_path.read_text())
            if int(config["random_seed"]) != 2022:
                continue
            fusion = config["modal_interaction"]
            with csv_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    dataset = row["dataset"]
                    if dataset not in DATASETS or fusion not in FUSIONS:
                        continue
                    key = (repeat, dataset, fusion)
                    if key in records:
                        raise ValueError(f"Duplicate result for {key}: {csv_path}")
                    records[key] = {metric: float(row[metric]) for metric in METRICS}

    expected = {
        (repeat, dataset, fusion)
        for repeat in range(1, repeats + 1)
        for dataset in DATASETS
        for fusion in FUSIONS
    }
    missing = sorted(expected - records.keys())
    if missing:
        raise ValueError(f"Missing {len(missing)} results: {missing}")
    return records


def aggregate(records, repeats):
    rows = []
    for dataset in DATASETS:
        for fusion in FUSIONS:
            row = {
                "dataset": dataset,
                "modal_interaction": fusion,
                "random_seed": 2022,
                "n_repeats": repeats,
            }
            for metric in METRICS:
                values = [records[(repeat, dataset, fusion)][metric] for repeat in range(1, repeats + 1)]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if repeats > 1 else 0.0
            rows.append(row)
    return rows


def main():
    args = parse_args()
    wait_for_workers(args.status_dir, args.workers, args.poll_seconds)
    records = load_records(args.result_root, args.repeats)
    rows = aggregate(records, args.repeats)
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
