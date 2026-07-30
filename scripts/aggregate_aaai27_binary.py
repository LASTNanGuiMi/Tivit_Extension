#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import time
from pathlib import Path


DATASETS = (
    "PADS_09_task06_DrinkGlas",
    "PADS_10_task07_CrossArms",
    "Shimmer_11_session11_DRINK",
    "Shimmer_12_session12_PICK",
)
FUSIONS = ("concat", "concat_attn", "cross_attn_gate", "masked_pretrain")
METRICS = (
    "val_accuracy",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
    "val_macro_auroc",
    "val_macro_auprc",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_macro_auroc",
    "test_macro_auprc",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate AAAI27 binary runs.")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def wait_for_workers(status_dir, workers, poll_seconds):
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


def config_matches(config, dataset, fusion):
    return (
        config.get("datasets") == "aaai27"
        and config.get("dataset_names") == [dataset]
        and config.get("aaai27_label_mode") == "zero_vs_rest"
        and config.get("random_seed") == 2022
        and config.get("image_mode") == "activity_graph"
        and config.get("vit_1_layer") == 14
        and config.get("mantis") is True
        and config.get("classifier_type") == "mlp"
        and config.get("modal_interaction") == fusion
        and config.get("fusion_dim") == 128
        and config.get("fusion_heads") == 2
        and config.get("mlp_hidden_dim") == 128
        and config.get("mlp_num_layers") == 1
        and config.get("mlp_lr") == 3e-4
        and config.get("mlp_weight_decay") == 1e-3
        and config.get("mlp_class_weight") == "balanced"
        and config.get("mlp_epochs") == 40
        and config.get("mlp_early_stop_patience") == 8
        and config.get("batch_size") == 16
        and bool(config.get("feature_cache_dir"))
    )


def load_result(task_dir, dataset, fusion):
    matches = []
    for args_path in sorted(task_dir.glob("*/args.json")):
        result_path = args_path.parent / "train_val.csv"
        if not result_path.is_file():
            continue
        config = json.loads(args_path.read_text())
        if not config_matches(config, dataset, fusion):
            continue
        with result_path.open(newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("dataset") == dataset
            ]
        if len(rows) == 1:
            matches.append((args_path, rows[0]))

    if len(matches) != 1:
        paths = [str(path) for path, _ in matches]
        raise ValueError(
            f"Expected one result for {dataset}/{fusion}, found "
            f"{len(matches)}: {paths}"
        )

    args_path, result = matches[0]
    row = {
        "dataset": dataset,
        "label_mapping": "0_vs_1_or_2",
        "fusion": fusion,
        "random_seed": 2022,
        "args_path": str(args_path),
    }
    row.update({metric: float(result[metric]) for metric in METRICS})
    return row


def main():
    args = parse_args()
    wait_for_workers(args.status_dir, args.workers, args.poll_seconds)
    rows = [
        load_result(args.result_root / dataset / fusion, dataset, fusion)
        for dataset in DATASETS
        for fusion in FUSIONS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} AAAI27 binary results to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        raise
