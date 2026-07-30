#!/usr/bin/env python3
"""Aggregate the two remaining AAAI27 binary ablation comparisons."""

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
CONDITIONS = (
    "vision_line_plot",
    "vision_activity_graph",
    "timeseries_mantis",
    "multimodal_concat",
)
REPRESENTATION_CONDITIONS = (
    "vision_line_plot",
    "vision_activity_graph",
)
FEATURE_CONDITIONS = (
    "timeseries_mantis",
    "vision_activity_graph",
    "multimodal_concat",
)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output-all", required=True, type=Path)
    parser.add_argument("--output-representation", required=True, type=Path)
    parser.add_argument("--output-feature", required=True, type=Path)
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


def config_matches(config, dataset, condition):
    expected_image_mode = (
        "multichannel_line_plot"
        if condition == "vision_line_plot"
        else "activity_graph"
    )
    expected_vision = condition != "timeseries_mantis"
    expected_mantis = condition in {"timeseries_mantis", "multimodal_concat"}
    return (
        config.get("datasets") == "aaai27"
        and config.get("dataset_names") == [dataset]
        and config.get("aaai27_label_mode") == "zero_vs_rest"
        and config.get("random_seed") == 2022
        and config.get("image_mode") == expected_image_mode
        and bool(config.get("vit_1_name")) == expected_vision
        and config.get("vit_1_layer") == (14 if expected_vision else None)
        and config.get("mantis") is expected_mantis
        and config.get("classifier_type") == "mlp"
        and config.get("modal_interaction") == "concat"
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


def load_result(task_dir, dataset, condition):
    matches = []
    for args_path in sorted(task_dir.glob("*/args.json")):
        result_path = args_path.parent / "train_val.csv"
        if not result_path.is_file():
            continue
        config = json.loads(args_path.read_text())
        if not config_matches(config, dataset, condition):
            continue
        with result_path.open(newline="") as handle:
            rows = [
                row for row in csv.DictReader(handle) if row.get("dataset") == dataset
            ]
        if len(rows) == 1:
            matches.append((args_path, rows[0]))

    if len(matches) != 1:
        paths = [str(path) for path, _ in matches]
        raise ValueError(
            f"Expected one result for {dataset}/{condition}, found "
            f"{len(matches)}: {paths}"
        )

    args_path, result = matches[0]
    row = {
        "dataset": dataset,
        "label_mapping": "0_vs_1_or_2",
        "condition": condition,
        "random_seed": 2022,
        "args_path": str(args_path),
    }
    row.update({metric: float(result[metric]) for metric in METRICS})
    return row


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    args = parse_args()
    wait_for_workers(args.status_dir, args.workers, args.poll_seconds)
    records = {
        (dataset, condition): load_result(
            args.result_root / dataset / condition, dataset, condition
        )
        for dataset in DATASETS
        for condition in CONDITIONS
    }
    all_rows = [
        records[(dataset, condition)]
        for dataset in DATASETS
        for condition in CONDITIONS
    ]
    representation_rows = [
        records[(dataset, condition)]
        for dataset in DATASETS
        for condition in REPRESENTATION_CONDITIONS
    ]
    feature_rows = [
        records[(dataset, condition)]
        for dataset in DATASETS
        for condition in FEATURE_CONDITIONS
    ]
    write_rows(args.output_all, all_rows)
    write_rows(args.output_representation, representation_rows)
    write_rows(args.output_feature, feature_rows)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        raise
