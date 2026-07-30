#!/usr/bin/env python3
"""Aggregate fixed-split AAAI27 runs across outer TiVit training seeds."""

import argparse
import csv
import hashlib
import json
import statistics
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
    "concat_attn",
    "cross_attn_gate",
    "masked_pretrain",
)
REPRESENTATION_CONDITIONS = ("vision_line_plot", "vision_activity_graph")
FEATURE_CONDITIONS = (
    "timeseries_mantis",
    "vision_activity_graph",
    "multimodal_concat",
)
FUSION_CONDITIONS = (
    "multimodal_concat",
    "concat_attn",
    "cross_attn_gate",
    "masked_pretrain",
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
TEST_METRICS = tuple(metric for metric in METRICS if metric.startswith("test_"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--outer-seeds", required=True, nargs="+", type=int)
    parser.add_argument("--split-reference", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def expected_condition(condition):
    image_mode = "multichannel_line_plot" if condition == "vision_line_plot" else "activity_graph"
    has_vision = condition != "timeseries_mantis"
    has_mantis = condition not in {"vision_line_plot", "vision_activity_graph"}
    modal_interaction = {
        "multimodal_concat": "concat",
        "vision_line_plot": "concat",
        "vision_activity_graph": "concat",
        "timeseries_mantis": "concat",
    }.get(condition, condition)
    return image_mode, has_vision, has_mantis, modal_interaction


def config_matches(config, dataset, condition, outer_seed):
    image_mode, has_vision, has_mantis, modal_interaction = expected_condition(condition)
    return (
        config.get("datasets") == "aaai27"
        and config.get("dataset_names") == [dataset]
        and config.get("aaai27_label_mode") == "zero_vs_rest"
        and config.get("random_seed") == outer_seed
        and config.get("image_mode") == image_mode
        and bool(config.get("vit_1_name")) == has_vision
        and config.get("vit_1_layer") == (14 if has_vision else None)
        and config.get("mantis") is has_mantis
        and config.get("classifier_type") == "mlp"
        and config.get("modal_interaction") == modal_interaction
        and config.get("fusion_dim") == 128
        and config.get("fusion_heads") == 2
        and config.get("cross_attn_query") == "ts"
        and config.get("mask_prob") == 0.2
        and config.get("pretrain_epochs") == 3
        and config.get("mlp_hidden_dim") == 128
        and config.get("mlp_num_layers") == 1
        and config.get("mlp_dropout") == 0.3
        and config.get("mlp_lr") == 3e-4
        and config.get("mlp_weight_decay") == 1e-3
        and config.get("mlp_class_weight") == "balanced"
        and config.get("mlp_epochs") == 40
        and config.get("mlp_early_stop_patience") == 8
        and config.get("batch_size") == 16
        and bool(config.get("feature_cache_dir"))
    )


def load_result(task_dir, dataset, condition, outer_seed):
    matches = []
    for args_path in sorted(task_dir.glob("*/args.json")):
        result_path = args_path.parent / "train_val.csv"
        audit_path = args_path.parent / "splits" / f"{dataset}_subject_split.csv"
        if not result_path.is_file() or not audit_path.is_file():
            continue
        config = json.loads(args_path.read_text())
        if not config_matches(config, dataset, condition, outer_seed):
            continue
        with result_path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("dataset") == dataset]
        if len(rows) == 1:
            matches.append((args_path, audit_path, rows[0]))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one result for seed={outer_seed}/{dataset}/{condition}, "
            f"found {len(matches)}"
        )
    args_path, audit_path, result = matches[0]
    row = {
        "dataset": dataset,
        "condition": condition,
        "outer_seed": outer_seed,
        "split_seed": 42,
        "split_audit_sha256": sha256(audit_path),
        "args_path": str(args_path),
    }
    row.update({metric: float(result[metric]) for metric in METRICS})
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def aggregate(records, outer_seeds):
    rows = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            group = [
                records[(seed, dataset, condition)]
                for seed in outer_seeds
            ]
            row = {
                "dataset": dataset,
                "condition": condition,
                "outer_seeds": "|".join(map(str, outer_seeds)),
                "split_seed": 42,
                "n": len(group),
            }
            for metric in TEST_METRICS:
                values = [record[metric] for record in group]
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.stdev(values)
            rows.append(row)
    return rows


def subset(rows, conditions):
    wanted = set(conditions)
    return [row for row in rows if row["condition"] in wanted]


def main():
    args = parse_args()
    if len(args.outer_seeds) < 2 or len(set(args.outer_seeds)) != len(args.outer_seeds):
        raise ValueError("Provide at least two unique outer seeds")
    if args.split_reference.name != "split_reference_seed42.csv":
        raise ValueError(f"Expected the fixed seed-42 split reference: {args.split_reference}")
    if not args.split_reference.is_file():
        raise FileNotFoundError(args.split_reference)

    wait_for_workers(args.status_dir, args.workers, args.poll_seconds)
    records = {
        (seed, dataset, condition): load_result(
            args.result_root / f"seed_{seed}" / dataset / condition,
            dataset,
            condition,
            seed,
        )
        for seed in args.outer_seeds
        for dataset in DATASETS
        for condition in CONDITIONS
    }

    audit_hashes = {}
    for dataset in DATASETS:
        hashes = {
            records[(seed, dataset, condition)]["split_audit_sha256"]
            for seed in args.outer_seeds
            for condition in CONDITIONS
        }
        if len(hashes) != 1:
            raise ValueError(f"Split audit changed across runs for {dataset}: {hashes}")
        audit_hashes[dataset] = next(iter(hashes))

    raw_rows = [
        records[(seed, dataset, condition)]
        for seed in args.outer_seeds
        for dataset in DATASETS
        for condition in CONDITIONS
    ]
    summary_rows = aggregate(records, args.outer_seeds)
    write_csv(args.result_root / "all_runs.csv", raw_rows)
    write_csv(args.result_root / "condition_mean_std.csv", summary_rows)
    write_csv(
        args.result_root / "activity_graph_mean_std.csv",
        subset(summary_rows, REPRESENTATION_CONDITIONS),
    )
    write_csv(
        args.result_root / "multimodal_feature_mean_std.csv",
        subset(summary_rows, FEATURE_CONDITIONS),
    )
    write_csv(
        args.result_root / "fusion_mean_std.csv",
        subset(summary_rows, FUSION_CONDITIONS),
    )
    protocol = {
        "outer_seeds": args.outer_seeds,
        "split_seed": 42,
        "split_reference": str(args.split_reference),
        "split_reference_sha256": sha256(args.split_reference),
        "split_audit_sha256_by_dataset": audit_hashes,
        "datasets": list(DATASETS),
        "conditions": list(CONDITIONS),
        "runs": len(raw_rows),
        "rng_reset_after_feature_loading": True,
        "std": "sample standard deviation (ddof=1)",
    }
    (args.result_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    print(f"Fixed-split audit passed for all {len(raw_rows)} runs")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        raise
