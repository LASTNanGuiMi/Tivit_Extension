#!/usr/bin/env python3
"""Validate and aggregate the seed-2022 FallTL binary main experiment."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def config_matches(config, fusion):
    return (
        config.get("datasets") == "falltl"
        and config.get("falltl_protocol") == "comparison_binary"
        and config.get("random_seed") == 2022
        and config.get("image_mode") == "activity_graph"
        and config.get("vit_1_layer") == 14
        and config.get("mantis") is True
        and config.get("classifier_type") == "mlp"
        and config.get("modal_interaction") == fusion
        and config.get("fusion_dim") == 512
        and config.get("fusion_heads") == 4
        and config.get("cross_attn_query") == "ts"
        and config.get("mask_prob") == 0.3
        and config.get("pretrain_epochs") == 5
        and config.get("mlp_hidden_dim") == 512
        and config.get("mlp_num_layers") == 2
        and config.get("mlp_dropout") == 0.1
        and config.get("mlp_lr") == 1e-4
        and config.get("mlp_weight_decay") == 1e-4
        and config.get("mlp_epochs") == 20
        and config.get("mlp_early_stop_patience") == 3
        and config.get("batch_size") == 32
        and bool(config.get("feature_cache_dir"))
    )


def load_result(task_dir, fusion):
    matches = []
    for args_path in sorted(task_dir.glob("*/args.json")):
        result_path = args_path.parent / "train_val.csv"
        audit_path = args_path.parent / "splits/FallTL_comparison_binary_split.csv"
        if not result_path.is_file() or not audit_path.is_file():
            continue
        config = json.loads(args_path.read_text())
        if not config_matches(config, fusion):
            continue
        with result_path.open(newline="") as handle:
            rows = [
                row for row in csv.DictReader(handle)
                if row.get("dataset") == "FallTL"
            ]
        if len(rows) == 1:
            matches.append((args_path, audit_path, rows[0]))
    if len(matches) != 1:
        raise ValueError(f"Expected one result for {fusion}, found {len(matches)}")
    args_path, audit_path, result = matches[0]
    audit_rows = list(csv.DictReader(audit_path.open(newline="")))
    split_counts = {
        split: sum(row["split"] == split for row in audit_rows)
        for split in ("train", "vali", "test")
    }
    if split_counts != {"train": 7674, "vali": 2558, "test": 2559}:
        raise ValueError(f"Unexpected split counts for {fusion}: {split_counts}")
    row = {
        "dataset": "FallTL",
        "label_mapping": "D=0,F=1",
        "fusion": fusion,
        "random_seed": 2022,
        "split_seed": 42,
        "split_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "args_path": str(args_path),
    }
    row.update({metric: float(result[metric]) for metric in METRICS})
    return row


def main():
    args = parse_args()
    rows = [load_result(args.result_root / fusion, fusion) for fusion in FUSIONS]
    hashes = {row["split_audit_sha256"] for row in rows}
    if len(hashes) != 1:
        raise ValueError(f"FallTL split changed across fusion modes: {hashes}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    protocol = {
        "dataset": "FallTL",
        "label_mapping": {"D": 0, "F": 1},
        "sequence_length": 256,
        "split_seed": 42,
        "split_counts": {"train": 7674, "vali": 2558, "test": 2559},
        "random_seed": 2022,
        "fusion_modes": list(FUSIONS),
        "split_audit_sha256": next(iter(hashes)),
    }
    (args.result_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    print(f"Wrote {len(rows)} FallTL binary results to {args.output}")


if __name__ == "__main__":
    main()
