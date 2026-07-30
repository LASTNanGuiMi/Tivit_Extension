#!/usr/bin/env python3
"""Aggregate completed Medformer AAAI27 jobs into one CSV."""

import argparse
import csv
import json
from pathlib import Path


ORIGINAL_SEQUENCE_LENGTHS = {
    "PADS_09_task06_DrinkGlas": 976,
    "Shimmer_11_session11_DRINK": 4096,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.result_root.glob("**/metrics.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row.setdefault("label_mapping", "0_vs_1_or_2")
        row.setdefault(
            "original_seq_len", ORIGINAL_SEQUENCE_LENGTHS.get(row.get("dataset"))
        )
        rows.append(row)
    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("model", "")))

    output = args.output or args.result_root / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "dataset",
        "model",
        "label_mapping",
        "seed",
        "original_seq_len",
        "target_seq_len",
        "class_weight",
        "val_loss",
        "test_loss",
        "val_Macro-F1",
        "val_Macro-Recall",
        "val_Macro-AUPRC",
        "val_Macro-Precision",
        "val_Macro-AUROC",
        "val_Accuracy",
        "test_Macro-F1",
        "test_Macro-Recall",
        "test_Macro-AUPRC",
        "test_Macro-Precision",
        "test_Macro-AUROC",
        "test_Accuracy",
    ]
    fieldnames = [name for name in preferred if any(name in row for row in rows)]
    fieldnames.extend(
        sorted({key for row in rows for key in row if key not in fieldnames})
    )
    if not fieldnames:
        fieldnames = ["dataset", "model", "seed", "target_seq_len"]

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
