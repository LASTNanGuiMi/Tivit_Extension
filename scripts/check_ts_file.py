#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_header(lines):
    header = {}
    data_start = None
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("@data"):
            data_start = idx
            break
        if lower.startswith("@"):
            parts = stripped.split()
            header[parts[0].lower()] = parts[1:]
    return header, data_start


def main():
    parser = argparse.ArgumentParser(description="Check aeon/sktime .ts file format.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--show", type=int, default=20, help="Max issues to print")
    args = parser.parse_args()

    lines = args.path.read_text(encoding="utf-8", errors="replace").splitlines()
    header, data_start = parse_header(lines)
    if data_start is None:
        raise SystemExit(f"No @data marker found in {args.path}")

    class_info = header.get("@classlabel", [])
    has_class_label = bool(class_info and class_info[0].lower() == "true")
    class_labels = set(class_info[1:]) if has_class_label else set()

    print(f"file: {args.path}")
    print(f"data starts at line: {data_start}")
    print(f"class labels: {sorted(class_labels) if class_labels else 'none declared'}")
    for key in ("@dimensions", "@equallength", "@serieslength"):
        if key in header:
            print(f"{key}: {' '.join(header[key])}")

    issues = []
    total_cases = 0
    colon_hist = {}
    comma_label_candidates = 0

    for line_no, line in enumerate(lines[data_start:], start=data_start + 1):
        stripped = line.strip()
        if not stripped:
            issues.append((line_no, "empty data line", ""))
            continue

        total_cases += 1
        colon_count = stripped.count(":")
        colon_hist[colon_count] = colon_hist.get(colon_count, 0) + 1

        if has_class_label and colon_count == 0:
            parts = stripped.split(",")
            last = parts[-1].strip() if parts else ""
            if last in class_labels:
                comma_label_candidates += 1
                issues.append(
                    (
                        line_no,
                        "missing ':' before class label; last comma token looks like a label",
                        stripped[:120],
                    )
                )
            else:
                issues.append(
                    (
                        line_no,
                        "missing ':' before class label; cannot infer label",
                        stripped[:120],
                    )
                )
        elif has_class_label and colon_count < 1:
            issues.append((line_no, "no class-label separator ':'", stripped[:120]))

    print(f"data cases: {total_cases}")
    print(f"colon count histogram: {colon_hist}")
    print(f"comma-label repair candidates: {comma_label_candidates}")

    if issues:
        print("\nfirst issues:")
        for line_no, message, sample in issues[: args.show]:
            print(f"line {line_no}: {message}")
            if sample:
                print(f"  {sample}")
        raise SystemExit(1)

    print("no obvious .ts format issues found")


if __name__ == "__main__":
    main()
