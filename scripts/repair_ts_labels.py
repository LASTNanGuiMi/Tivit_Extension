#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def parse_header(lines):
    header = {}
    data_start = None
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("@data"):
            data_start = idx
            break
        if lower.startswith("@"):
            parts = stripped.split()
            header[parts[0].lower()] = parts[1:]
    return header, data_start


def repair_line(line, class_labels):
    stripped = line.strip()
    if not stripped or ":" in stripped:
        return line, False, None

    parts = stripped.split(",")
    if len(parts) < 2:
        return line, False, "not enough comma-separated tokens"

    label = parts[-1].strip()
    if label not in class_labels:
        return line, False, f"last token {label!r} is not one of {sorted(class_labels)}"

    repaired = ",".join(parts[:-1]) + ":" + label
    return repaired, True, None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repair univariate .ts files where the class label is separated by a "
            "comma instead of the required ':' separator."
        )
    )
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path. If omitted, writes <input>.fixed.ts unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input file after writing a backup.",
    )
    parser.add_argument("--backup-suffix", default=".bak")
    args = parser.parse_args()

    lines = args.path.read_text(encoding="utf-8", errors="replace").splitlines()
    header, data_start = parse_header(lines)
    if data_start is None:
        raise SystemExit(f"No @data marker found in {args.path}")

    class_info = header.get("@classlabel", [])
    if not class_info or class_info[0].lower() != "true" or len(class_info) < 2:
        raise SystemExit("No declared class labels found in header.")
    class_labels = set(class_info[1:])

    repaired_lines = []
    repaired_count = 0
    failures = []

    for line_no, line in enumerate(lines, start=1):
        if line_no <= data_start:
            repaired_lines.append(line)
            continue

        repaired, changed, error = repair_line(line, class_labels)
        repaired_lines.append(repaired)
        if changed:
            repaired_count += 1
        elif error:
            failures.append((line_no, error, line[:120]))

    if failures:
        print("could not repair all malformed lines:")
        for line_no, error, sample in failures[:20]:
            print(f"line {line_no}: {error}")
            print(f"  {sample}")
        raise SystemExit(1)

    if args.in_place:
        backup_path = args.path.with_name(args.path.name + args.backup_suffix)
        shutil.copy2(args.path, backup_path)
        output_path = args.path
        print(f"backup written to {backup_path}")
    else:
        output_path = args.output or args.path.with_suffix(args.path.suffix + ".fixed.ts")

    output_path.write_text("\n".join(repaired_lines) + "\n", encoding="utf-8")
    print(f"repaired lines: {repaired_count}")
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
