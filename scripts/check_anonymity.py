#!/usr/bin/env python3
"""Fail when submission-facing files contain common identity disclosures."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".aris",
    "__pycache__",
    "data",
    "results",
    "logs",
    "feature_cache",
    "models",
    "checkpoints",
    "anonymous_artifact",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {".gitignore", "LICENSE"}
PATTERNS = (
    ("Unix user home", re.compile(r"/(?:home|Users)/[^/\s]+/")),
    ("Windows user profile", re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+[\\/]")),
    ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("IPv4 address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
)


def iter_submission_text_files():
    checker = Path(__file__).resolve()
    for directory, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        for filename in filenames:
            path = Path(directory) / filename
            if path.resolve() == checker:
                continue
            if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
                yield path


def main():
    findings = []
    for path in iter_submission_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append((path, line_number, label, line.strip()))

    for directory, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in (*dirnames, *filenames):
            path = Path(directory) / name
            if not path.is_symlink():
                continue
            target = os.readlink(path)
            if os.path.isabs(target) or re.match(r"^[A-Za-z]:[\\/]", target):
                findings.append((path, 0, "absolute symlink target", target))

    if findings:
        for path, line_number, label, excerpt in findings:
            location = f"{path.relative_to(ROOT)}:{line_number}" if line_number else str(path.relative_to(ROOT))
            print(f"{location}: {label}: {excerpt}")
        print(f"ANONYMITY CHECK FAILED: {len(findings)} finding(s)", file=sys.stderr)
        return 1

    print("ANONYMITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
