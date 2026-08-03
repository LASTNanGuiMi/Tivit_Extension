"""Helpers for keeping machine-specific paths out of exported artifacts."""

from __future__ import annotations

import os
import re
from argparse import Namespace
from typing import Any


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _looks_like_absolute_path(value: str) -> bool:
    return os.path.isabs(value) or bool(_WINDOWS_ABSOLUTE_PATH.match(value))


def anonymize_runtime_value(value: Any) -> Any:
    """Recursively replace absolute paths while retaining a useful basename."""

    if isinstance(value, str):
        if not _looks_like_absolute_path(value):
            return value
        normalized = value.rstrip("/\\")
        basename = re.split(r"[\\/]", normalized)[-1] if normalized else ""
        return f"<ABSOLUTE_PATH>/{basename}" if basename else "<ABSOLUTE_PATH>"
    if isinstance(value, dict):
        return {key: anonymize_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [anonymize_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(anonymize_runtime_value(item) for item in value)
    return value


def anonymize_runtime_arguments(args: Namespace) -> dict[str, Any]:
    """Return a JSON-safe argument dictionary without machine-specific paths."""

    return {
        key: anonymize_runtime_value(value)
        for key, value in vars(args).items()
    }
