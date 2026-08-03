#!/usr/bin/env python3
import sys
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.privacy import anonymize_runtime_arguments, anonymize_runtime_value  # noqa: E402


def main():
    unix_path = "/" + "home" + "/example_user/project/data"
    windows_path = "C:" + "\\Users\\example_user\\models\\encoder"
    assert anonymize_runtime_value(unix_path) == "<ABSOLUTE_PATH>/data"
    assert anonymize_runtime_value(windows_path) == "<ABSOLUTE_PATH>/encoder"
    assert anonymize_runtime_value("../models/encoder") == "../models/encoder"

    sanitized = anonymize_runtime_arguments(
        Namespace(
            data_dir=unix_path,
            model=windows_path,
            relative="data",
            nested=[unix_path, {"cache": windows_path}],
        )
    )
    assert sanitized["data_dir"] == "<ABSOLUTE_PATH>/data"
    assert sanitized["model"] == "<ABSOLUTE_PATH>/encoder"
    assert sanitized["relative"] == "data"
    assert sanitized["nested"][0] == "<ABSOLUTE_PATH>/data"
    assert sanitized["nested"][1]["cache"] == "<ABSOLUTE_PATH>/encoder"
    print("ANONYMITY SANITIZATION TESTS PASSED")


if __name__ == "__main__":
    main()
