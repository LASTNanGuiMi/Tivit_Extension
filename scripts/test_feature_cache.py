#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mlp_classifier import _load_feature_cache, _save_feature_cache


def assert_rejected(callback, expected_text):
    try:
        callback()
    except ValueError as exc:
        if expected_text not in str(exc):
            raise AssertionError(f"Unexpected cache error: {exc}") from exc
    else:
        raise AssertionError(f"Cache mismatch was accepted: {expected_text}")


def main():
    labels = np.asarray([0, 1, 1], dtype=np.int64)
    branch_names = ["vision_model_1", "mantis_model"]
    features = [
        torch.arange(6, dtype=torch.float32).reshape(3, 2),
        torch.arange(9, dtype=torch.float32).reshape(3, 3),
    ]
    signature = '{"dataset":"cache-test","vit_layer":14}'

    with tempfile.TemporaryDirectory(prefix="tivit_feature_cache_") as temp_dir:
        cache_path = Path(temp_dir) / "train.npz"
        _save_feature_cache(
            cache_path,
            labels,
            branch_names,
            features,
            signature,
        )
        loaded = _load_feature_cache(
            cache_path,
            labels,
            branch_names,
            signature,
        )
        for expected, actual in zip(features, loaded):
            torch.testing.assert_close(actual, expected)

        assert_rejected(
            lambda: _load_feature_cache(
                cache_path,
                labels,
                branch_names,
                "different-signature",
            ),
            "model/configuration mismatch",
        )
        assert_rejected(
            lambda: _load_feature_cache(
                cache_path,
                np.asarray([0, 0, 1], dtype=np.int64),
                branch_names,
                signature,
            ),
            "labels do not match",
        )
        assert_rejected(
            lambda: _load_feature_cache(
                cache_path,
                labels,
                ["vision_model_1"],
                signature,
            ),
            "branch mismatch",
        )

    print("FEATURE CACHE VALIDATION PASSED")


if __name__ == "__main__":
    main()
