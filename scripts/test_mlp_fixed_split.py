#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mlp_classifier import train_mlp_classifier  # noqa: E402


class DummyVisionModel(nn.Module):
    image_mode = "activity_graph"

    def forward(self, batch):
        return batch.mean(dim=-1)


def make_loader(samples, batch_size=4):
    return DataLoader(
        TensorDataset(torch.as_tensor(samples, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def run_training(cache_dir):
    generator = np.random.default_rng(2022)
    train_samples = generator.normal(size=(12, 3, 8))
    validation_samples = generator.normal(size=(4, 3, 8))
    test_samples = generator.normal(size=(4, 3, 8))
    train_labels = np.asarray([0, 1] * 6, dtype=np.int64)
    validation_labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    test_labels = np.asarray([1, 0, 1, 0], dtype=np.int64)

    return train_mlp_classifier(
        train_loader=make_loader(train_samples),
        train_labels=train_labels,
        val_loader=make_loader(validation_samples),
        val_labels=validation_labels,
        test_loader=make_loader(test_samples),
        test_labels=test_labels,
        channels=3,
        device="cpu",
        batch_size=4,
        random_seed=2022,
        val_ratio=0.25,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        lr=1e-3,
        weight_decay=0.0,
        class_weight="balanced",
        epochs=2,
        early_stop_patience=1,
        modal_interaction="concat",
        fusion_dim=8,
        fusion_heads=2,
        cross_attn_query="ts",
        mask_prob=0.2,
        pretrain_epochs=0,
        vision_model_1=DummyVisionModel(),
        feature_cache_dir=cache_dir,
        feature_cache_signature="fixed-split-test",
    )


def main():
    with tempfile.TemporaryDirectory(prefix="neurosigvit_mlp_fixed_split_") as cache_dir:
        first = run_training(cache_dir)
        second = run_training(cache_dir)

        for result in (first, second):
            val_metrics, test_metrics, train_indices, val_indices = result
            assert train_indices == list(range(12))
            assert val_indices == list(range(4))
            assert "macro_f1" in val_metrics
            assert "macro_f1" in test_metrics

        for split in ("train", "vali", "test"):
            assert (Path(cache_dir) / f"{split}.npz").is_file()

    print("MLP FIXED-SPLIT AND CACHE VALIDATION PASSED")


if __name__ == "__main__":
    main()
