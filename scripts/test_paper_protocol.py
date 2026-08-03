#!/usr/bin/env python3
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.utils\._pytree\._register_pytree_node` is deprecated\..*",
    category=FutureWarning,
    module=r"transformers\.utils\.generic",
)

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datautils import (  # noqa: E402
    _interpolate_falltl_sequence,
    _split_uci_har_train_subjects,
    _standardize_and_pad_falltl,
)
from src.mlp_classifier import _forward_mantis_batch  # noqa: E402
from src.neurosigvit import NeuroSigViT_OpenCLIP  # noqa: E402


class DummyProcessor:
    transforms = [nn.Identity()]


class DummyTransformer:
    def __init__(self):
        self.resblocks = []


class DummyOpenCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = DummyTransformer()
        self.ln_post = nn.Identity()
        projection = torch.zeros(1280, 1024)
        projection[:1024, :1024] = torch.eye(1024)
        self.proj = nn.Parameter(projection, requires_grad=False)


class DummyMantis(nn.Module):
    def forward(self, inputs):
        scale = inputs.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        basis = torch.arange(1, 513, device=inputs.device, dtype=inputs.dtype)
        return scale * basis.unsqueeze(0)


def check_visual_pooling_and_projection():
    model = NeuroSigViT_OpenCLIP(
        processor=DummyProcessor(),
        vit=DummyOpenCLIP(),
        layer_idx=-1,
        aggregation="mean",
        patch_size=None,
        stride=None,
        image_mode="activity_graph",
    )
    hidden = torch.ones(2, 5, 1280)
    hidden[:, 0, :] = 1000.0
    pooled = model.aggregate_hidden_representations(hidden, "mean")
    assert pooled.shape == (2, 1024)
    torch.testing.assert_close(pooled, torch.ones_like(pooled))


def check_mantis_channel_pooling():
    batch = torch.stack(
        [
            torch.stack([torch.ones(32), torch.full((32,), 3.0)]),
            torch.stack([torch.full((32,), 2.0), torch.full((32,), 4.0)]),
        ]
    )
    output = _forward_mantis_batch(
        DummyMantis(), batch, channels=2, device="cpu"
    )
    assert output.shape == (2, 512)
    torch.testing.assert_close(
        torch.linalg.vector_norm(output, dim=1), torch.ones(2), atol=1e-6, rtol=1e-6
    )


def check_falltl_preprocessing():
    first = np.asarray(
        [[1.0, 2.0], [np.nan, 4.0], [3.0, 6.0]], dtype=np.float32
    )
    second = np.asarray(
        [[2.0, 1.0], [4.0, np.nan], [6.0, 5.0], [8.0, 7.0]],
        dtype=np.float32,
    )
    first = _interpolate_falltl_sequence(first, "first.csv")
    second = _interpolate_falltl_sequence(second, "second.csv")
    assert first[1, 0] == 2.0
    assert second[1, 1] == 3.0

    train, validation, test = _standardize_and_pad_falltl(
        [first, second], [first], [second]
    )
    assert train.shape == (2, 2, 4)
    assert validation.shape == (1, 2, 3)
    assert test.shape == (1, 2, 4)
    assert np.isfinite(train).all()
    np.testing.assert_allclose(train[0, :, 3], 0.0)


def check_uci_subject_split():
    sample_subjects = np.repeat(np.arange(1, 22), 3)
    train_subjects, validation_subjects = _split_uci_har_train_subjects(
        sample_subjects, val_ratio=0.25, split_seed=42
    )
    assert len(train_subjects) == 16
    assert len(validation_subjects) == 5
    assert not set(train_subjects) & set(validation_subjects)
    assert set(train_subjects) | set(validation_subjects) == set(range(1, 22))


def main():
    check_visual_pooling_and_projection()
    check_mantis_channel_pooling()
    check_falltl_preprocessing()
    check_uci_subject_split()
    print("PAPER PROTOCOL CHECKS PASSED")


if __name__ == "__main__":
    main()
