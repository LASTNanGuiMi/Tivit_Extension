#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tivit import preprocess_graph, preprocess_multichannel_lineplot


def main():
    time = torch.linspace(0, 4 * torch.pi, 200)
    sample = torch.stack([torch.sin(time + offset) for offset in torch.linspace(0, 2, 6)])
    batch = torch.stack([sample, sample * 0.5])

    line_plot = preprocess_multichannel_lineplot(batch)
    activity_graph = preprocess_graph(batch, mode="multicolumn", render="waveform")

    assert line_plot.shape == (2, 3, 224, 224)
    assert activity_graph.shape == (2, 3, 224, 224)
    assert torch.isfinite(line_plot).all()
    assert torch.isfinite(activity_graph).all()
    assert line_plot.std() > 0
    assert activity_graph.std() > 0
    assert not torch.allclose(line_plot[:, 0], line_plot[:, 1])
    assert torch.allclose(activity_graph[:, 0], activity_graph[:, 1])
    assert not torch.allclose(line_plot, activity_graph)
    print("Ablation image-mode checks passed.")


if __name__ == "__main__":
    main()
