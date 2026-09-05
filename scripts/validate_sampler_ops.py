"""Validate NumPy sampler primitives against the RoMaV2 PyTorch primitives."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = ROOT / "triton/model_repository/romav2_sampler/1/sampler.py"


def load_sampler_module():
    spec = importlib.util.spec_from_file_location("roma_sampler", SAMPLER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import sampler from {SAMPLER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    sampler = load_sampler_module()
    rng = np.random.default_rng(0)

    values = rng.normal(size=(1, 12, 10, 4)).astype(np.float32)
    grid = rng.uniform(-1.2, 1.2, size=(1, 12, 10, 2)).astype(np.float32)
    np_sampled = sampler._grid_sample_bhwc(values, grid)
    torch_sampled = F.grid_sample(
        torch.from_numpy(values).permute(0, 3, 1, 2),
        torch.from_numpy(grid),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).numpy()
    np.testing.assert_allclose(np_sampled, torch_sampled, rtol=1e-5, atol=1e-5)

    matches = rng.uniform(-1, 1, size=(64, 4)).astype(np.float32)
    np_kde = sampler.kde(matches)
    torch_matches = torch.from_numpy(matches)
    torch_kde = (
        (-(torch.cdist(torch_matches, torch_matches) ** 2) / (2 * 0.1**2))
        .exp()
        .sum(dim=-1)
        .numpy()
    )
    np.testing.assert_allclose(np_kde, torch_kde, rtol=1e-4, atol=1e-4)
    print("Sampler primitive validation passed.")


if __name__ == "__main__":
    main()
