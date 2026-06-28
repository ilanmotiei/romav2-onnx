from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - Triton image is expected to include torch.
    torch = None
    F = None


def normalized_grid(height: int, width: int) -> np.ndarray:
    xs = np.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, dtype=np.float32)
    ys = np.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack((grid_x, grid_y), axis=-1)


def _weighted_sample(
    weights: np.ndarray,
    count: int,
    *,
    replace: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if count <= 0 or weights.size == 0:
        return np.empty((0,), dtype=np.int64)
    if not replace:
        count = min(count, weights.size)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        weights = np.ones_like(weights, dtype=np.float64)
        total = weights.sum()
    if not replace and np.count_nonzero(weights > 0) < count:
        positive = np.flatnonzero(weights > 0)
        zero = np.flatnonzero(weights <= 0)
        positive_probs = weights[positive] / weights[positive].sum()
        sampled_positive = rng.choice(
            positive,
            size=positive.size,
            replace=False,
            p=positive_probs,
        )
        sampled_zero = rng.choice(
            zero,
            size=count - positive.size,
            replace=False,
        )
        return np.concatenate((sampled_positive, sampled_zero)).astype(np.int64)
    probabilities = weights / total
    return rng.choice(weights.size, size=count, replace=replace, p=probabilities).astype(np.int64)


def _grid_sample_bhwc(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Bilinear grid_sample for BHWC values with align_corners=False."""
    batch, height, width, channels = values.shape
    if batch != 1:
        raise ValueError("RoMaV2 sample() only consumes the first batch item")

    gx = grid[0, ..., 0]
    gy = grid[0, ..., 1]
    x = ((gx + 1.0) * width - 1.0) / 2.0
    y = ((gy + 1.0) * height - 1.0) / 2.0

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0
    wy = y - y0

    def gather(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        clipped_x = np.clip(ix, 0, width - 1)
        clipped_y = np.clip(iy, 0, height - 1)
        out = values[0, clipped_y, clipped_x]
        return out * valid[..., None]

    top_left = gather(x0, y0)
    top_right = gather(x1, y0)
    bottom_left = gather(x0, y1)
    bottom_right = gather(x1, y1)

    return (
        top_left * ((1.0 - wx) * (1.0 - wy))[..., None]
        + top_right * (wx * (1.0 - wy))[..., None]
        + bottom_left * ((1.0 - wx) * wy)[..., None]
        + bottom_right * (wx * wy)[..., None]
    )[None].astype(values.dtype, copy=False)


def kde(matches: np.ndarray, std: float = 0.1) -> np.ndarray:
    matches = matches.astype(np.float32, copy=False)
    diff = matches[:, None, :] - matches[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    return np.exp(-sq_dist / (2.0 * std * std)).sum(axis=-1).astype(np.float32)


def normalized_grid_torch(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(
        -1.0 + 1.0 / width,
        1.0 - 1.0 / width,
        width,
        dtype=torch.float32,
        device=device,
    )
    ys = torch.linspace(
        -1.0 + 1.0 / height,
        1.0 - 1.0 / height,
        height,
        dtype=torch.float32,
        device=device,
    )
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def bhwc_grid_sample_torch(
    values: torch.Tensor,
    grid: torch.Tensor,
    *,
    mode: str = "bilinear",
    align_corners: bool | None = False,
) -> torch.Tensor:
    return F.grid_sample(
        values.permute(0, 3, 1, 2),
        grid,
        mode=mode,
        align_corners=align_corners,
    ).permute(0, 2, 3, 1)


def kde_torch(matches: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    return (-(torch.cdist(matches, matches) ** 2) / (2 * std**2)).exp().sum(dim=-1)


def _torch_multinomial(
    weights: torch.Tensor,
    count: int,
    *,
    replacement: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if count <= 0 or weights.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=weights.device)
    if not replacement:
        count = min(count, weights.numel())
    return torch.multinomial(weights, count, replacement=replacement, generator=generator)


def sample_roma_outputs_torch(
    *,
    warp_ab: torch.Tensor,
    overlap_ab: torch.Tensor,
    precision_ab: torch.Tensor,
    warp_ba: torch.Tensor | None,
    overlap_ba: torch.Tensor | None,
    precision_ba: torch.Tensor | None,
    num_corresp: int,
    seed: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch implementation of RoMaV2.sample() for Triton Python backend.

    This keeps the dense tensors on CUDA when Triton passes GPU buffers through
    DLPack. It intentionally mirrors the original RoMaV2.sample() data flow.
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for sample_roma_outputs_torch")
    if num_corresp < 0:
        raise ValueError("num_corresp must be non-negative")
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=warp_ab.device)
        generator.manual_seed(seed)

    warp = warp_ab[0].float()
    confidence_ab = overlap_ab[0].reshape(-1).float()
    precision_ab_0 = precision_ab[0].float()

    height, width, _ = warp.shape
    grid = normalized_grid_torch(height, width, device=warp.device)
    matches_ab = torch.cat((grid, warp), dim=-1).reshape(-1, 4)

    has_ba = warp_ba is not None and overlap_ba is not None and precision_ba is not None
    if has_ba:
        warp_ba_0 = warp_ba[0].float()
        confidence_ba = overlap_ba[0].reshape(-1).float()
        precision_ba_0 = precision_ba[0].float()

        precision_a = bhwc_grid_sample_torch(
            precision_ba_0[None].reshape(1, height, width, -1),
            warp[None],
            mode="bilinear",
            align_corners=False,
        ).reshape(height, width, 2, 2)
        precision_b = bhwc_grid_sample_torch(
            precision_ab_0[None].reshape(1, height, width, -1),
            warp_ba_0[None],
            mode="bilinear",
            align_corners=False,
        ).reshape(height, width, 2, 2)
        precision_fwd = torch.stack((precision_a, precision_ab_0), dim=-3).reshape(
            -1, 2, 2, 2
        )
        precision_bwd = torch.stack((precision_ba_0, precision_b), dim=-3).reshape(
            -1, 2, 2, 2
        )
        precision = torch.cat((precision_fwd, precision_bwd), dim=0)

        matches_ba = torch.cat((warp_ba_0, grid), dim=-1).reshape(-1, 4)
        confidence = torch.cat((confidence_ab, confidence_ba), dim=0)
        matches = torch.cat((matches_ab, matches_ba), dim=0)
    else:
        precision = precision_ab_0.reshape(-1, 2, 2)
        confidence = confidence_ab.reshape(-1)
        matches = matches_ab

    confidence = confidence * matches.abs().amax(dim=-1).le(1 - 1 / height).float()

    expansion_factor = 4
    first_count = min(expansion_factor * num_corresp, confidence.numel())
    corresp_inds = _torch_multinomial(
        confidence,
        first_count,
        replacement=False,
        generator=generator,
    )

    sampled_matches = matches[corresp_inds]
    sampled_confidence = confidence[corresp_inds]
    sampled_precision = precision[corresp_inds]

    density = kde_torch(sampled_matches)
    p = 1 / (density + 1)
    p[density < 10] = 1e-7
    final_count = min(num_corresp, sampled_confidence.numel())
    balanced = _torch_multinomial(
        p,
        final_count,
        replacement=False,
        generator=generator,
    )

    final_precision = sampled_precision[balanced]
    if has_ba:
        precision_a_out = final_precision[:, 0]
        precision_b_out = final_precision[:, 1]
    else:
        precision_a_out = final_precision[:, 0]
        precision_b_out = final_precision[:, 1]
    return (
        sampled_matches[balanced].contiguous(),
        sampled_confidence[balanced].contiguous(),
        precision_a_out.contiguous(),
        precision_b_out.contiguous(),
    )


def sample_roma_outputs(
    *,
    warp_ab: np.ndarray,
    overlap_ab: np.ndarray,
    precision_ab: np.ndarray,
    warp_ba: np.ndarray | None,
    overlap_ba: np.ndarray | None,
    precision_ba: np.ndarray | None,
    num_corresp: int,
    seed: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample dense RoMaV2 outputs with the same data flow as RoMaV2.sample().

    The original implementation samples only the first batch item; this function
    intentionally preserves that behavior.
    """
    if num_corresp < 0:
        raise ValueError("num_corresp must be non-negative")

    rng = np.random.default_rng(None if seed < 0 else seed)

    warp_ab = warp_ab[0].astype(np.float32, copy=False)
    overlap_ab = overlap_ab[0].reshape(-1).astype(np.float32, copy=False)
    precision_ab = precision_ab[0].astype(np.float32, copy=False)

    height, width, _ = warp_ab.shape
    grid = normalized_grid(height, width)
    matches_ab = np.concatenate((grid, warp_ab), axis=-1).reshape(-1, 4)

    has_ba = warp_ba is not None and overlap_ba is not None and precision_ba is not None
    if has_ba:
        warp_ba_0 = warp_ba[0].astype(np.float32, copy=False)
        overlap_ba_0 = overlap_ba[0].reshape(-1).astype(np.float32, copy=False)
        precision_ba_0 = precision_ba[0].astype(np.float32, copy=False)

        precision_a = _grid_sample_bhwc(
            precision_ba_0.reshape(1, height, width, -1),
            warp_ab[None],
        ).reshape(height, width, 2, 2)
        precision_b = _grid_sample_bhwc(
            precision_ab.reshape(1, height, width, -1),
            warp_ba_0[None],
        ).reshape(height, width, 2, 2)
        precision_fwd = np.stack((precision_a, precision_ab), axis=-3).reshape(-1, 2, 2, 2)
        precision_bwd = np.stack((precision_ba_0, precision_b), axis=-3).reshape(-1, 2, 2, 2)
        precision = np.concatenate((precision_fwd, precision_bwd), axis=0)

        matches_ba = np.concatenate((warp_ba_0, grid), axis=-1).reshape(-1, 4)
        confidence = np.concatenate((overlap_ab, overlap_ba_0), axis=0)
        matches = np.concatenate((matches_ab, matches_ba), axis=0)
    else:
        precision = precision_ab.reshape(-1, 2, 2)[:, None]
        confidence = overlap_ab
        matches = matches_ab

    in_frame = (np.max(np.abs(matches), axis=-1) <= (1.0 - 1.0 / height)).astype(np.float32)
    confidence = confidence * in_frame

    expansion_factor = 4
    first_count = min(expansion_factor * num_corresp, confidence.shape[0])
    corresp_inds = _weighted_sample(confidence, first_count, replace=False, rng=rng)

    sampled_matches = matches[corresp_inds]
    sampled_confidence = confidence[corresp_inds]
    sampled_precision = precision[corresp_inds]

    density = kde(sampled_matches)
    probabilities = 1.0 / (density + 1.0)
    probabilities[density < 10.0] = 1e-7

    final_count = min(num_corresp, sampled_confidence.shape[0])
    balanced = _weighted_sample(probabilities, final_count, replace=False, rng=rng)

    final_precision = sampled_precision[balanced]
    return (
        sampled_matches[balanced].astype(np.float32, copy=False),
        sampled_confidence[balanced].astype(np.float32, copy=False),
        final_precision[:, 0].astype(np.float32, copy=False),
        final_precision[:, 1].astype(np.float32, copy=False),
    )
