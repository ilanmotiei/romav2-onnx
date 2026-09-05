from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - CuPy ships in the Triton py3 image; NumPy path covers its absence.
    cp = None


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


def normalized_grid_cp(height: int, width: int) -> "cp.ndarray":
    xs = cp.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, dtype=cp.float32)
    ys = cp.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height, dtype=cp.float32)
    grid_y, grid_x = cp.meshgrid(ys, xs, indexing="ij")
    return cp.stack((grid_x, grid_y), axis=-1)


def _grid_sample_bhwc_cp(values: "cp.ndarray", grid: "cp.ndarray") -> "cp.ndarray":
    """Bilinear grid_sample for BHWC values, align_corners=False (mirrors the NumPy path)."""
    _, height, width, _ = values.shape
    gx, gy = grid[0, ..., 0], grid[0, ..., 1]
    x = ((gx + 1.0) * width - 1.0) / 2.0
    y = ((gy + 1.0) * height - 1.0) / 2.0

    x0 = cp.floor(x).astype(cp.int64)
    y0 = cp.floor(y).astype(cp.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = x - x0, y - y0

    def gather(ix, iy):
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        out = values[0, cp.clip(iy, 0, height - 1), cp.clip(ix, 0, width - 1)]
        return out * valid[..., None]

    return (
        gather(x0, y0) * ((1.0 - wx) * (1.0 - wy))[..., None]
        + gather(x1, y0) * (wx * (1.0 - wy))[..., None]
        + gather(x0, y1) * ((1.0 - wx) * wy)[..., None]
        + gather(x1, y1) * (wx * wy)[..., None]
    )[None].astype(cp.float32, copy=False)


# Cap on the elements of the KDE distance tile. The pairwise matrix is N x N with
# N = expansion_factor * num_corresp, so materializing it whole is 6 GiB at
# num_corresp=5000 and 25 GiB at 10000. Tiling by rows bounds it to ~256 MiB and
# makes the cost linear in N for a fixed tile.
_KDE_TILE_ELEMS = 64 << 20


def kde_cp(matches: "cp.ndarray", std: float = 0.1) -> "cp.ndarray":
    """Row-tiled KDE. Same result as the NumPy kde(), but the N x N distance matrix is
    never materialized: each row block is a GEMM (|a|^2 + |b|^2 - 2ab) reduced in place."""
    n = matches.shape[0]
    if n == 0:
        return cp.zeros((0,), cp.float32)
    sq_norm = (matches * matches).sum(axis=1)
    inv = cp.float32(-1.0 / (2.0 * std * std))
    tile = max(1, min(n, int(_KDE_TILE_ELEMS // n)))
    density = cp.empty((n,), cp.float32)
    for s in range(0, n, tile):
        e = min(s + tile, n)
        d = sq_norm[s:e, None] + sq_norm[None, :] - 2.0 * (matches[s:e] @ matches.T)
        cp.maximum(d, 0.0, out=d)  # kill negative round-off before exp
        density[s:e] = cp.exp(d * inv).sum(axis=1)
    return density


def _weighted_sample_cp(weights: "cp.ndarray", count: int, *, rs) -> "cp.ndarray":
    """Weighted sample WITHOUT replacement, via Gumbel-top-k.

    key_i = log(w_i) + Gumbel_i, take the top k. This is distributionally identical to
    sequential proportional-to-remaining-weight draws (i.e. torch.multinomial(
    replacement=False) / rng.choice(replace=False, p=...)), but it is one vectorized
    pass instead of k dependent ones -- the only formulation that is actually fast on GPU.
    """
    weights = weights.astype(cp.float32, copy=False).reshape(-1)
    if count <= 0 or weights.size == 0:
        return cp.empty((0,), cp.int64)
    count = min(count, int(weights.size))

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = cp.ones_like(weights)

    u = rs.random_sample(size=int(weights.size), dtype=cp.float32)
    u = cp.clip(u, 1e-20, 1.0 - 1e-7)
    gumbel = -cp.log(-cp.log(u))
    # log(0) -> -inf would make zero-weight ties unbreakable; a large finite floor keeps
    # the Gumbel noise meaningful, so zero-weight entries are drawn uniformly at random
    # among themselves -- matching the NumPy path's "pad with random zeros" branch.
    log_w = cp.where(weights > 0, cp.log(cp.maximum(weights, 1e-38)), cp.float32(-1e30))
    keys = log_w + gumbel

    if count == weights.size:
        idx = cp.argsort(-keys)
    else:
        part = cp.argpartition(-keys, count - 1)[:count]
        idx = part[cp.argsort(-keys[part])]
    return idx.astype(cp.int64)


def sample_roma_outputs_cupy(
    *,
    warp_ab: "cp.ndarray",
    overlap_ab: "cp.ndarray",
    precision_ab: "cp.ndarray",
    warp_ba: "cp.ndarray | None",
    overlap_ba: "cp.ndarray | None",
    precision_ba: "cp.ndarray | None",
    num_corresp: int,
    seed: int = -1,
) -> tuple["cp.ndarray", "cp.ndarray", "cp.ndarray", "cp.ndarray"]:
    """CuPy implementation of RoMaV2.sample() for the Triton Python backend.

    Keeps the dense tensors on CUDA when Triton hands over GPU buffers via DLPack
    (requires FORCE_CPU_ONLY_INPUT_TENSORS=no on a KIND_GPU instance). Mirrors the
    NumPy path's data flow exactly.
    """
    if cp is None:
        raise RuntimeError("CuPy is required for sample_roma_outputs_cupy")
    if num_corresp < 0:
        raise ValueError("num_corresp must be non-negative")

    rs = cp.random.RandomState(None if seed < 0 else int(seed))

    warp = warp_ab[0].astype(cp.float32, copy=False)
    confidence_ab = overlap_ab[0].reshape(-1).astype(cp.float32, copy=False)
    precision_ab_0 = precision_ab[0].astype(cp.float32, copy=False)

    height, width, _ = warp.shape
    grid = normalized_grid_cp(height, width)
    matches_ab = cp.concatenate((grid, warp), axis=-1).reshape(-1, 4)

    has_ba = warp_ba is not None and overlap_ba is not None and precision_ba is not None
    if has_ba:
        warp_ba_0 = warp_ba[0].astype(cp.float32, copy=False)
        confidence_ba = overlap_ba[0].reshape(-1).astype(cp.float32, copy=False)
        precision_ba_0 = precision_ba[0].astype(cp.float32, copy=False)

        precision_a = _grid_sample_bhwc_cp(
            precision_ba_0.reshape(1, height, width, -1), warp[None]
        ).reshape(height, width, 2, 2)
        precision_b = _grid_sample_bhwc_cp(
            precision_ab_0.reshape(1, height, width, -1), warp_ba_0[None]
        ).reshape(height, width, 2, 2)

        precision_fwd = cp.stack((precision_a, precision_ab_0), axis=-3).reshape(-1, 2, 2, 2)
        precision_bwd = cp.stack((precision_ba_0, precision_b), axis=-3).reshape(-1, 2, 2, 2)
        precision = cp.concatenate((precision_fwd, precision_bwd), axis=0)

        matches_ba = cp.concatenate((warp_ba_0, grid), axis=-1).reshape(-1, 4)
        confidence = cp.concatenate((confidence_ab, confidence_ba), axis=0)
        matches = cp.concatenate((matches_ab, matches_ba), axis=0)
    else:
        precision = cp.broadcast_to(
            precision_ab_0.reshape(-1, 1, 2, 2), (matches_ab.shape[0], 2, 2, 2)
        ).copy()
        confidence = confidence_ab
        matches = matches_ab

    in_frame = (cp.abs(matches).max(axis=-1) <= (1.0 - 1.0 / height)).astype(cp.float32)
    confidence = confidence * in_frame

    expansion_factor = 4
    first_count = min(expansion_factor * num_corresp, int(confidence.shape[0]))
    corresp_inds = _weighted_sample_cp(confidence, first_count, rs=rs)

    sampled_matches = matches[corresp_inds]
    sampled_confidence = confidence[corresp_inds]
    sampled_precision = precision[corresp_inds]

    density = kde_cp(sampled_matches)
    probabilities = 1.0 / (density + 1.0)
    probabilities[density < 10.0] = 1e-7

    final_count = min(num_corresp, int(sampled_confidence.shape[0]))
    balanced = _weighted_sample_cp(probabilities, final_count, rs=rs)

    final_precision = sampled_precision[balanced]
    return (
        cp.ascontiguousarray(sampled_matches[balanced]),
        cp.ascontiguousarray(sampled_confidence[balanced]),
        cp.ascontiguousarray(final_precision[:, 0]),
        cp.ascontiguousarray(final_precision[:, 1]),
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
