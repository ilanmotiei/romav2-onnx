from typing import Literal
import torch
import torch.nn.functional as F
try:
    import local_corr
except ImportError:
    local_corr = None


def local_corr_wrapper(
    feature0: torch.Tensor,
    feature1: torch.Tensor,
    coords: torch.Tensor,
    local_window: torch.Tensor,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear",
    dtype=torch.float32,
):
    assert local_corr is not None
    assert padding_mode == "zeros"
    warp = (coords[..., None, :] + local_window[:, None, None]).reshape(B, h * w, K, 2)
    corr = (
        local_corr.local_corr(
            feature0.reshape(B, c, h * w).permute(0, 2, 1).float() / (c**0.5),
            feature1.permute(0, 2, 3, 1).clone().detach().float(),
            warp.clone().detach(),
            mode=sample_mode,
            normalized_coords=True,
        )
        .permute(0, 2, 1)
        .reshape(B, K, h, w)
    )
    return corr


def native_torch_local_corr(
    feature0,
    feature1,
    warp,
    local_window,
    B,
    K,
    c,
    r,
    h,
    w,
    device,
    padding_mode="zeros",
    sample_mode="bilinear",
    dtype=torch.float32,
):
    # One grid_sample per window offset.  Gathering all K offsets at once
    # (B, c, h, w*K) is what the CUDA extension avoids and what makes the
    # native path memory-hungry: the patch-4 refiner at 1280x1280 (c=192,
    # K=49, 320x320 grid) needs two 3.85 GB intermediates, which pushed the
    # precise ONNX model to ~20 GB and out of memory inside Triton on a 24 GB
    # GPU.  Per offset the largest intermediate is (B, c, h, w): 78 MB there.
    # K is static, so the loop unrolls to K GridSample nodes in the ONNX graph
    # while the batch dimension stays dynamic; the arithmetic (dot product
    # over c per offset) is unchanged.
    # warp: (B, h, w, 2), local_window: (1, K, 2)
    f0 = feature0 / (c**0.5)                            # (B, c, h, w)
    corrs = []
    for k in range(K):
        coords = warp + local_window[:, k][:, None, None, :]   # (B, h, w, 2)
        window_feature = F.grid_sample(
            feature1,
            coords,
            padding_mode=padding_mode,
            align_corners=False,
            mode=sample_mode,
        )                                               # (B, c, h, w)
        corrs.append((f0 * window_feature).sum(dim=1))  # (B, h, w)
    return torch.stack(corrs, dim=1)                    # (B, K, h, w)


def local_correlation(
    feature0: torch.Tensor,  # (B x C x H x W)
    feature1: torch.Tensor,  # (B x C x H x W)
    local_radius: int,
    warp: torch.Tensor,  # (B x H x W x 2)
    scale_factor: torch.Tensor,
    padding_mode="zeros",
    sample_mode: Literal["bilinear", "nearest"] = "bilinear",
):
    r = local_radius
    K = (2 * r + 1) ** 2
    B, c, h, w = feature0.size()
    local_h, local_w = h, w
    device = feature0.device
    dtype = feature0.dtype
    local_window = torch.meshgrid(
        [
            torch.linspace(
                -2 * local_radius / local_h,
                2 * local_radius / local_h,
                2 * r + 1,
                device=device,
            ),
            torch.linspace(
                -2 * local_radius / local_w,
                2 * local_radius / local_w,
                2 * r + 1,
                device=device,
            ),
        ],
        indexing="ij",
    )
    local_window = (
        torch.stack((local_window[1], local_window[0]), dim=-1)[None]
        .expand(1, 2 * r + 1, 2 * r + 1, 2)
        .reshape(1, K, 2)
    )
    if local_corr is None:
        corr = native_torch_local_corr(
            feature0,
            feature1,
            warp,
            local_window,
            B,
            K,
            c,
            r,
            h,
            w,
            device,
            padding_mode,
            sample_mode,
            dtype,
        )
    else:
        corr = local_corr_wrapper(
            feature0,
            feature1,
            warp,
            local_window,
            B,
            K,
            c,
            r,
            h,
            w,
            device,
            padding_mode,
            sample_mode,
            dtype,
        )
    return corr
