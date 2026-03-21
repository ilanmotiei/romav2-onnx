"""Visualize RoMaV2 matching on the toronto sample images.

Usage:
    python scripts/visualize.py                        # PyTorch model
    python scripts/visualize.py --onnx romav2_fast.onnx  # ONNX model
    python scripts/visualize.py --img-a assets/toronto_A.jpg --img-b assets/toronto_B.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # RoMaV2 dev install
sys.path.insert(0, str(Path(__file__).resolve().parent))               # scripts/ → export_onnx


# ── helpers ──────────────────────────────────────────────────────────────────

def load_image(path: str, H: int, W: int) -> tuple[np.ndarray, torch.Tensor]:
    """Load and resize image; return (HxWx3 uint8, 1x3xHxW float32 in [0,1])."""
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    arr = np.array(img)                              # H W 3  uint8
    tensor = torch.from_numpy(arr).float() / 255.0  # H W 3
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)   # 1 3 H W
    return arr, tensor


def warp_image(img_np: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Sample img_B at positions given by warp (H,W,2) in [-1,1] → warped into img_A space."""
    H, W = warp.shape[:2]
    # warp: H W 2  (x, y) normalised coords in B
    xs = (warp[..., 0] + 1) / 2 * (W - 1)   # pixel x in B
    ys = (warp[..., 1] + 1) / 2 * (H - 1)   # pixel y in B
    xs = np.clip(xs.round().astype(int), 0, W - 1)
    ys = np.clip(ys.round().astype(int), 0, H - 1)
    return img_np[ys, xs]   # H W 3


def draw_correspondences(
    img_A: np.ndarray,
    img_B: np.ndarray,
    warp: np.ndarray,
    overlap: np.ndarray,
    n: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Draw n random correspondence lines on a side-by-side canvas."""
    import cv2

    H, W = img_A.shape[:2]
    canvas = np.concatenate([img_A, img_B], axis=1).copy()

    rng = np.random.default_rng(seed)
    # Only sample from high-overlap regions
    mask = (overlap[..., 0] > 0.5)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ys, xs = ys.ravel(), xs.ravel()
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)

    for i in idx:
        y_A, x_A = int(ys[i]), int(xs[i])
        wx = (warp[y_A, x_A, 0] + 1) / 2 * (W - 1)
        wy = (warp[y_A, x_A, 1] + 1) / 2 * (H - 1)
        x_B = int(np.clip(round(wx), 0, W - 1)) + W  # offset for right panel
        y_B = int(np.clip(round(wy), 0, H - 1))
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        cv2.line(canvas, (x_A, y_A), (x_B, y_B), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x_A, y_A), 3, color, -1)
        cv2.circle(canvas, (x_B, y_B), 3, color, -1)

    return canvas


# ── inference ────────────────────────────────────────────────────────────────

def run_pytorch(img_A_t: torch.Tensor, img_B_t: torch.Tensor, setting: str):
    from export_onnx import build_model
    torch.set_float32_matmul_precision("highest")
    wrapper = build_model(setting, force_cpu=True)
    with torch.no_grad():
        warp, overlap = wrapper(img_A_t, img_B_t)
    return warp[0].numpy(), overlap[0].numpy()  # H W 2,  H W 1


def run_onnx(
    img_A_np: np.ndarray,
    img_B_np: np.ndarray,
    img_A_t: torch.Tensor,
    img_B_t: torch.Tensor,
    onnx_path: str,
):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    warp, overlap = sess.run(
        None,
        {"img_A": img_A_t.numpy(), "img_B": img_B_t.numpy()},
    )
    return warp[0], overlap[0]  # H W 2,  H W 1


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-a",   default="assets/toronto_A.jpg")
    parser.add_argument("--img-b",   default="assets/toronto_B.jpg")
    parser.add_argument("--setting", default="fast",
                        choices=["turbo", "fast", "base"])
    parser.add_argument("--onnx",    default=None,
                        help="Path to .onnx file; omit to use PyTorch")
    parser.add_argument("--out",     default="assets/match_result.png")
    args = parser.parse_args()

    # Determine resolution from setting
    res = {"turbo": 256, "fast": 512, "base": 640}[args.setting]
    H = W = res

    print(f"Loading images at {H}x{W} ...")
    img_A_np, img_A_t = load_image(args.img_a, H, W)
    img_B_np, img_B_t = load_image(args.img_b, H, W)

    if args.onnx:
        print(f"Running ONNX inference from {args.onnx} ...")
        warp, overlap = run_onnx(img_A_np, img_B_np, img_A_t, img_B_t, args.onnx)
    else:
        print("Running PyTorch inference ...")
        warp, overlap = run_pytorch(img_A_t, img_B_t, args.setting)

    print(f"warp   : shape={warp.shape}, min={warp.min():.3f}, max={warp.max():.3f}")
    print(f"overlap: shape={overlap.shape}, min={overlap.min():.3f}, max={overlap.max():.3f}")

    # 1) Warp image B into image A's coordinate system
    warped_B = warp_image(img_B_np, warp)

    # 2) Overlap / confidence map  (grayscale → colourmap)
    conf_uint8 = (overlap[..., 0] * 255).clip(0, 255).astype(np.uint8)
    import cv2
    conf_colour = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_INFERNO)
    conf_colour = cv2.cvtColor(conf_colour, cv2.COLOR_BGR2RGB)

    # 3) Correspondence lines
    lines = draw_correspondences(img_A_np, img_B_np, warp, overlap, n=150)

    # 4) Blend: alpha-composite warped_B onto img_A using overlap as mask
    alpha = overlap[..., :1].clip(0, 1)
    blend = (img_A_np.astype(float) * (1 - alpha) + warped_B.astype(float) * alpha).clip(0, 255).astype(np.uint8)

    # Build composite output
    # Row 1: img_A | img_B | warped_B_in_A
    # Row 2: overlap_confidence | blend | correspondences (half-size)
    row1 = np.concatenate([img_A_np, img_B_np, warped_B], axis=1)

    # resize correspondences to same height as a single image
    lines_resized = cv2.resize(lines, (W, H))
    row2 = np.concatenate([conf_colour, blend, lines_resized], axis=1)

    composite = np.concatenate([row1, row2], axis=0)

    out_path = args.out
    Image.fromarray(composite).save(out_path)
    print(f"\nSaved → {out_path}")

    # Labels
    print("\nLayout:")
    print("  Top-left:    Image A (query)")
    print("  Top-center:  Image B (reference)")
    print("  Top-right:   Image B warped into A's coordinate system")
    print("  Bot-left:    Overlap confidence (bright = high confidence)")
    print("  Bot-center:  Alpha-blended A + warped-B")
    print("  Bot-right:   100+ correspondences drawn (high-confidence only)")


if __name__ == "__main__":
    main()
