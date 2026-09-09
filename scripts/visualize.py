"""Visualize RoMaV2 matching on the toronto sample images.

Usage:
    python scripts/visualize.py                                  # PyTorch model, fast
    python scripts/visualize.py --onnx romav2_fast.onnx          # ONNX model
    python scripts/visualize.py --onnx romav2_precise.onnx --setting precise
    python scripts/visualize.py --img-a assets/toronto_A.jpg --img-b assets/toronto_B.jpg

Every setting takes one image per side at its input size (see INPUT_SIZES) and
returns both directions plus precision, so the composite is always:
    A->B  row 1: image A            | image B         | B warped into A (warp_AB)
          row 2: overlap_AB heatmap | alpha blend     | correspondences (overlap > 0.5)
    B->A  (same, with the roles swapped)
    err   expected error A->B | expected error B->A | legend

The image helpers here are shared with scripts/triton_client.py and
scripts/triton_sampled_client.py; torch is only imported for the PyTorch path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # embedded romav2
sys.path.insert(0, str(Path(__file__).resolve().parent))               # scripts/ → export_onnx

# Input size per setting — the image the exported model takes (H_hr for the
# two-stage precise setting, which derives its 800 low-res pass in-graph).
INPUT_SIZES = {"turbo": 320, "fast": 512, "base": 640, "precise": 1280}


# ── inputs ───────────────────────────────────────────────────────────────────

def load_image(path: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize to size×size; return (HxWx3 uint8 for display, 1x3xHxW float32 in [0,1]).

    PIL's BICUBIC is the same antialiased cubic RoMaV2.match() applies with
    torch (bicubic, antialias=True), so this matches the upstream preprocessing.
    """
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(img)                                                  # H W 3 uint8
    tensor = (arr.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # 1 3 H W
    return arr, tensor


def prepare(img_A_path: str, img_B_path: str, setting: str
            ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Display images + the model's named inputs {img_A, img_B} at the setting's input size.

    Outputs come back at that same size, so the display images line up with them.
    """
    size = INPUT_SIZES[setting]
    disp_A, in_A = load_image(img_A_path, size)
    disp_B, in_B = load_image(img_B_path, size)
    return disp_A, disp_B, {"img_A": in_A, "img_B": in_B}


# ── drawing helpers ──────────────────────────────────────────────────────────

def to_pixel(warp: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised [-1,1] coords → integer pixel indices (align_corners=False, as RoMaV2's grid)."""
    xs = np.clip(np.round((warp[..., 0] + 1) / 2 * W - 0.5).astype(int), 0, W - 1)
    ys = np.clip(np.round((warp[..., 1] + 1) / 2 * H - 0.5).astype(int), 0, H - 1)
    return xs, ys


def warp_image(img_np: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Sample the warp's target image at warp positions → that image in the source frame."""
    H, W = img_np.shape[:2]
    xs, ys = to_pixel(warp, H, W)
    return img_np[ys, xs]


def draw_correspondences(
    img_A: np.ndarray,
    img_B: np.ndarray,
    warp: np.ndarray,
    overlap: np.ndarray,
    n: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Draw n random correspondence lines on a side-by-side canvas."""
    H, W = img_A.shape[:2]
    canvas = Image.fromarray(np.concatenate([img_A, img_B], axis=1).copy())
    draw = ImageDraw.Draw(canvas)

    rng = np.random.default_rng(seed)
    # Only sample from high-overlap regions
    ys, xs = np.where(overlap[..., 0] > 0.5)
    if len(ys) == 0:
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ys, xs = ys.ravel(), xs.ravel()
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    wx, wy = to_pixel(warp, H, W)

    for i in idx:
        y_A, x_A = int(ys[i]), int(xs[i])
        x_B, y_B = int(wx[y_A, x_A]) + W, int(wy[y_A, x_A])  # offset for right panel
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        draw.line([(x_A, y_A), (x_B, y_B)], fill=color, width=1)
        draw.ellipse((x_A - 3, y_A - 3, x_A + 3, y_A + 3), fill=color)
        draw.ellipse((x_B - 3, y_B - 3, x_B + 3, y_B + 3), fill=color)

    return np.asarray(canvas)


def confidence_to_colour(confidence: np.ndarray) -> np.ndarray:
    """Convert a map in [0, 1] to a compact inferno-like heatmap."""
    x = confidence.clip(0, 1)[..., None]
    stops = np.array(
        [
            [0, 0, 4],
            [87, 15, 109],
            [187, 55, 84],
            [249, 142, 8],
            [252, 255, 164],
        ],
        dtype=np.float32,
    )
    scaled = x * (len(stops) - 1)
    lo = np.floor(scaled).astype(np.int32).clip(0, len(stops) - 1)
    hi = (lo + 1).clip(0, len(stops) - 1)
    frac = scaled - lo
    colour = stops[lo[..., 0]] * (1 - frac) + stops[hi[..., 0]] * frac
    return colour.clip(0, 255).astype(np.uint8)


ERROR_MAX_PX = 4.0


def expected_error_map(precision: np.ndarray, overlap: np.ndarray, max_px: float = ERROR_MAX_PX) -> np.ndarray:
    """Expected match error in pixels, det(P)^(-1/4), where overlap > 0.5; black elsewhere."""
    det = precision[..., 0, 0] * precision[..., 1, 1] - precision[..., 0, 1] * precision[..., 1, 0]
    err_px = np.power(np.clip(det, 1e-12, None), -0.25)
    vis = (err_px / max_px).clip(0, 1)
    vis[overlap[..., 0] <= 0.5] = 0.0
    return confidence_to_colour(vis)


def error_legend(H: int, W: int, max_px: float = ERROR_MAX_PX) -> np.ndarray:
    """A panel with the colour bar used by expected_error_map."""
    panel = Image.fromarray(np.full((H, W, 3), 24, dtype=np.uint8))
    draw = ImageDraw.Draw(panel)
    bar_w, bar_h = int(W * 0.7), max(12, H // 25)
    x0, y0 = (W - bar_w) // 2, H // 2 - bar_h
    bar = confidence_to_colour(np.tile(np.linspace(0, 1, bar_w, dtype=np.float32), (bar_h, 1)))
    panel.paste(Image.fromarray(bar), (x0, y0))
    draw.text((x0, y0 + bar_h + 6), "0 px", fill=(230, 230, 230))
    draw.text((x0 + bar_w - 40, y0 + bar_h + 6), f">= {max_px:g} px", fill=(230, 230, 230))
    draw.text((x0, y0 - 18), "expected match error (from precision); black = overlap <= 0.5",
              fill=(230, 230, 230))
    return np.asarray(panel)


def build_direction_composite(
    img_query: np.ndarray,
    img_reference: np.ndarray,
    warp: np.ndarray,
    overlap: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Build a 2x3 visualization for one matching direction."""
    warped_reference = warp_image(img_reference, warp)

    conf_colour = confidence_to_colour(overlap[..., 0])

    lines = draw_correspondences(img_query, img_reference, warp, overlap, n=150, seed=seed)

    alpha = overlap[..., :1].clip(0, 1)
    blend = (
        img_query.astype(float) * (1 - alpha)
        + warped_reference.astype(float) * alpha
    ).clip(0, 255).astype(np.uint8)

    H, W = img_query.shape[:2]
    row1 = np.concatenate([img_query, img_reference, warped_reference], axis=1)
    lines_resized = np.asarray(Image.fromarray(lines).resize((W, H), Image.LANCZOS))
    row2 = np.concatenate([conf_colour, blend, lines_resized], axis=1)
    return np.concatenate([row1, row2], axis=0)


def build_composite(img_A: np.ndarray, img_B: np.ndarray, outputs: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the A->B block, the B->A block (if present) and the error block (if present)."""
    blocks = [build_direction_composite(img_A, img_B, outputs["warp_AB"], outputs["overlap_AB"], seed=0)]
    separator = np.full((12, blocks[0].shape[1], 3), 255, dtype=np.uint8)

    has_ba = "warp_BA" in outputs and "overlap_BA" in outputs
    if has_ba:
        blocks += [separator, build_direction_composite(
            img_B, img_A, outputs["warp_BA"], outputs["overlap_BA"], seed=1)]

    if "precision_AB" in outputs:
        H, W = img_A.shape[:2]
        err_AB = expected_error_map(outputs["precision_AB"], outputs["overlap_AB"])
        err_BA = (expected_error_map(outputs["precision_BA"], outputs["overlap_BA"])
                  if has_ba and "precision_BA" in outputs else np.zeros_like(err_AB))
        blocks += [separator, np.concatenate([err_AB, err_BA, error_legend(H, W)], axis=1)]
    return np.concatenate(blocks, axis=0)


# ── inference ────────────────────────────────────────────────────────────────

def run_pytorch(inputs: dict[str, np.ndarray], setting: str) -> dict[str, np.ndarray]:
    import torch
    from export_onnx import build_model, io_spec
    wrapper = build_model(setting, force_cpu=True)
    input_names, output_names, _, _ = io_spec(wrapper)
    with torch.no_grad():
        outs = wrapper(*[torch.from_numpy(inputs[n]) for n in input_names])
    return {n: o[0].numpy() for n, o in zip(output_names, outs)}


def run_onnx(inputs: dict[str, np.ndarray], onnx_path: str) -> dict[str, np.ndarray]:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    if set(names) != set(inputs):
        raise SystemExit(f"{onnx_path} expects inputs {names}, but the setting provides "
                         f"{list(inputs)}; wrong --setting?")
    values = sess.run(None, {n: inputs[n] for n in names})
    return {o.name: v[0] for o, v in zip(sess.get_outputs(), values)}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-a",   default="assets/toronto_A.jpg")
    parser.add_argument("--img-b",   default="assets/toronto_B.jpg")
    parser.add_argument("--setting", default="fast", choices=list(INPUT_SIZES))
    parser.add_argument("--onnx",    default=None, help="Path to .onnx file; omit to use PyTorch")
    parser.add_argument("--out",     default="assets/match_result.png")
    args = parser.parse_args()

    print(f"Loading images (setting={args.setting}: {INPUT_SIZES[args.setting]}x{INPUT_SIZES[args.setting]}) ...")
    img_A_np, img_B_np, inputs = prepare(args.img_a, args.img_b, args.setting)

    if args.onnx:
        print(f"Running ONNX inference from {args.onnx} ...")
        outputs = run_onnx(inputs, args.onnx)
    else:
        print("Running PyTorch inference ...")
        outputs = run_pytorch(inputs, args.setting)

    for name, value in outputs.items():
        print(f"{name:<13}: shape={value.shape}, min={value.min():.3f}, max={value.max():.3f}")

    Image.fromarray(build_composite(img_A_np, img_B_np, outputs)).save(args.out)
    print(f"\nSaved → {args.out}")

    # Labels
    print("\nLayout:")
    print("  A→B row 1: Image A | Image B | Image B warped into A")
    print("  A→B row 2: Overlap confidence | Alpha blend | Correspondences")
    if "warp_BA" in outputs:
        print("  B→A row 1: Image B | Image A | Image A warped into B")
        print("  B→A row 2: Overlap confidence | Alpha blend | Correspondences")
    if "precision_AB" in outputs:
        print(f"  err   row: Expected error A→B | Expected error B→A | legend (0..{ERROR_MAX_PX:g} px)")


if __name__ == "__main__":
    main()
