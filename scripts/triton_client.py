"""RoMaV2 Triton Inference Client.

Usage:
    # Match two images
    python scripts/triton_client.py assets/toronto_A.jpg assets/toronto_B.jpg

    # Save visualisation
    python scripts/triton_client.py assets/toronto_A.jpg assets/toronto_B.jpg --out result.png

    # Custom server address
    python scripts/triton_client.py img_A.jpg img_B.jpg --url myserver:8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# ── image helpers ─────────────────────────────────────────────────────────────

MODEL_H = MODEL_W = 512   # fixed for the "fast" export


def load_image(path: str) -> np.ndarray:
    """Load image → float32 CHW in [0, 1] for Triton (no batch dim)."""
    img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # H W 3
    return arr.transpose(2, 0, 1)                    # 3 H W


def visualise(
    img_A_path: str,
    img_B_path: str,
    warp: np.ndarray,
    overlap: np.ndarray,
    out_path: str,
) -> None:
    """Save a 6-panel composite (same layout as scripts/visualize.py)."""
    import cv2

    img_A = np.array(Image.open(img_A_path).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS))
    img_B = np.array(Image.open(img_B_path).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS))

    # Warp B into A's frame
    H, W = warp.shape[:2]
    xs = np.clip(((warp[..., 0] + 1) / 2 * (W - 1)).round().astype(int), 0, W - 1)
    ys = np.clip(((warp[..., 1] + 1) / 2 * (H - 1)).round().astype(int), 0, H - 1)
    warped_B = img_B[ys, xs]

    # Confidence colourmap
    conf_u8 = (overlap[..., 0] * 255).clip(0, 255).astype(np.uint8)
    conf_colour = cv2.cvtColor(cv2.applyColorMap(conf_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    # Blend
    alpha = overlap[..., :1].clip(0, 1)
    blend = (img_A * (1 - alpha) + warped_B * alpha).clip(0, 255).astype(np.uint8)

    # Correspondences
    canvas = np.concatenate([img_A, img_B], axis=1).copy()
    rng = np.random.default_rng(0)
    mask_ys, mask_xs = np.where(overlap[..., 0] > 0.5)
    if len(mask_ys) > 0:
        idx = rng.choice(len(mask_ys), size=min(150, len(mask_ys)), replace=False)
        for i in idx:
            y_A, x_A = int(mask_ys[i]), int(mask_xs[i])
            wx = int(np.clip(round((warp[y_A, x_A, 0] + 1) / 2 * (W - 1)), 0, W - 1))
            wy = int(np.clip(round((warp[y_A, x_A, 1] + 1) / 2 * (H - 1)), 0, H - 1))
            color = tuple(int(c) for c in rng.integers(80, 255, 3))
            cv2.line(canvas, (x_A, y_A), (wx + W, wy), color, 1, cv2.LINE_AA)
            cv2.circle(canvas, (x_A, y_A), 3, color, -1)
            cv2.circle(canvas, (wx + W, wy), 3, color, -1)
    lines_resized = cv2.resize(canvas, (W, H))

    row1 = np.concatenate([img_A, img_B, warped_B], axis=1)
    row2 = np.concatenate([conf_colour, blend, lines_resized], axis=1)
    Image.fromarray(np.concatenate([row1, row2], axis=0)).save(out_path)
    print(f"Saved → {out_path}")


# ── Triton client ─────────────────────────────────────────────────────────────

def infer(
    img_A_path: str,
    img_B_path: str,
    *,
    url: str = "localhost:8000",
    model_name: str = "romav2",
    model_version: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a Triton HTTP endpoint.

    Returns:
        warp_AB:    (H, W, 2)  dense warp in normalised coords [-1, 1]
        overlap_AB: (H, W, 1)  overlap probability in [0, 1]
    """
    try:
        import tritonclient.http as httpclient
    except ImportError:
        raise SystemExit(
            "tritonclient is required: pip install tritonclient[http]"
        )

    img_A = load_image(img_A_path)   # 3 H W  float32
    img_B = load_image(img_B_path)   # 3 H W  float32

    # Triton HTTP client expects a batch dimension
    img_A_batch = img_A[np.newaxis]  # 1 3 H W
    img_B_batch = img_B[np.newaxis]  # 1 3 H W

    client = httpclient.InferenceServerClient(url=url)

    inp_A = httpclient.InferInput("img_A", img_A_batch.shape, "FP32")
    inp_A.set_data_from_numpy(img_A_batch)

    inp_B = httpclient.InferInput("img_B", img_B_batch.shape, "FP32")
    inp_B.set_data_from_numpy(img_B_batch)

    out_warp    = httpclient.InferRequestedOutput("warp_AB")
    out_overlap = httpclient.InferRequestedOutput("overlap_AB")

    response = client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=[inp_A, inp_B],
        outputs=[out_warp, out_overlap],
    )

    warp_AB    = response.as_numpy("warp_AB")[0]    # H W 2
    overlap_AB = response.as_numpy("overlap_AB")[0] # H W 1

    return warp_AB, overlap_AB


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoMaV2 Triton inference client")
    parser.add_argument("img_a", help="Path to image A")
    parser.add_argument("img_b", help="Path to image B")
    parser.add_argument("--url",   default="localhost:8000",
                        help="Triton HTTP endpoint (default: localhost:8000)")
    parser.add_argument("--model", default="romav2",
                        help="Model name registered in Triton (default: romav2)")
    parser.add_argument("--out",   default=None,
                        help="Save visualisation to this path (optional)")
    args = parser.parse_args()

    print(f"Sending request to {args.url} / model={args.model} ...")
    warp, overlap = infer(args.img_a, args.img_b, url=args.url, model_name=args.model)

    print(f"warp_AB:    shape={warp.shape}, min={warp.min():.4f}, max={warp.max():.4f}")
    print(f"overlap_AB: shape={overlap.shape}, min={overlap.min():.4f}, max={overlap.max():.4f}")

    if args.out:
        visualise(args.img_a, args.img_b, warp, overlap, args.out)


if __name__ == "__main__":
    main()
