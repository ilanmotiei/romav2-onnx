"""RoMaV2 Triton sampled ensemble client.

Usage:
    python scripts/triton_sampled_client.py assets/toronto_A.jpg assets/toronto_B.jpg
    python scripts/triton_sampled_client.py assets/toronto_A.jpg assets/toronto_B.jpg --out sampled.png
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw


MODEL_H = MODEL_W = 512


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def infer_sampled(
    img_a: str,
    img_b: str,
    *,
    url: str,
    model_name: str,
    num_corresp: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import tritonclient.http as httpclient
    except ImportError:
        raise SystemExit("tritonclient is required: pip install tritonclient[http]")

    image_a = load_image(img_a)
    image_b = load_image(img_b)
    num = np.array([num_corresp], dtype=np.int64)
    seed_arr = np.array([seed], dtype=np.int64)

    inputs = [
        httpclient.InferInput("img_A", image_a.shape, "FP32"),
        httpclient.InferInput("img_B", image_b.shape, "FP32"),
        httpclient.InferInput("num_corresp", num.shape, "INT64"),
        httpclient.InferInput("seed", seed_arr.shape, "INT64"),
    ]
    inputs[0].set_data_from_numpy(image_a)
    inputs[1].set_data_from_numpy(image_b)
    inputs[2].set_data_from_numpy(num)
    inputs[3].set_data_from_numpy(seed_arr)

    outputs = [
        httpclient.InferRequestedOutput("sampled_matches"),
        httpclient.InferRequestedOutput("sampled_confidence"),
        httpclient.InferRequestedOutput("sampled_precision_A"),
        httpclient.InferRequestedOutput("sampled_precision_B"),
    ]
    client = httpclient.InferenceServerClient(url=url)
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    return (
        response.as_numpy("sampled_matches"),
        response.as_numpy("sampled_confidence"),
        response.as_numpy("sampled_precision_A"),
        response.as_numpy("sampled_precision_B"),
    )


def _to_pixel(points: np.ndarray, *, width: int, height: int) -> np.ndarray:
    x = (points[:, 0] + 1.0) / 2.0 * width - 0.5
    y = (points[:, 1] + 1.0) / 2.0 * height - 0.5
    return np.stack((x, y), axis=-1)


def visualise_sampled(
    img_a_path: str,
    img_b_path: str,
    matches: np.ndarray,
    confidence: np.ndarray,
    out_path: str,
    *,
    max_lines: int = 512,
) -> None:
    img_a = Image.open(img_a_path).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS)
    img_b = Image.open(img_b_path).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS)
    canvas = Image.new("RGB", (MODEL_W * 2, MODEL_H), "white")
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (MODEL_W, 0))

    count = min(max_lines, matches.shape[0])
    order = np.argsort(confidence)[::-1][:count]
    pts_a = _to_pixel(matches[order, :2], width=MODEL_W, height=MODEL_H)
    pts_b = _to_pixel(matches[order, 2:], width=MODEL_W, height=MODEL_H)
    conf = confidence[order]
    conf_min = float(conf.min()) if conf.size else 0.0
    conf_max = float(conf.max()) if conf.size else 1.0
    denom = max(conf_max - conf_min, 1e-6)

    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for point_a, point_b, score in zip(pts_a, pts_b, conf):
        t = float((score - conf_min) / denom)
        color = (
            int(40 + 215 * t),
            int(220 - 180 * t),
            int(255 - 220 * t),
            165,
        )
        x_a, y_a = point_a
        x_b, y_b = point_b
        x_b += MODEL_W
        draw.line([(x_a, y_a), (x_b, y_b)], fill=color, width=1)
        draw.ellipse((x_a - 2, y_a - 2, x_a + 2, y_a + 2), fill=color)
        draw.ellipse((x_b - 2, y_b - 2, x_b + 2, y_b + 2), fill=color)

    visual = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    visual.save(out_path)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the RoMaV2 sampled Triton ensemble")
    parser.add_argument("img_a")
    parser.add_argument("img_b")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model", default="romav2_bidirectional_sampled")
    parser.add_argument("--num-corresp", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=-1,
                        help="Sampling seed; -1 uses non-deterministic sampling")
    parser.add_argument("--out", default=None,
                        help="Save sampled correspondence visualisation to this path")
    parser.add_argument("--max-lines", type=int, default=512,
                        help="Maximum correspondences to draw when --out is set")
    args = parser.parse_args()

    matches, confidence, precision_a, precision_b = infer_sampled(
        args.img_a,
        args.img_b,
        url=args.url,
        model_name=args.model,
        num_corresp=args.num_corresp,
        seed=args.seed,
    )
    print(f"sampled_matches:     shape={matches.shape}, min={matches.min():.4f}, max={matches.max():.4f}")
    print(f"sampled_confidence:  shape={confidence.shape}, min={confidence.min():.4f}, max={confidence.max():.4f}")
    print(f"sampled_precision_A: shape={precision_a.shape}, min={precision_a.min():.4f}, max={precision_a.max():.4f}")
    print(f"sampled_precision_B: shape={precision_b.shape}, min={precision_b.min():.4f}, max={precision_b.max():.4f}")
    if args.out:
        visualise_sampled(
            args.img_a,
            args.img_b,
            matches,
            confidence,
            args.out,
            max_lines=args.max_lines,
        )


if __name__ == "__main__":
    main()
