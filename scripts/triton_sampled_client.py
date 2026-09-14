"""RoMaV2 Triton sampled ensemble client.

Usage:
    python scripts/triton_sampled_client.py assets/toronto_A.jpg assets/toronto_B.jpg
    python scripts/triton_sampled_client.py A.jpg B.jpg --model romav2_bidirectional_sampled --setting base --out sampled.png
    python scripts/triton_sampled_client.py A.jpg B.jpg --model romav2_bidirectional_sampled --setting base --out sampled.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ → visualize helpers
from visualize import INPUT_SIZES, prepare  # noqa: E402


def _display_size(setting: str) -> int:
    """Resolution the ensemble's matches refer to: the model's input size."""
    return INPUT_SIZES[setting]


def infer_sampled(
    img_a: str,
    img_b: str,
    *,
    url: str,
    model_name: str,
    num_corresp: int,
    seed: int,
    setting: str = "fast",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import tritonclient.http as httpclient
    except ImportError:
        raise SystemExit("tritonclient is required: pip install tritonclient[http]")

    _, _, images = prepare(img_a, img_b, setting)   # named inputs the setting defines
    num = np.array([num_corresp], dtype=np.int64)
    seed_arr = np.array([seed], dtype=np.int64)

    inputs = []
    for name, arr in images.items():
        inp = httpclient.InferInput(name, arr.shape, "FP32")
        inp.set_data_from_numpy(arr)
        inputs.append(inp)
    for name, arr in (("num_corresp", num), ("seed", seed_arr)):
        inp = httpclient.InferInput(name, arr.shape, "INT64")
        inp.set_data_from_numpy(arr)
        inputs.append(inp)

    outputs = [
        httpclient.InferRequestedOutput("sampled_matches"),
        httpclient.InferRequestedOutput("sampled_confidence"),
        httpclient.InferRequestedOutput("sampled_precision_A"),
        httpclient.InferRequestedOutput("sampled_precision_B"),
    ]
    client = httpclient.InferenceServerClient(url=url, network_timeout=600.0, connection_timeout=600.0)
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
    setting: str = "fast",
    max_lines: int = 512,
) -> None:
    size = _display_size(setting)
    img_a = Image.open(img_a_path).convert("RGB").resize((size, size), Image.LANCZOS)
    img_b = Image.open(img_b_path).convert("RGB").resize((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size * 2, size), "white")
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (size, 0))

    count = min(max_lines, matches.shape[0])
    order = np.argsort(confidence)[::-1][:count]
    pts_a = _to_pixel(matches[order, :2], width=size, height=size)
    pts_b = _to_pixel(matches[order, 2:], width=size, height=size)
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
        x_b += size
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
    parser.add_argument("--setting", default="base", choices=list(INPUT_SIZES),
                        help="setting the ensemble's dense model was exported with "
                             "(romav2_bidirectional_sampled: base/640)")
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
        setting=args.setting,
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
            setting=args.setting,
            max_lines=args.max_lines,
        )


if __name__ == "__main__":
    main()
