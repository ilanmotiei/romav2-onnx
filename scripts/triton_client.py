"""RoMaV2 Triton Inference Client (dense models).

Usage:
    # Fast model (512x512, A->B only)
    python scripts/triton_client.py assets/toronto_A.jpg assets/toronto_B.jpg

    # Base model at 640 (romav2_bidirectional_dense)
    python scripts/triton_client.py A.jpg B.jpg --model romav2_bidirectional_dense --setting base

    # Precise model (one 1280x1280 image per side; the graph resizes for its 800 pass)
    python scripts/triton_client.py A.jpg B.jpg --model romav2_precise_dense --setting precise --out result.png

Every served RoMaV2 model has the same interface (img_A, img_B -> warp/overlap/precision
for AB and BA); --setting only fixes the input size.

    # Custom server address
    python scripts/triton_client.py img_A.jpg img_B.jpg --url myserver:8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ → visualize helpers
from visualize import INPUT_SIZES, build_composite, prepare  # noqa: E402


def infer(
    img_A_path: str,
    img_B_path: str,
    *,
    url: str = "localhost:8000",
    model_name: str = "romav2",
    model_version: str = "",
    setting: str = "fast",
    network_timeout: float = 600.0,
) -> dict[str, np.ndarray]:
    """Run inference on a Triton HTTP endpoint.

    Sends img_A / img_B at the setting's input size (see visualize.prepare) and asks
    for every output the served model declares.  Returns them without the batch dim:
        warp_*      (S, S, 2)     dense warp in normalised coords [-1, 1]
        overlap_*   (S, S, 1)     overlap probability in [0, 1]
        precision_* (S, S, 2, 2)  precision matrices
    """
    try:
        import tritonclient.http as httpclient
    except ImportError:
        raise SystemExit("tritonclient is required: pip install tritonclient[http]")

    _, _, inputs = prepare(img_A_path, img_B_path, setting)   # each 1 3 H W float32

    client = httpclient.InferenceServerClient(
        url=url, network_timeout=network_timeout, connection_timeout=network_timeout,
    )
    meta = client.get_model_metadata(model_name=model_name, model_version=model_version)
    expected = [i["name"] for i in meta["inputs"]]
    if set(expected) != set(inputs):
        raise SystemExit(f"model '{model_name}' expects inputs {expected}, but setting "
                         f"'{setting}' provides {list(inputs)}; pass the matching --setting")
    output_names = [o["name"] for o in meta["outputs"]]

    triton_inputs = []
    for name in expected:
        inp = httpclient.InferInput(name, inputs[name].shape, "FP32")
        inp.set_data_from_numpy(inputs[name])
        triton_inputs.append(inp)
    response = client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=triton_inputs,
        outputs=[httpclient.InferRequestedOutput(n) for n in output_names],
    )
    return {n: response.as_numpy(n)[0] for n in output_names}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoMaV2 Triton inference client")
    parser.add_argument("img_a", help="Path to image A")
    parser.add_argument("img_b", help="Path to image B")
    parser.add_argument("--url",     default="localhost:8000", help="Triton HTTP endpoint")
    parser.add_argument("--model",   default="romav2_bidirectional_dense", help="Model name registered in Triton")
    parser.add_argument("--setting", default="base", choices=list(INPUT_SIZES),
                        help="setting the served model was exported with (fixes the input size)")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds")
    parser.add_argument("--out",     default=None, help="Save visualisation to this path (optional)")
    args = parser.parse_args()

    print(f"Sending request to {args.url} / model={args.model} (setting={args.setting}) ...")
    outs = infer(args.img_a, args.img_b, url=args.url, model_name=args.model,
                 setting=args.setting, network_timeout=args.timeout)
    for name, arr in outs.items():
        print(f"{name:<13}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

    if args.out:
        img_A, img_B, _ = prepare(args.img_a, args.img_b, args.setting)
        Image.fromarray(build_composite(img_A, img_B, outs)).save(args.out)
        print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
