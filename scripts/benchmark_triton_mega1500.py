"""Run the Mega-1500 pose benchmark against a sampled RoMaV2 Triton ensemble.

The upstream RoMaV2 benchmark expects a Python model object with match(),
sample(), and to_pixel_coordinates() methods. This adapter keeps that surface
while delegating sampling to the Triton ensemble, which already returns sparse
correspondences.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


MODEL_H = MODEL_W = 512
OUTPUT_NAMES = [
    "sampled_matches",
    "sampled_confidence",
    "sampled_precision_A",
    "sampled_precision_B",
]
MEGA1500_SCENES = [
    "0015_0.1_0.3.npz",
    "0015_0.3_0.5.npz",
    "0022_0.1_0.3.npz",
    "0022_0.3_0.5.npz",
    "0022_0.5_0.7.npz",
]


def load_mega1500_class() -> type:
    benchmark_path = Path(__file__).resolve().parents[1] / "src" / "romav2" / "benchmarks" / "mega1500.py"
    spec = importlib.util.spec_from_file_location("romav2_mega1500_direct", benchmark_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Mega1500 benchmark from {benchmark_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Mega1500


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def make_client(protocol: str, url: str):
    if protocol == "grpc":
        import tritonclient.grpc as client_mod
    else:
        import tritonclient.http as client_mod

    return client_mod, client_mod.InferenceServerClient(url=url)


class TritonSampledRoMaV2:
    def __init__(
        self,
        *,
        url: str,
        model_name: str,
        protocol: str,
        seed: int,
        timeout: float,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.timeout = timeout
        self.protocol = protocol
        self.client_mod, self.client = make_client(protocol, url)
        self.sample_index = 0

    def match(self, img_a: str, img_b: str) -> tuple[str, str]:
        return img_a, img_b

    def sample(self, preds: tuple[str, str], num_corresp: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_a, img_b = preds
        image_a = load_image(img_a)
        image_b = load_image(img_b)
        num = np.array([num_corresp], dtype=np.int64)
        seed = self._next_seed()

        inputs = [
            self.client_mod.InferInput("img_A", image_a.shape, "FP32"),
            self.client_mod.InferInput("img_B", image_b.shape, "FP32"),
            self.client_mod.InferInput("num_corresp", num.shape, "INT64"),
            self.client_mod.InferInput("seed", seed.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(image_a)
        inputs[1].set_data_from_numpy(image_b)
        inputs[2].set_data_from_numpy(num)
        inputs[3].set_data_from_numpy(seed)

        outputs = [self.client_mod.InferRequestedOutput(name) for name in OUTPUT_NAMES]
        infer_kwargs: dict[str, Any] = {
            "model_name": self.model_name,
            "inputs": inputs,
            "outputs": outputs,
        }
        if self.protocol == "grpc":
            infer_kwargs["client_timeout"] = self.timeout
        response = self.client.infer(**infer_kwargs)

        return (
            torch.from_numpy(response.as_numpy("sampled_matches")),
            torch.from_numpy(response.as_numpy("sampled_confidence")),
            torch.from_numpy(response.as_numpy("sampled_precision_A")),
            torch.from_numpy(response.as_numpy("sampled_precision_B")),
        )

    def _next_seed(self) -> np.ndarray:
        if self.seed < 0:
            value = -1
        else:
            value = self.seed + self.sample_index
            self.sample_index += 1
        return np.array([value], dtype=np.int64)

    def to_pixel_coordinates(
        self,
        matches: torch.Tensor,
        h_a: float,
        w_a: float,
        h_b: float,
        w_b: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._to_pixel(matches[..., :2], h_a, w_a), self._to_pixel(matches[..., 2:], h_b, w_b)

    @staticmethod
    def _to_pixel(points: torch.Tensor, height: float, width: float) -> torch.Tensor:
        return torch.stack(
            (
                (points[..., 0] + 1.0) / 2.0 * width,
                (points[..., 1] + 1.0) / 2.0 * height,
            ),
            dim=-1,
        )


def check_data(data_root: Path) -> bool:
    missing = [scene for scene in MEGA1500_SCENES if not (data_root / scene).exists()]
    if not missing:
        return True
    print(f"Mega-1500 data is missing under {data_root}:")
    for scene in missing:
        print(f"  missing: {scene}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Triton RoMaV2 on Mega-1500")
    parser.add_argument("--data-root", default="data/megadepth")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model", default="romav2_bidirectional_fast_sampled")
    parser.add_argument("--protocol", choices=["http", "grpc"], default="http")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--check-data", action="store_true", help="Only check that Mega-1500 files exist")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_ok = check_data(data_root)
    if args.check_data or not data_ok:
        raise SystemExit(0 if data_ok else 1)

    Mega1500 = load_mega1500_class()
    benchmark = Mega1500(str(data_root))
    model = TritonSampledRoMaV2(
        url=args.url,
        model_name=args.model,
        protocol=args.protocol,
        seed=args.seed,
        timeout=args.timeout,
    )
    print(benchmark.benchmark(model, model_name=args.model))


if __name__ == "__main__":
    main()
