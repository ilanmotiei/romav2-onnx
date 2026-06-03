"""Benchmark the sampled RoMaV2 Triton ensemble under repeated load."""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image


MODEL_H = MODEL_W = 512
OUTPUT_NAMES = [
    "sampled_matches",
    "sampled_confidence",
    "sampled_precision_A",
    "sampled_precision_B",
]


@dataclass(frozen=True)
class RequestData:
    img_a: np.ndarray
    img_b: np.ndarray
    num_corresp: np.ndarray
    seed: np.ndarray


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def make_http_client(url: str, model_name: str, timeout: float):
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=url, concurrency=1)

    def infer(data: RequestData):
        inputs = [
            httpclient.InferInput("img_A", data.img_a.shape, "FP32"),
            httpclient.InferInput("img_B", data.img_b.shape, "FP32"),
            httpclient.InferInput("num_corresp", data.num_corresp.shape, "INT64"),
            httpclient.InferInput("seed", data.seed.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(data.img_a)
        inputs[1].set_data_from_numpy(data.img_b)
        inputs[2].set_data_from_numpy(data.num_corresp)
        inputs[3].set_data_from_numpy(data.seed)
        outputs = [httpclient.InferRequestedOutput(name) for name in OUTPUT_NAMES]
        return client.infer(
            model_name,
            inputs=inputs,
            outputs=outputs,
            request_id="benchmark",
        )

    return infer


def make_grpc_client(url: str, model_name: str, timeout: float):
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=url)

    def infer(data: RequestData):
        inputs = [
            grpcclient.InferInput("img_A", data.img_a.shape, "FP32"),
            grpcclient.InferInput("img_B", data.img_b.shape, "FP32"),
            grpcclient.InferInput("num_corresp", data.num_corresp.shape, "INT64"),
            grpcclient.InferInput("seed", data.seed.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(data.img_a)
        inputs[1].set_data_from_numpy(data.img_b)
        inputs[2].set_data_from_numpy(data.num_corresp)
        inputs[3].set_data_from_numpy(data.seed)
        outputs = [grpcclient.InferRequestedOutput(name) for name in OUTPUT_NAMES]
        return client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=timeout,
        )

    return infer


def run_one(infer, data: RequestData) -> float:
    start = time.perf_counter()
    infer(data)
    return time.perf_counter() - start


def summarize(name: str, latencies: list[float], wall_time: float):
    qps = len(latencies) / wall_time if wall_time > 0 else float("nan")
    print(f"\n{name}")
    print(f"  requests: {len(latencies)}")
    print(f"  wall:     {wall_time:.3f}s")
    print(f"  qps:      {qps:.3f}")
    print(f"  mean:     {statistics.mean(latencies):.3f}s")
    print(f"  p50:      {percentile(latencies, 50):.3f}s")
    print(f"  p95:      {percentile(latencies, 95):.3f}s")
    print(f"  p99:      {percentile(latencies, 99):.3f}s")
    print(f"  min/max:  {min(latencies):.3f}s / {max(latencies):.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sampled RoMaV2 Triton ensemble")
    parser.add_argument("img_a")
    parser.add_argument("img_b")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model", default="romav2_bidirectional_sampled")
    parser.add_argument("--protocol", choices=["http", "grpc"], default="http")
    parser.add_argument("--num-corresp", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    control_shape = (1, 1) if args.model.endswith("_batch") else (1,)
    data = RequestData(
        img_a=load_image(args.img_a),
        img_b=load_image(args.img_b),
        num_corresp=np.full(control_shape, args.num_corresp, dtype=np.int64),
        seed=np.full(control_shape, args.seed, dtype=np.int64),
    )
    make_client = make_http_client if args.protocol == "http" else make_grpc_client

    def make_infer():
        return make_client(args.url, args.model, args.timeout)

    print(
        f"protocol={args.protocol} url={args.url} model={args.model} requests={args.requests} "
        f"concurrency={args.concurrency} num_corresp={args.num_corresp}"
    )

    warmup_infer = make_infer()
    for _ in range(args.warmup):
        warmup_infer(data)

    start = time.perf_counter()
    if args.concurrency == 1:
        infer = make_infer()
        latencies = [run_one(infer, data) for _ in range(args.requests)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(run_one, make_infer(), data)
                for _ in range(args.requests)
            ]
            latencies = [future.result() for future in concurrent.futures.as_completed(futures)]
    wall_time = time.perf_counter() - start
    summarize("sampled ensemble", latencies, wall_time)


if __name__ == "__main__":
    main()
