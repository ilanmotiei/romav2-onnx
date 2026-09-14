"""Evaluate a served RoMaV2 Triton module against the PyTorch checkpoint.

Meant to run on the machine that hosts Triton (the GPU box).  Two phases:

    # 1) reference outputs from the checkpoint on the GPU -- scale Triton down first
    python scripts/triton_eval.py reference --setting precise --out bench_gpu/precise_ref.npz

    # 2) served module: accuracy against that reference, then timings
    python scripts/triton_eval.py triton --setting precise --name romav2_precise \\
        --url localhost:8000 --ref bench_gpu/precise_ref.npz --out bench_gpu/precise_eval.json

Reference outputs come from two configurations of the same checkpoint, run on the
sample pairs of scripts/benchmark.py:

  * upstream  RoMaV2 defaults (AMP/bf16 on CUDA), low- and high-res inputs both
              resized from the original like RoMaV2.match() does.  "What upstream
              returns."  (torch.compile is left off: same numerics, minutes saved.)
  * fp32      the export configuration (AMP off, RoPE fp32) on the same inputs, so
              AMP noise can be told apart from export error.

Accuracy is compared where the reference says the images overlap (overlap > 0.5);
see export_onnx.validate for why a global max-abs is meaningless for this model.
Warp differences are also given in pixels of the S x S input.

Timings are medians of several runs.  Client wall time is HTTP end-to-end (the
dense response alone is ~4 * S*S*10 bytes, 65 MB at 1280).  Server-side times
are taken from Triton's per-model statistics, read before and after each request,
so they exclude the transfer: `server` is the request's total time inside Triton,
`compute` the backend's compute_infer.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import INPUT_SIZES, make_samples, resize, to_tensor  # noqa: E402
from triton_configs import DEFAULT_NAME, model_names  # noqa: E402

DIRS = ("AB", "BA")
KEYS = ("warp", "overlap", "precision")
STAT_KEYS = ("success", "queue", "compute_input", "compute_infer", "compute_output")


# ── reference (PyTorch checkpoint) ───────────────────────────────────────────

def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_model(model, A, B, runs: int):
    """Run the checkpoint the way RoMaV2.match() feeds it (every input resized from
    the original).  Returns (outputs without batch dim, forward seconds per run)."""
    import torch
    from romav2.romav2 import _map_confidence

    dev = next(model.parameters()).device
    args = [resize(A, model.H_lr).to(dev), resize(B, model.H_lr).to(dev)]
    kw = {}
    if model.H_hr is not None:
        kw = dict(img_A_hr=resize(A, model.H_hr).to(dev), img_B_hr=resize(B, model.H_hr).to(dev))
    times = []
    with torch.no_grad():
        for i in range(runs + 1):          # first pass is a warm-up
            _sync()
            t = time.perf_counter()
            preds = model(*args, **kw)
            _sync()
            if i:
                times.append(time.perf_counter() - t)
    outs = {}
    for d in DIRS:
        overlap, precision = _map_confidence(confidence=preds[f"confidence_{d}"].float(),
                                             threshold=model.threshold)
        outs[f"warp_{d}"] = preds[f"warp_{d}"][0].float().cpu().numpy()
        outs[f"overlap_{d}"] = overlap[0].cpu().numpy()
        outs[f"precision_{d}"] = precision[0].cpu().numpy()
    return outs, times


def reference(setting: str, out: str, runs: int) -> None:
    import torch
    from export_onnx import build_model
    from romav2.romav2 import RoMaV2

    print(f"[reference] building checkpoints for setting '{setting}' ...", flush=True)
    fp32 = build_model(setting, force_cpu=False).model.eval()      # export config
    dev = next(fp32.parameters()).device
    upstream = RoMaV2(RoMaV2.Cfg(setting=setting, compile=False))   # upstream defaults
    upstream.bidirectional = True
    upstream = upstream.to(dev).eval()
    torch.set_float32_matmul_precision("highest")
    print(f"[reference] device {dev}, cuda {torch.version.cuda}, torch {torch.__version__}", flush=True)

    result: dict[str, np.ndarray] = {}
    timing: dict[str, list[float]] = {"upstream": [], "fp32": []}
    names = []
    for pname, A_img, B_img in make_samples():
        names.append(pname)
        A, B = to_tensor(A_img), to_tensor(B_img)
        for tag, model in (("upstream", upstream), ("fp32", fp32)):
            outs, times = _run_model(model, A, B, runs)
            timing[tag].extend(times)
            for k, v in outs.items():
                result[f"{tag}/{pname}/{k}"] = v
            print(f"[reference] {pname:28s} {tag:8s} median {statistics.median(times)*1000:6.0f} ms "
                  f"({runs} runs)", flush=True)
    torch_ms = {tag: statistics.median(t) * 1000 for tag, t in timing.items()}
    for tag, ms in torch_ms.items():
        print(f"[reference] torch {tag} forward: median {ms:.0f} ms over {len(timing[tag])} runs")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, _meta=json.dumps({"setting": setting, "names": names, "torch_ms": torch_ms,
                                    "runs": runs, "device": str(dev)}), **result)
    print(f"[reference] saved {out} ({Path(out).stat().st_size / 1e6:.0f} MB)")


# ── accuracy metric ──────────────────────────────────────────────────────────

def compare(ref: dict[str, np.ndarray], got: dict[str, np.ndarray], S: int) -> dict[str, float]:
    """Overlap-masked differences (mask = reference overlap > 0.5)."""
    mask = ref["overlap"][..., 0] > 0.5
    dw = np.abs(ref["warp"] - got["warp"]).max(-1)
    do = np.abs(ref["overlap"][..., 0] - got["overlap"][..., 0])
    pa = ref["precision"].reshape(*mask.shape, 4)
    pb = got["precision"].reshape(*mask.shape, 4)
    rel = np.abs(pa - pb).max(-1) / (np.abs(pa).max(-1) + 1e-6)
    n = int(mask.sum())
    r = {
        "overlap_frac": float(mask.mean()), "confident_px": n,
        "warp_median_all": float(np.median(dw)), "warp_max_all": float(dw.max()),
        "overlap_mean_diff": float(do.mean()), "overlap_max_diff": float(do.max()),
    }
    if n:
        wm, rm = dw[mask], rel[mask]
        r.update(
            warp_p99_conf=float(np.percentile(wm, 99)), warp_max_conf=float(wm.max()),
            warp_p99_conf_px=float(np.percentile(wm, 99) * S / 2), warp_max_conf_px=float(wm.max() * S / 2),
            prec_rel_median_conf=float(np.median(rm)), prec_rel_p99_conf=float(np.percentile(rm, 99)),
        )
    return r


# ── Triton helpers ───────────────────────────────────────────────────────────

def _infer_dense(client, model: str, a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    import tritonclient.http as httpclient
    inputs = []
    for name, arr in (("img_A", a), ("img_B", b)):
        inp = httpclient.InferInput(name, arr.shape, "FP32")
        inp.set_data_from_numpy(arr)
        inputs.append(inp)
    wanted = [f"{k}_{d}" for d in DIRS for k in KEYS]
    r = client.infer(model_name=model, inputs=inputs,
                     outputs=[httpclient.InferRequestedOutput(n) for n in wanted])
    return {n: r.as_numpy(n)[0] for n in wanted}


def _infer_sampled(client, model: str, a: np.ndarray, b: np.ndarray, num_corresp: int, seed: int):
    import tritonclient.http as httpclient
    inputs = []
    for name, arr in (("img_A", a), ("img_B", b)):
        inp = httpclient.InferInput(name, arr.shape, "FP32")
        inp.set_data_from_numpy(arr)
        inputs.append(inp)
    for name, val in (("num_corresp", num_corresp), ("seed", seed)):
        arr = np.array([val], dtype=np.int64)
        inp = httpclient.InferInput(name, arr.shape, "INT64")
        inp.set_data_from_numpy(arr)
        inputs.append(inp)
    wanted = ["sampled_matches", "sampled_confidence", "sampled_precision_A", "sampled_precision_B"]
    r = client.infer(model_name=model, inputs=inputs,
                     outputs=[httpclient.InferRequestedOutput(n) for n in wanted])
    return {n: r.as_numpy(n) for n in wanted}


def _stats(client, model: str) -> dict[str, int]:
    """Cumulative nanoseconds per stage for a model (Triton per-model statistics)."""
    st = client.get_inference_statistics(model_name=model)["model_stats"][0]["inference_stats"]
    return {k: int(st.get(k, {}).get("ns", 0)) for k in STAT_KEYS}


def _delta_ms(before: dict[str, int], after: dict[str, int]) -> dict[str, float]:
    return {k: (after[k] - before[k]) / 1e6 for k in before}


def _median(xs) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


# ── triton phase ─────────────────────────────────────────────────────────────

def triton(setting: str, name: str, url: str, ref_path: str, out: str, *,
           runs: int, sampled_runs: int, num_corresps: list[int], seed: int) -> None:
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=url, network_timeout=600.0, connection_timeout=600.0)
    dense, sampled = model_names(name)
    S = INPUT_SIZES[setting]
    ref = np.load(ref_path)
    meta = json.loads(str(ref["_meta"]))
    if meta["setting"] != setting:
        raise SystemExit(f"{ref_path} holds setting '{meta['setting']}', not '{setting}'")
    samples = make_samples()
    inputs = {pname: (resize(to_tensor(A), S).numpy(), resize(to_tensor(B), S).numpy())
              for pname, A, B in samples}
    report = {"setting": setting, "url": url, "dense_model": dense, "sampled_model": sampled,
              "reference": meta, "accuracy": {}, "dense_timing": {}, "sampled_timing": {}}

    # ── accuracy ─────────────────────────────────────────────────────────────
    print(f"[accuracy] {dense} vs checkpoint, S={S}; warp in px of the {S}x{S} input, "
          "masked by reference overlap > 0.5", flush=True)
    print(f"  {'pair':28s} {'dir':3s} {'ovl%':>5s}  {'comparison':18s} "
          f"{'warp p99 px':>11s} {'warp max px':>11s} {'ovl mean':>9s} {'prec p99':>9s}")
    for pname, (a, b) in inputs.items():
        got = _infer_dense(client, dense, a, b)
        for d in DIRS:
            g = {k: got[f"{k}_{d}"] for k in KEYS}
            refs = {tag: {k: ref[f"{tag}/{pname}/{k}_{d}"] for k in KEYS} for tag in ("upstream", "fp32")}
            rows = {
                "triton_vs_upstream": compare(refs["upstream"], g, S),
                "triton_vs_fp32": compare(refs["fp32"], g, S),
                "fp32_vs_upstream": compare(refs["upstream"], refs["fp32"], S),   # AMP noise floor
            }
            report["accuracy"][f"{pname}/{d}"] = rows
            for i, (label, r) in enumerate(rows.items()):
                head = f"  {pname:28s} {d:3s} {r['overlap_frac']*100:5.1f}" if i == 0 else f"  {'':28s} {'':3s} {'':5s}"
                if "warp_max_conf_px" in r:
                    print(f"{head}  {label:18s} {r['warp_p99_conf_px']:11.2f} {r['warp_max_conf_px']:11.2f} "
                          f"{r['overlap_mean_diff']:9.4f} {r['prec_rel_p99_conf']:9.3f}", flush=True)
                else:
                    print(f"{head}  {label:18s} {'(no confident pixels)':>32s}", flush=True)

    # ── dense timing ─────────────────────────────────────────────────────────
    pname, (a, b) = next(iter(inputs.items()))
    wall, server, compute = [], [], []
    for i in range(runs + 1):                       # first request is a warm-up
        s0 = _stats(client, dense)
        t = time.perf_counter()
        _infer_dense(client, dense, a, b)
        w = time.perf_counter() - t
        dt = _delta_ms(s0, _stats(client, dense))
        if i:
            wall.append(w * 1000); server.append(dt["success"]); compute.append(dt["compute_infer"])
    report["dense_timing"] = {"pair": pname, "runs": runs, "wall_ms": wall, "server_ms": server,
                              "compute_ms": compute,
                              "median": {"wall_ms": _median(wall), "server_ms": _median(server),
                                         "compute_ms": _median(compute)}}
    print(f"\n[timing] {dense} on {pname}, median of {runs} runs (min..max):")
    print(f"  client wall  {_median(wall):7.0f} ms  ({min(wall):.0f}..{max(wall):.0f})")
    print(f"  server total {_median(server):7.0f} ms  ({min(server):.0f}..{max(server):.0f})")
    print(f"  compute      {_median(compute):7.0f} ms  ({min(compute):.0f}..{max(compute):.0f})")
    for tag, ms in meta["torch_ms"].items():
        print(f"  torch {tag:8s} {ms:7.0f} ms  (forward only, GPU, median of {meta['runs']} x 4 pairs)")

    # ── sampled timing ───────────────────────────────────────────────────────
    print(f"\n[timing] {sampled} on {pname}, median of {sampled_runs} runs per num_corresp:")
    print(f"  {'num_corresp':>11s} {'client wall':>11s} {'server':>8s} {'dense':>8s} {'sampler':>8s}  matches")
    for n in num_corresps:
        wall, server, dense_c, sampler_c, shapes = [], [], [], [], set()
        for i in range(sampled_runs + 1):
            s0 = {m: _stats(client, m) for m in (sampled, dense, "romav2_sampler")}
            t = time.perf_counter()
            outs = _infer_sampled(client, sampled, a, b, n, seed + i)
            w = time.perf_counter() - t
            d = {m: _delta_ms(s0[m], _stats(client, m)) for m in s0}
            shapes.add(tuple(outs["sampled_matches"].shape))
            if i:
                wall.append(w * 1000); server.append(d[sampled]["success"])
                dense_c.append(d[dense]["compute_infer"]); sampler_c.append(d["romav2_sampler"]["compute_infer"])
        report["sampled_timing"][str(n)] = {
            "wall_ms": wall, "server_ms": server, "dense_compute_ms": dense_c, "sampler_compute_ms": sampler_c,
            "matches_shape": sorted(shapes),
            "median": {"wall_ms": _median(wall), "server_ms": _median(server),
                       "dense_compute_ms": _median(dense_c), "sampler_compute_ms": _median(sampler_c)},
        }
        print(f"  {n:11d} {_median(wall):8.0f} ms {_median(server):8.0f} {_median(dense_c):8.0f} "
              f"{_median(sampler_c):8.0f}  {sorted(shapes)}", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=1))
    print(f"\nsaved {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("reference", help="checkpoint outputs + forward timing on the GPU")
    p.add_argument("--setting", default="precise", choices=list(INPUT_SIZES))
    p.add_argument("--out", default="bench_gpu/reference.npz")
    p.add_argument("--runs", type=int, default=5, help="timed forward passes per pair and config")
    p = sub.add_parser("triton", help="served module: accuracy vs reference, then timings")
    p.add_argument("--setting", default="precise", choices=list(INPUT_SIZES))
    p.add_argument("--name", default=DEFAULT_NAME, help="Triton module name (<name>_dense / <name>_sampled)")
    p.add_argument("--url", default="localhost:8000")
    p.add_argument("--ref", required=True, help="npz written by the reference phase")
    p.add_argument("--out", default="bench_gpu/triton_eval.json")
    p.add_argument("--runs", type=int, default=7, help="timed dense requests")
    p.add_argument("--sampled-runs", type=int, default=5, help="timed sampled requests per num_corresp")
    p.add_argument("--num-corresp", default="1000,3000,10000,30000,100000")
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.phase == "reference":
        reference(args.setting, args.out, args.runs)
    else:
        triton(args.setting, args.name, args.url, args.ref, args.out, runs=args.runs,
               sampled_runs=args.sampled_runs, seed=args.seed,
               num_corresps=[int(x) for x in args.num_corresp.split(",")])


if __name__ == "__main__":
    main()
