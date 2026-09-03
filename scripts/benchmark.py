"""Benchmark RoMaV2: PyTorch vs ONNX Runtime on a few image pairs.

One engine per invocation, so each run has its own process memory and thread
pool and a slow/large engine cannot skew another.  Every run writes
<out-dir>/<engine>.json (timings) and <out-dir>/<engine>.npz (outputs) so the
engines can be diffed afterwards with --report.

Usage:
    python scripts/benchmark.py --engine torch-cpu --setting precise --out-dir bench
    python scripts/benchmark.py --engine torch-mps --setting precise --out-dir bench
    python scripts/benchmark.py --engine onnx --onnx romav2_precise.onnx --setting precise --out-dir bench
    python scripts/benchmark.py --report bench

Samples: the repo only ships the Toronto pair, so the extra pairs are derived
from it with crops/rotations (synthetic viewpoint changes).  Timing does not
depend on image content; the PyTorch-vs-ONNX diffs in --report do, which is
why real photographs are used rather than noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# (H_lr, H_hr) per setting — mirrors RoMaV2.apply_setting.
RESOLUTIONS = {"turbo": (320, None), "fast": (512, None),
               "base": (640, None), "precise": (800, 1280)}


# ── samples ──────────────────────────────────────────────────────────────────

def make_samples() -> list[tuple[str, Image.Image, Image.Image]]:
    A = Image.open(ROOT / "assets/toronto_A.jpg").convert("RGB")
    B = Image.open(ROOT / "assets/toronto_B.jpg").convert("RGB")

    def crop_rot(img: Image.Image, frac: float, angle: float, dx: float = 0.5, dy: float = 0.5):
        w, h = img.size
        cw, ch = int(w * frac), int(h * frac)
        x0, y0 = int((w - cw) * dx), int((h - ch) * dy)
        return img.rotate(angle, resample=Image.BICUBIC).crop((x0, y0, x0 + cw, y0 + ch))

    return [
        ("toronto_A->B", A, B),
        ("toronto_B->A", B, A),
        ("toronto_A->A_crop70_rot15", A, crop_rot(A, 0.70, 15.0)),
        ("toronto_B->B_crop60_rot-10", B, crop_rot(B, 0.60, -10.0, dx=0.2, dy=0.3)),
    ]


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return arr[None]  # 1 3 H W


def resize(img: torch.Tensor, size: int) -> torch.Tensor:
    # Same resize RoMaV2.match() applies (bicubic + antialias), done on CPU.
    return F.interpolate(img, size=(size, size), mode="bicubic",
                         align_corners=False, antialias=True)


def prepare_inputs(img_A: Image.Image, img_B: Image.Image, setting: str) -> list[torch.Tensor]:
    H_lr, H_hr = RESOLUTIONS[setting]
    tA, tB = to_tensor(img_A), to_tensor(img_B)
    inputs = [resize(tA, H_lr), resize(tB, H_lr)]
    if H_hr is not None:
        inputs += [resize(tA, H_hr), resize(tB, H_hr)]
    return inputs


# ── engines ──────────────────────────────────────────────────────────────────

class TorchEngine:
    def __init__(self, setting: str, device: str):
        from export_onnx import build_model, io_spec
        self.device = device
        self.wrapper = build_model(setting, force_cpu=(device == "cpu"))
        dev = next(self.wrapper.parameters()).device
        if dev.type != device:
            raise SystemExit(f"Requested device {device} but model landed on {dev}")
        self.input_names, self.output_names, _, _ = io_spec(self.wrapper)
        self.info = {"device": str(dev), "torch": torch.__version__,
                     "threads": torch.get_num_threads()}

    def _sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def run(self, inputs: list[torch.Tensor]) -> tuple[float, list[np.ndarray]]:
        inputs = [x.to(self.device) for x in inputs]
        self._sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            outs = self.wrapper(*inputs)
        self._sync()
        dt = time.perf_counter() - t0
        return dt, [o.detach().cpu().numpy() for o in outs]


class OnnxEngine:
    def __init__(self, onnx_path: str, providers: list[str]):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(onnx_path, so, providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.info = {"onnxruntime": ort.__version__, "providers": self.sess.get_providers(),
                     "path": str(onnx_path)}

    def run(self, inputs: list[torch.Tensor]) -> tuple[float, list[np.ndarray]]:
        feed = {n: x.numpy() for n, x in zip(self.input_names, inputs)}
        t0 = time.perf_counter()
        outs = self.sess.run(None, feed)
        return time.perf_counter() - t0, outs


# ── run ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = make_samples()

    t0 = time.perf_counter()
    if args.engine == "onnx":
        if not args.onnx:
            raise SystemExit("--onnx PATH is required for --engine onnx")
        engine = OnnxEngine(args.onnx, providers=[args.provider])
    else:
        engine = TorchEngine(args.setting, device=args.engine.split("-", 1)[1])
    load_s = time.perf_counter() - t0
    print(f"[{args.engine}] loaded in {load_s:.1f}s  {engine.info}")

    n_in = 4 if RESOLUTIONS[args.setting][1] is not None else 2
    if len(engine.input_names) != n_in:
        raise SystemExit(f"{args.engine} expects {len(engine.input_names)} inputs "
                         f"{engine.input_names}, setting '{args.setting}' provides {n_in}")

    # Warm-up (first call pays for allocator growth / kernel selection).
    inputs = prepare_inputs(samples[0][1], samples[0][2], args.setting)
    warm_s, _ = engine.run(inputs)
    print(f"[{args.engine}] warm-up ({samples[0][0]}): {warm_s:.2f}s")

    times: dict[str, float] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, img_A, img_B in samples:
        inputs = prepare_inputs(img_A, img_B, args.setting)
        dt, outs = engine.run(inputs)
        times[name] = dt
        for oname, arr in zip(engine.output_names, outs):
            outputs[f"{name}/{oname}"] = arr.astype(np.float32)
        shapes = {o: tuple(a.shape) for o, a in zip(engine.output_names, outs)}
        print(f"[{args.engine}] {name}: {dt:.2f}s  {shapes}")

    result = {
        "engine": args.engine, "setting": args.setting, "info": engine.info,
        "load_s": load_s, "warmup_s": warm_s, "times_s": times,
        "mean_s": float(np.mean(list(times.values()))),
        "output_names": engine.output_names,
    }
    (out_dir / f"{args.engine}.json").write_text(json.dumps(result, indent=2))
    np.savez(out_dir / f"{args.engine}.npz", **outputs)
    print(f"[{args.engine}] mean {result['mean_s']:.2f}s over {len(times)} samples "
          f"→ {out_dir / (args.engine + '.json')}")


# ── report ───────────────────────────────────────────────────────────────────

def report(out_dir: str, reference: str = "torch-cpu") -> None:
    out_dir = Path(out_dir)
    results = {p.stem: json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))}
    if not results:
        raise SystemExit(f"no *.json results in {out_dir}")
    engines = list(results)
    samples = list(next(iter(results.values()))["times_s"])
    setting = next(iter(results.values()))["setting"]

    print(f"\nSetting: {setting}   (seconds per image pair, batch 1)\n")
    w = max(len(s) for s in samples + ["sample", "mean", "warm-up (cold)", "load"])
    print(f"{'sample':<{w}}  " + "  ".join(f"{e:>10}" for e in engines))
    for s in samples:
        print(f"{s:<{w}}  " + "  ".join(f"{results[e]['times_s'].get(s, float('nan')):>10.2f}" for e in engines))
    print(f"{'mean':<{w}}  " + "  ".join(f"{results[e]['mean_s']:>10.2f}" for e in engines))
    print(f"{'warm-up (cold)':<{w}}  " + "  ".join(f"{results[e]['warmup_s']:>10.2f}" for e in engines))
    print(f"{'load':<{w}}  " + "  ".join(f"{results[e]['load_s']:>10.2f}" for e in engines))
    for e in engines:
        print(f"  {e}: {results[e]['info']}")

    ref_npz = out_dir / f"{reference}.npz"
    if reference in results and ref_npz.exists():
        ref = np.load(ref_npz)
        print(f"\nMax |diff| vs {reference} (over all samples):")
        for e in engines:
            if e == reference or not (out_dir / f"{e}.npz").exists():
                continue
            other = np.load(out_dir / f"{e}.npz")
            per_out = {}
            for key in ref.files:
                if key in other.files:
                    oname = key.split("/", 1)[1]
                    d = float(np.abs(ref[key] - other[key]).max())
                    per_out[oname] = max(per_out.get(oname, 0.0), d)
            print(f"  {e}: " + ", ".join(f"{k}={v:.4f}" for k, v in per_out.items()))


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RoMaV2 PyTorch vs ONNX Runtime")
    parser.add_argument("--engine", choices=["torch-cpu", "torch-mps", "torch-cuda", "onnx"])
    parser.add_argument("--setting", default="precise", choices=list(RESOLUTIONS))
    parser.add_argument("--onnx", help="path to .onnx (for --engine onnx)")
    parser.add_argument("--provider", default="CPUExecutionProvider",
                        help="ORT execution provider (for --engine onnx)")
    parser.add_argument("--out-dir", default="bench", help="where to write <engine>.json/.npz")
    parser.add_argument("--report", metavar="DIR", help="print a table from a results dir and exit")
    args = parser.parse_args()

    if args.report:
        report(args.report)
    elif args.engine:
        run(args)
    else:
        parser.error("either --engine or --report is required")
