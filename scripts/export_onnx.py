"""Export RoMaV2 to ONNX.

Usage:
    # Fast setting (512x512, unidirectional, no HR) — simplest graph:
    python scripts/export_onnx.py

    # Precise setting (800 low-res + 1280 high-res stage, bidirectional):
    python scripts/export_onnx.py --setting precise --output romav2_precise.onnx

    # Validate an already-exported model:
    python scripts/export_onnx.py --validate romav2_fast.onnx
    python scripts/export_onnx.py --validate romav2_precise.onnx --setting precise

Exported inputs (float32, values in [0, 1]) — turbo / fast / base:
    img_A  (B, 3, H, W)
    img_B  (B, 3, H, W)

Exported outputs — turbo / fast / base:
    warp_AB     (B, H, W, 2)   — dense warp in normalized coords [-1, 1]
    overlap_AB  (B, H, W, 1)   — overlap probability in [0, 1]

Precise is two-stage and bidirectional, so it takes both resolutions
(the antialiased bicubic resize RoMaV2.match() uses has no ONNX symbolic,
so resizing stays on the client) and returns both directions plus the
2x2 precision matrices.  All spatial outputs are at the high resolution:
    inputs   img_A_lr, img_B_lr  (B, 3, 800, 800)
             img_A_hr, img_B_hr  (B, 3, 1280, 1280)
    outputs  warp_AB, overlap_AB, precision_AB      (B, 1280, 1280, 2 | 1 | 2x2)
             warp_BA, overlap_BA, precision_BA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import romav2 — __init__.py eagerly loads all submodules, which bind the
# global `device` variable (auto-detected as CUDA/MPS/CPU) in their own
# module namespaces via `from romav2.device import device`.
from romav2.features import Descriptor, FineFeatures
from romav2.matcher import Matcher
from romav2.refiner import Refiners
from romav2.romav2 import RoMaV2, _map_confidence

_CPU = torch.device("cpu")


# ── Wrapper ──────────────────────────────────────────────────────────────────

class RoMaV2OnnxWrapper(nn.Module):
    """Thin wrapper that exposes a flat tensor interface.

    The underlying RoMaV2.forward returns an OrderedDict with many
    intermediate tensors and potential None values, neither of which
    are valid ONNX outputs.  This wrapper extracts only the final
    outputs and returns them as a plain tuple.

    Unidirectional settings (turbo / fast / base):
        (img_A, img_B) -> (warp_AB, overlap_AB)
    Bidirectional, two-stage settings (precise):
        (img_A_lr, img_B_lr, img_A_hr, img_B_hr)
            -> (warp_AB, overlap_AB, precision_AB,
                warp_BA, overlap_BA, precision_BA)

    The branch is chosen by the model's setting, which is fixed at trace
    time, so the exported graph is straight-line either way.
    """

    def __init__(self, model: RoMaV2) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        img_A_lr: torch.Tensor,
        img_B_lr: torch.Tensor,
        img_A_hr: torch.Tensor | None = None,
        img_B_hr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        preds = self.model(img_A_lr, img_B_lr, img_A_hr=img_A_hr, img_B_hr=img_B_hr)
        # Use the setting's threshold so the graph matches RoMaV2.match().
        # (None for turbo/fast/base/precise; 0.05 for the benchmark settings.)
        threshold = self.model.threshold
        overlap_AB, precision_AB = _map_confidence(
            confidence=preds["confidence_AB"], threshold=threshold
        )
        if not self.model.bidirectional:
            return preds["warp_AB"], overlap_AB
        overlap_BA, precision_BA = _map_confidence(
            confidence=preds["confidence_BA"], threshold=threshold
        )
        return (
            preds["warp_AB"], overlap_AB, precision_AB,
            preds["warp_BA"], overlap_BA, precision_BA,
        )


def io_spec(
    wrapper: RoMaV2OnnxWrapper,
) -> tuple[list[str], list[str], tuple[torch.Tensor, ...], dict[str, dict[int, str]]]:
    """(input_names, output_names, dummy_inputs, dynamic_axes) for the wrapper's setting.

    Dummy inputs are CPU tensors at the setting's pinned resolution(s); only the
    batch axis is dynamic.
    """
    m = wrapper.model
    H, W = m.H_lr, m.W_lr
    if m.H_hr is None:
        input_names = ["img_A", "img_B"]
        dummies = (torch.randn(1, 3, H, W), torch.randn(1, 3, H, W))
    else:
        input_names = ["img_A_lr", "img_B_lr", "img_A_hr", "img_B_hr"]
        dummies = (
            torch.randn(1, 3, H, W), torch.randn(1, 3, H, W),
            torch.randn(1, 3, m.H_hr, m.W_hr), torch.randn(1, 3, m.H_hr, m.W_hr),
        )
    if m.bidirectional:
        output_names = ["warp_AB", "overlap_AB", "precision_AB",
                        "warp_BA", "overlap_BA", "precision_BA"]
    else:
        output_names = ["warp_AB", "overlap_AB"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    return input_names, output_names, dummies, dynamic_axes


# ── Build ────────────────────────────────────────────────────────────────────

def _native_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(setting: str = "fast", *, force_cpu: bool = True) -> RoMaV2OnnxWrapper:
    """Build the model wrapper.

    force_cpu=True  → patches all romav2 module device bindings to CPU and
                      moves the model to CPU.  Required for ONNX export/tracing.
    force_cpu=False → leaves the model on the native device (CUDA/MPS/CPU).
                      Use this for the PyTorch reference pass during validation
                      so that the accelerator is utilised and it does not stall.
    """
    torch.set_float32_matmul_precision("highest")

    # Patch every romav2 module's local `device` binding to the target device
    # BEFORE creating the model, so that RoMaV2.__init__'s self.to(device) and
    # all forward-time tensor creation (get_normalized_grid, scale_factor, etc.)
    # both target the same device.  This must happen on every call so that
    # switching between CPU (export) and native (validation) works correctly.
    target_dev = _CPU if force_cpu else _native_device()
    for _mod in sys.modules.values():
        if getattr(_mod, "__name__", "").startswith("romav2") and hasattr(_mod, "device"):
            _mod.device = target_dev

    cfg = RoMaV2.Cfg(
        # Disable AMP everywhere so the entire graph stays in float32 during
        # ONNX tracing.  AMP casts ops to bfloat16 which embeds bf16 Cast
        # nodes that ORT rejects with INVALID_GRAPH.
        descriptor=Descriptor.Cfg(enable_amp=False),
        matcher=Matcher.Cfg(enable_amp=False, pos_embed_rope_dtype="fp32"),  # covers mv_vit + DPTHead + RoPE
        refiners=Refiners.Cfg(enable_amp=False),     # covers ConvRefiner.Block
        refiner_features=FineFeatures.Cfg(enable_amp=False),  # covers VGG
        compile=False,
        setting=setting,
    )
    model = RoMaV2(cfg)

    model.to(target_dev).float()

    model.eval()
    wrapper = RoMaV2OnnxWrapper(model)
    wrapper.eval()
    return wrapper


# ── Export ───────────────────────────────────────────────────────────────────

def export(
    output_path: str = "romav2_fast.onnx",
    setting: str = "fast",
    opset: int = 17,
) -> None:
    wrapper = build_model(setting)

    # Input resolution is determined by the chosen setting.
    H, W = wrapper.model.H_lr, wrapper.model.W_lr

    input_names, output_names, dummies, dynamic_axes = io_spec(wrapper)

    res = f"{H}x{W}" + (f" + {wrapper.model.H_hr}x{wrapper.model.W_hr} hr"
                        if wrapper.model.H_hr is not None else "")
    print(f"Exporting with setting='{setting}', input {res}, opset={opset}, "
          f"inputs={input_names}, outputs={output_names} ...")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummies,
            output_path,
            input_names=input_names,
            output_names=output_names,
            # batch dimension is dynamic; H/W are static (baked into the graph).
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            # Use the classic JIT-trace exporter instead of Dynamo.
            # Dynamo is stricter about Python control flow; the trace-based
            # path handles the model's Python-driven loops more gracefully.
            dynamo=False,
        )

    print(f"Saved → {output_path}")


# ── Validate ─────────────────────────────────────────────────────────────────

def validate(onnx_path: str, setting: str = "fast", atol: float = 2e-2) -> None:
    import time

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime is required for validation: pip install onnxruntime")

    # ── Step 1: build PyTorch model on CPU ───────────────────────────────────
    # ONNX Runtime always runs on CPU.  To get numerically identical results
    # we must also run the PyTorch reference pass on CPU (force_cpu=True).
    # MPS vs CPU float32 diverge by ~0.23 in warp coords for a 24-layer ViT.
    print(f"[1/5] Building PyTorch model on CPU (downloads weights on first run) ...")
    t0 = time.time()
    wrapper = build_model(setting, force_cpu=True)
    print(f"      Done in {time.time() - t0:.1f}s  (model device: cpu)")

    input_names, output_names, dummies, _ = io_spec(wrapper)
    # Images are expected in [0, 1]; io_spec's randn dummies are only for tracing.
    inputs = [torch.rand_like(d) for d in dummies]
    print(f"      Inputs: " + ", ".join(f"{n}{tuple(x.shape)}" for n, x in zip(input_names, inputs)))

    # ── Step 2: PyTorch forward pass ─────────────────────────────────────────
    print(f"[2/5] Running PyTorch forward pass on CPU ...")
    t0 = time.time()
    with torch.no_grad():
        pt_outs = [o.numpy() for o in wrapper(*inputs)]
    print(f"      Done in {time.time() - t0:.1f}s")
    for name, arr in zip(output_names, pt_outs):
        print(f"      pt {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

    # ── Step 3: load ONNX session ─────────────────────────────────────────────
    print(f"[3/5] Loading ONNX model from {onnx_path} ...")
    t0 = time.time()
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    print(f"      Done in {time.time() - t0:.1f}s")
    ort_in = [i.name for i in sess.get_inputs()]
    ort_out = [o.name for o in sess.get_outputs()]
    print(f"      ORT inputs:  {ort_in}")
    print(f"      ORT outputs: {ort_out}")
    if ort_in != input_names or ort_out != output_names:
        raise SystemExit(
            f"ONNX model IO {ort_in} -> {ort_out} does not match setting "
            f"'{setting}' ({input_names} -> {output_names}); wrong --setting?"
        )

    # ── Step 4: ONNX inference ────────────────────────────────────────────────
    print("[4/5] Running ONNX inference (CPU) ...")
    t0 = time.time()
    onnx_outs = sess.run(None, {n: x.numpy() for n, x in zip(input_names, inputs)})
    print(f"      Done in {time.time() - t0:.1f}s")
    for name, arr in zip(output_names, onnx_outs):
        print(f"      onnx {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

    # ── Step 5: compare outputs ───────────────────────────────────────────────
    print(f"[5/5] Comparing outputs (atol={atol}) ...")
    for name, pt_arr, onnx_arr in zip(output_names, pt_outs, onnx_outs):
        print(f"      {name}: max|diff|={np.abs(pt_arr - onnx_arr).max():.5f}")
        np.testing.assert_allclose(
            pt_arr, onnx_arr, rtol=1e-3, atol=atol,
            err_msg=f"{name} mismatch between PyTorch and ONNX"
        )
    print("Validation passed — PyTorch and ONNX outputs match.")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export or validate RoMaV2 ONNX model")
    parser.add_argument("--validate", metavar="ONNX_PATH",
                        help="path to .onnx file to validate (skips export)")
    parser.add_argument("--output",   default="romav2_fast.onnx",
                        help="output .onnx path")
    parser.add_argument("--setting",  default="fast",
                        choices=["turbo", "fast", "base", "precise"],
                        help="model setting (determines input resolution)")
    parser.add_argument("--opset",    type=int, default=17,
                        help="ONNX opset version")
    args = parser.parse_args()

    if args.validate:
        validate(args.validate, setting=args.setting)
    else:
        export(args.output, setting=args.setting, opset=args.opset)
