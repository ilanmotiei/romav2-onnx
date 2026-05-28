"""Export RoMaV2 to ONNX.

Usage:
    # Fast setting (512x512, unidirectional, no HR) - simplest graph:
    python scripts/export_onnx.py

    # Fast setting with both A->B and B->A outputs:
    python scripts/export_onnx.py --bidirectional --output romav2_fast_bidir.onnx

    # Validate an already-exported model:
    python scripts/export_onnx.py --validate romav2_fast.onnx

Exported inputs  (float32, values in [0, 1]):
    img_A  (B, 3, H, W)
    img_B  (B, 3, H, W)

Exported outputs:
    warp_AB     (B, H, W, 2)   — dense warp in normalized coords [-1, 1]
    overlap_AB  (B, H, W, 1)   — overlap probability in [0, 1]

With --bidirectional, the export also includes:
    warp_BA     (B, H, W, 2)
    overlap_BA  (B, H, W, 1)

With --include-precision, the export also includes:
    precision_AB  (B, H, W, 2, 2)
    precision_BA  (B, H, W, 2, 2)  # when combined with --bidirectional
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
    """Thin wrapper that exposes a flat ONNX-friendly interface.

    The underlying RoMaV2.forward returns an OrderedDict with many
    intermediate tensors and potential None values, neither of which
    are valid ONNX outputs.  This wrapper extracts only the final
    outputs and returns them as a plain tuple.
    """

    def __init__(
        self,
        model: RoMaV2,
        *,
        bidirectional_outputs: bool = False,
        precision_outputs: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.bidirectional_outputs = bidirectional_outputs
        self.precision_outputs = precision_outputs

    def forward(
        self, img_A: torch.Tensor, img_B: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        preds = self.model(img_A, img_B)
        warp_AB = preds["warp_AB"]
        confidence_AB = preds["confidence_AB"]
        overlap_AB, precision_AB = _map_confidence(confidence=confidence_AB, threshold=None)
        if not self.bidirectional_outputs:
            if self.precision_outputs:
                return warp_AB, overlap_AB, precision_AB
            return warp_AB, overlap_AB

        warp_BA = preds["warp_BA"]
        confidence_BA = preds["confidence_BA"]
        if warp_BA is None or confidence_BA is None:
            raise RuntimeError(
                "Bidirectional export requested, but model did not produce B->A outputs"
            )
        overlap_BA, precision_BA = _map_confidence(confidence=confidence_BA, threshold=None)
        if self.precision_outputs:
            return warp_AB, overlap_AB, precision_AB, warp_BA, overlap_BA, precision_BA
        return warp_AB, overlap_AB, warp_BA, overlap_BA


# ── Build ────────────────────────────────────────────────────────────────────

def _native_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(
    setting: str = "fast",
    *,
    force_cpu: bool = True,
    bidirectional: bool = False,
    include_precision: bool = False,
) -> RoMaV2OnnxWrapper:
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
    if bidirectional:
        model.bidirectional = True

    model.to(target_dev).float()

    model.eval()
    wrapper = RoMaV2OnnxWrapper(
        model,
        bidirectional_outputs=bidirectional,
        precision_outputs=include_precision,
    )
    wrapper.eval()
    return wrapper


# ── Export ───────────────────────────────────────────────────────────────────

def export(
    output_path: str = "romav2_fast.onnx",
    setting: str = "fast",
    opset: int = 17,
    bidirectional: bool = False,
    include_precision: bool = False,
) -> None:
    wrapper = build_model(
        setting,
        bidirectional=bidirectional,
        include_precision=include_precision,
    )

    # Input resolution is determined by the chosen setting.
    H, W = wrapper.model.H_lr, wrapper.model.W_lr
    dummy_A = torch.randn(1, 3, H, W)
    dummy_B = torch.randn(1, 3, H, W)

    output_names = ["warp_AB", "overlap_AB"]
    if include_precision:
        output_names.append("precision_AB")
    if bidirectional:
        output_names.extend(["warp_BA", "overlap_BA"])
        if include_precision:
            output_names.append("precision_BA")

    dynamic_axes = {
        "img_A": {0: "batch"},
        "img_B": {0: "batch"},
    }
    dynamic_axes.update({name: {0: "batch"} for name in output_names})

    print(
        f"Exporting with setting='{setting}', input {H}x{W}, "
        f"opset={opset}, bidirectional={bidirectional}, "
        f"include_precision={include_precision} ..."
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_A, dummy_B),
            output_path,
            input_names=["img_A", "img_B"],
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

def validate(
    onnx_path: str,
    setting: str = "fast",
    atol: float = 2e-2,
    bidirectional: bool = False,
    include_precision: bool = False,
) -> None:
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
    # ← potential hang: downloading 1 GB weights from GitHub
    if sys.gettrace() is not None: breakpoint()  # inspect `setting`
    wrapper = build_model(
        setting,
        force_cpu=True,
        bidirectional=bidirectional,
        include_precision=include_precision,
    )
    print(f"      Done in {time.time() - t0:.1f}s  (model device: cpu)")

    H, W = wrapper.model.H_lr, wrapper.model.W_lr
    img_A = torch.rand(1, 3, H, W)
    img_B = torch.rand(1, 3, H, W)
    print(f"      Input resolution: {H}x{W}, dtype={img_A.dtype}, device={img_A.device}")

    # ── Step 2: PyTorch forward pass ─────────────────────────────────────────
    print(f"[2/5] Running PyTorch forward pass on CPU ...")
    t0 = time.time()
    # ← potential hang: CPU inference of a large model is slow (~same wall time
    #   as the ONNX CPU pass below; unavoidable for CPU-only comparison)
    if sys.gettrace() is not None: breakpoint()  # inspect `wrapper` before the forward pass
    with torch.no_grad():
        pt_outputs = wrapper(img_A, img_B)
    output_names = ["warp_AB", "overlap_AB"]
    if include_precision:
        output_names.append("precision_AB")
    if bidirectional:
        output_names.extend(["warp_BA", "overlap_BA"])
        if include_precision:
            output_names.append("precision_BA")
    pt_outputs_np = {
        name: value.numpy() for name, value in zip(output_names, pt_outputs)
    }
    print(f"      Done in {time.time() - t0:.1f}s")
    for name, value in pt_outputs_np.items():
        print(
            f"      pt_{name}: shape={value.shape}, "
            f"min={value.min():.4f}, max={value.max():.4f}"
        )

    # Inputs are already on CPU — just convert to numpy for ORT.
    img_A_np = img_A.numpy()
    img_B_np = img_B.numpy()

    # ── Step 3: load ONNX session ─────────────────────────────────────────────
    print(f"[3/5] Loading ONNX model from {onnx_path} ...")
    t0 = time.time()
    # ← potential hang: constant-folding on a large graph can be slow
    if sys.gettrace() is not None: breakpoint()  # inspect graph before ORT loads it
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 0       # verbose ORT logs
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    print(f"      Done in {time.time() - t0:.1f}s")
    print(f"      ORT inputs:  {[i.name for i in sess.get_inputs()]}")
    print(f"      ORT outputs: {[o.name for o in sess.get_outputs()]}")

    # ── Step 4: ONNX inference ────────────────────────────────────────────────
    print("[4/5] Running ONNX inference (CPU) ...")
    t0 = time.time()
    # ← potential hang: CPU inference of a large model is slow (~same wall time
    #   as the PyTorch CPU pass above; unavoidable for ONNX runtime on CPU)
    if sys.gettrace() is not None: breakpoint()  # inspect `sess` and inputs before inference
    onnx_values = sess.run(
        None, {"img_A": img_A_np, "img_B": img_B_np}
    )
    onnx_outputs = {name: value for name, value in zip(output_names, onnx_values)}
    print(f"      Done in {time.time() - t0:.1f}s")
    for name, value in onnx_outputs.items():
        print(
            f"      onnx_{name}: shape={value.shape}, "
            f"min={value.min():.4f}, max={value.max():.4f}"
        )

    # ── Step 5: compare outputs ───────────────────────────────────────────────
    print(f"[5/5] Comparing outputs (atol={atol}) ...")
    if sys.gettrace() is not None: breakpoint()  # inspect output arrays before assert_allclose
    for name in output_names:
        np.testing.assert_allclose(
            pt_outputs_np[name], onnx_outputs[name], rtol=1e-3, atol=atol,
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
                        choices=["turbo", "fast", "base"],
                        help="model setting (determines input resolution)")
    parser.add_argument("--opset",    type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--bidirectional", action="store_true",
                        help="export both A->B and B->A dense warp/overlap outputs")
    parser.add_argument("--include-precision", action="store_true",
                        help="also export precision matrices used by RoMaV2.sample")
    args = parser.parse_args()

    if args.validate:
        validate(
            args.validate,
            setting=args.setting,
            bidirectional=args.bidirectional,
            include_precision=args.include_precision,
        )
    else:
        export(
            args.output,
            setting=args.setting,
            opset=args.opset,
            bidirectional=args.bidirectional,
            include_precision=args.include_precision,
        )
