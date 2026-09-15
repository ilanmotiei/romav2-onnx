"""Export RoMaV2 to ONNX with one interface for every setting.

Every export exposes the same Triton module; settings differ only in the
input size S (320 turbo / 512 fast / 640 base / 1280 precise):

    inputs   img_A, img_B   float32 (B, 3, S, S), values in [0, 1]
    outputs  warp_AB        (B, S, S, 2)     dense warp, normalized coords [-1, 1]
             overlap_AB     (B, S, S, 1)     overlap probability in [0, 1]
             precision_AB   (B, S, S, 2, 2)  precision matrices (RoMaV2.sample)
             warp_BA, overlap_BA, precision_BA   (same, B -> A)

Two-stage settings (precise) take the 1280x1280 image and derive the 800x800
low-res pass INSIDE the graph -- the same antialiased bicubic filter
RoMaV2.match() applies, expressed as a fixed separable gather (see _Resize) so
that every ONNX Runtime provider computes it exactly -- and the client sends one
image per side like every other setting.

The export writes the model straight into the Triton model repository and, in
the same pass, the config.pbtxt of the dense model and of the sampled ensemble
for that input size (see scripts/triton_configs.py), so the repository always
describes the model that was exported last:

    triton/model_repository/romav2_bidirectional_dense/1/model.onnx
    triton/model_repository/romav2_bidirectional_dense/config.pbtxt
    triton/model_repository/romav2_bidirectional_sampled/config.pbtxt

Usage:
    python scripts/export_onnx.py --setting base                  # -> model repository
    python scripts/export_onnx.py --setting precise               # same module, 1280 input
    python scripts/export_onnx.py --setting fast --output romav2_fast.onnx   # elsewhere; configs untouched

    # TensorRT-ready (bakes RoPE -> no If/Range so TensorRT can parse the graph):
    python scripts/export_onnx.py --setting base --trt

    # Validate an exported model against PyTorch on CPU:
    python scripts/export_onnx.py --validate romav2_fast.onnx --setting fast
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import romav2 — __init__.py eagerly loads all submodules, which bind the
# global `device` variable (auto-detected as CUDA/MPS/CPU) in their own
# module namespaces via `from romav2.device import device`.
from romav2.features import Descriptor, FineFeatures
from romav2.matcher import Matcher
from romav2.refiner import Refiners
from romav2.romav2 import RoMaV2, _map_confidence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triton_configs import DEFAULT_NAME, REPO as TRITON_REPO, SIZES, model_names, write_triton_configs

_CPU = torch.device("cpu")

INPUT_NAMES = ["img_A", "img_B"]
OUTPUT_NAMES = ["warp_AB", "overlap_AB", "precision_AB",
                "warp_BA", "overlap_BA", "precision_BA"]


# ── Wrapper ──────────────────────────────────────────────────────────────────

class _Resize(nn.Module):
    """RoMaV2.match()'s antialiased bicubic downscale as a fixed separable gather.

    The filter taps are read off torch's own ``F.interpolate(mode="bicubic",
    antialias=True)`` (PIL-style cubic, coefficient -0.5), so the eager result
    equals the upstream call to float precision and the traced graph carries
    Gather / Mul / ReduceSum with constant indices and weights instead of a
    Resize node.  That matters because ONNX Runtime 1.22's CUDA kernel for
    ``Resize(antialias=1)`` -- the runtime inside Triton 25.07 -- returns wrong
    pixels (mean |error| 0.19 on [0, 1] images; fixed by 1.26), whereas Gather
    and elementwise ops agree across providers to 1e-7.
    """

    def __init__(self, in_hw: tuple[int, int], out_hw: tuple[int, int]):
        super().__init__()
        self.size = tuple(out_hw)
        for axis, n_in, n_out in (("h", in_hw[0], out_hw[0]), ("w", in_hw[1], out_hw[1])):
            idx, wts = _resize_taps(n_in, n_out)
            self.register_buffer(f"idx_{axis}", idx, persistent=False)
            self.register_buffer(f"w_{axis}", wts, persistent=False)

    @staticmethod
    def _along_last(x: torch.Tensor, idx: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        n_out, K = idx.shape
        taps = x.index_select(-1, idx.reshape(-1)).unflatten(-1, (n_out, K))   # (..., n_out, K)
        return (taps * wts).sum(-1)                                            # (..., n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:              # (B, C, H_in, W_in)
        x = self._along_last(x, self.idx_w, self.w_w)                # (B, C, H_in, W_out)
        x = self._along_last(x.transpose(-1, -2), self.idx_h, self.w_h)   # (B, C, W_out, H_out)
        return x.transpose(-1, -2).contiguous()                      # (B, C, H_out, W_out)


def _resize_taps(n_in: int, n_out: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices and weights (n_out, K) of torch's antialiased bicubic 1-D resample
    from n_in to n_out samples:  out[r] = sum_k w[r, k] * in[idx[r, k]]."""
    # Resample an identity image along one axis only (the other axis keeps its
    # size, which is an exact identity): the result IS the filter matrix.  A
    # 1-pixel-wide basis would not do -- torch skips its antialias kernel then.
    eye = torch.eye(n_in)[None, None]                                # (1, 1, n_in, n_in)
    W = F.interpolate(eye, size=(n_out, n_in), mode="bicubic",
                      align_corners=False, antialias=True)[0, 0]    # (n_out, n_in)
    nz = W != 0
    K = int(nz.sum(1).max())
    idx = torch.zeros(n_out, K, dtype=torch.long)
    wts = torch.zeros(n_out, K)
    for r in range(n_out):
        cols = nz[r].nonzero().flatten()
        idx[r, : len(cols)] = cols
        wts[r, : len(cols)] = W[r, cols]
    return idx, wts


class RoMaV2OnnxWrapper(nn.Module):
    """Expose RoMaV2 as the flat, setting-independent Triton interface.

    The underlying RoMaV2.forward returns an OrderedDict with many intermediate
    tensors and potential None values, neither of which are valid ONNX outputs.
    This wrapper always runs both directions, takes one image per side at the
    setting's input size, and returns (warp, overlap, precision) x (AB, BA).
    Two-stage settings resize the input down to the low-res pass in-graph.
    """

    def __init__(self, model: RoMaV2) -> None:
        super().__init__()
        self.model = model
        self.two_stage = model.H_hr is not None
        if self.two_stage:
            self.resize_lr = _Resize((model.H_hr, model.W_hr), (model.H_lr, model.W_lr))
            self.input_hw = (model.H_hr, model.W_hr)
        else:
            self.input_hw = (model.H_lr, model.W_lr)

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.two_stage:
            preds = self.model(self.resize_lr(img_A), self.resize_lr(img_B),
                               img_A_hr=img_A, img_B_hr=img_B)
        else:
            preds = self.model(img_A, img_B)
        # Use the setting's threshold so the graph matches RoMaV2.match().
        # (None for turbo/fast/base/precise; 0.05 for the benchmark settings.)
        threshold = self.model.threshold
        overlap_AB, precision_AB = _map_confidence(confidence=preds["confidence_AB"], threshold=threshold)
        overlap_BA, precision_BA = _map_confidence(confidence=preds["confidence_BA"], threshold=threshold)
        return (preds["warp_AB"], overlap_AB, precision_AB,
                preds["warp_BA"], overlap_BA, precision_BA)


def io_spec(
    wrapper: RoMaV2OnnxWrapper,
) -> tuple[list[str], list[str], tuple[torch.Tensor, ...], dict[str, dict[int, str]]]:
    """(input_names, output_names, dummy_inputs, dynamic_axes) for a wrapper.

    Dummy inputs are CPU tensors at the setting's input size; only the batch
    axis is dynamic.
    """
    H, W = wrapper.input_hw
    dummies = (torch.randn(1, 3, H, W), torch.randn(1, 3, H, W))
    dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES}
    return list(INPUT_NAMES), list(OUTPUT_NAMES), dummies, dynamic_axes


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

    Every setting is exported bidirectional (the unified interface), so the
    B->A pass is switched on regardless of what the setting says upstream.
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
    model.bidirectional = True

    model.to(target_dev).float()

    model.eval()
    wrapper = RoMaV2OnnxWrapper(model)
    wrapper.eval()
    return wrapper


# ── TensorRT prep: bake RoPE to constants ─────────────────────────────────────

def bake_rope_for_trt(wrapper: RoMaV2OnnxWrapper, H: int, W: int) -> int:
    """Replace every RopePositionEmbedding.forward with a constant lookup.

    Why: RoMaV2's DINOv3 RoPE (src/romav2/vit/rope.py) computes sin/cos from the
    patch-grid (H, W) via ``torch.arange(...)`` + ``angles.tile(2)``.  The tracer
    emits these as a ``Range`` and an ``If`` with no static shape, and TensorRT's
    ONNX parser rejects the ``If`` ("has no shape specified"), so TRT cannot
    capture the backbone (whether via standalone trtexec or ONNX Runtime's
    TensorRT execution provider).

    The export pins the image resolution (dynamic_axes only covers batch), so the
    patch-grid (H, W) — and therefore sin/cos — is constant.  We run one dry
    forward to capture each RoPE module's (sin, cos) output, register them as
    buffers, and swap in a ``forward`` that just returns them.  Result: no
    Range/If, so TensorRT can parse the backbone — build the engine offline with
    trtexec (recommended for this ~1.4GB model) or via ORT's TensorRT EP.

    Numerically a no-op for the deployed fixed-resolution graph: RoPE output
    depends only on (H, W), never on image content.  GridSample and any other op
    TRT doesn't support fall back gracefully — no graph surgery needed.

    Works for two-stage settings too: RoPE only runs on the low-res pass (the
    high-res stage is VGG-only), so there is still a single grid to bake.

    Returns the number of RoPE modules baked.
    """
    # Match by class name, NOT isinstance: the matcher uses
    # romav2.vit.rope.RopePositionEmbedding, but the DINOv3 descriptor backbone
    # (loaded from torch.hub) carries its OWN dinov3.layers...RopePositionEmbedding
    # — identical code, different class object. Both emit the Range/Tile/If we must
    # bake, so we can't filter on a single imported class.
    rope_mods = [m for m in wrapper.modules()
                 if type(m).__name__ == "RopePositionEmbedding"]
    if not rope_mods:
        raise RuntimeError("No RopePositionEmbedding modules found — nothing to bake.")

    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    seen_hw: dict[int, tuple[int, int]] = {}

    def _make_hook(mod):
        def _hook(module, args, kwargs, output):
            hw = (int(kwargs["H"]), int(kwargs["W"]))
            if id(mod) in seen_hw and seen_hw[id(mod)] != hw:
                # A single module exercised at two resolutions can't be baked to
                # one constant — would only happen if the export stopped pinning
                # H/W. Fail loud rather than silently bake the wrong grid.
                raise RuntimeError(
                    f"RoPE module called at {hw} and {seen_hw[id(mod)]}; "
                    "cannot bake a constant. Keep the export resolution fixed."
                )
            seen_hw[id(mod)] = hw
            captured[id(mod)] = (output[0].detach().clone(), output[1].detach().clone())
        return _hook

    handles = [m.register_forward_hook(_make_hook(m), with_kwargs=True) for m in rope_mods]
    try:
        with torch.no_grad():
            wrapper(*io_spec(wrapper)[2])
    finally:
        for h in handles:
            h.remove()

    for m in rope_mods:
        if id(m) not in captured:
            raise RuntimeError(
                "A RoPE module was not exercised by the dry forward; cannot bake it."
            )
        sin, cos = captured[id(m)]
        m.register_buffer("_trt_sin", sin, persistent=False)
        m.register_buffer("_trt_cos", cos, persistent=False)

        def _const_forward(self, *, H, W):  # noqa: N803 — match the original signature
            return (self._trt_sin, self._trt_cos)

        m.forward = types.MethodType(_const_forward, m)

    grids = {hw for hw in seen_hw.values()}
    print(f"Baked RoPE for {len(rope_mods)} module(s) at patch grid(s) {sorted(grids)} "
          f"(image {H}x{W}) → no Range/If, backbone is TRT-parseable.")
    return len(rope_mods)


# ── Export ───────────────────────────────────────────────────────────────────

def default_output(setting: str, triton_name: str = DEFAULT_NAME) -> Path:
    """Where the export lands by default: the dense model's version directory."""
    return TRITON_REPO / model_names(triton_name)[0] / "1" / "model.onnx"


def export(
    output_path: str | Path | None = None,
    setting: str = "base",
    opset: int = 18,
    trt: bool = False,
    triton_name: str = DEFAULT_NAME,
    triton_configs: bool | None = None,
) -> Path:
    """Export one setting to ONNX and write the matching Triton configs.

    With ``output_path`` unset the model goes to the dense model's version
    directory in the Triton repository, and the dense and sampled config.pbtxt
    for this setting's input size are written in the same pass (the repository
    holds one module, so the last export wins).  A model written elsewhere
    with ``output_path`` leaves the repository's configs untouched, so they
    keep describing the model that sits beside them; ``triton_configs`` True
    forces the rewrite, False skips it even for a repository export.
    """
    output_path = Path(output_path) if output_path else default_output(setting, triton_name)
    in_repo = output_path.resolve() == default_output(setting, triton_name).resolve()
    if triton_configs is None:
        triton_configs = in_repo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = build_model(setting)
    H, W = wrapper.model.H_lr, wrapper.model.W_lr

    if trt:
        # Make the graph TensorRT-parseable so TRT can capture the backbone.
        bake_rope_for_trt(wrapper, H, W)

    input_names, output_names, dummies, dynamic_axes = io_spec(wrapper)

    res = f"{wrapper.input_hw[0]}x{wrapper.input_hw[1]}" + (
        f" (low-res pass {H}x{W} resized in-graph)" if wrapper.two_stage else "")
    print(f"Exporting setting='{setting}', input {res}, opset={opset}, trt={trt}\n"
          f"    inputs={input_names}\n    outputs={output_names} ...")

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

    if triton_configs:
        size = wrapper.input_hw[0]
        assert size == SIZES[setting], f"triton_configs.SIZES[{setting!r}] != exported size {size}"
        for path in write_triton_configs(setting, size, triton_name):
            print(f"Wrote  → {path}")
        if not in_repo:
            print(f"note: Triton expects the model at {default_output(setting, triton_name)}")
    elif not in_repo:
        print(f"note: model written outside the Triton repository; its configs were left "
              f"untouched (--triton-configs rewrites them for '{setting}')")
    return output_path


# ── Validate ─────────────────────────────────────────────────────────────────

ASSETS = Path(__file__).resolve().parents[1] / "assets"
DEFAULT_PAIR = (ASSETS / "toronto_A.jpg", ASSETS / "toronto_B.jpg")


def load_pair(input_hw, img_a=DEFAULT_PAIR[0], img_b=DEFAULT_PAIR[1]):
    """Load two images as float32 (1, 3, H, W) tensors in [0, 1], resized to
    ``input_hw`` the way RoMaV2.match() does it (antialiased bicubic straight
    from the original)."""
    from PIL import Image

    def load(path):
        x = torch.from_numpy(np.array(Image.open(path).convert("RGB")))
        x = x.permute(2, 0, 1).float()[None] / 255
        return F.interpolate(x, size=tuple(input_hw), mode="bicubic",
                             align_corners=False, antialias=True)

    return [load(img_a), load(img_b)]


def _compare_direction(d, pt, ox, *, warp_atol, overlap_atol, precision_rtol,
                       min_pixels=100):
    """Compare one direction ("AB" / "BA") of PyTorch vs ONNX outputs (batch
    element 0). Prints the statistics and returns a list of failure strings."""
    failures = []
    ovl = pt[f"overlap_{d}"][..., 0]
    mask = ovl > 0.5
    n = int(mask.sum())

    dw = np.abs(pt[f"warp_{d}"] - ox[f"warp_{d}"]).max(-1)
    do = np.abs(ovl - ox[f"overlap_{d}"][..., 0])
    pa = pt[f"precision_{d}"].reshape(*mask.shape, 4)
    pb = ox[f"precision_{d}"].reshape(*mask.shape, 4)
    rel = np.abs(pa - pb).max(-1) / (np.abs(pa).max(-1) + 1e-6)

    print(f"      {d}: PyTorch overlap > 0.5 on {mask.mean():.1%} of pixels ({n})")
    print(f"        warp      |diff| unmasked: median {np.median(dw):.5f}  "
          f"p99 {np.percentile(dw, 99):.4f}  max {dw.max():.4f}  (informational)")
    print(f"        overlap   |diff| mean {do.mean():.5f}  max {do.max():.4f}  "
          f"(mean limit {overlap_atol})")
    if do.mean() > overlap_atol:
        failures.append(f"{d}: overlap mean |diff| {do.mean():.4f} > {overlap_atol}")
    if n < min_pixels:
        print(f"        fewer than {min_pixels} confident pixels: warp/precision "
              "checks skipped for this direction")
        return failures
    wm, rm = dw[mask], rel[mask]
    print(f"        warp      |diff| in overlap>0.5: p99 {np.percentile(wm, 99):.4f}  "
          f"max {wm.max():.4f}  (max limit {warp_atol})")
    print(f"        precision rel diff in overlap>0.5: median {np.median(rm):.2e}  "
          f"p99 {np.percentile(rm, 99):.3f}  max {rm.max():.3f}  (p99 limit {precision_rtol})")
    if wm.max() > warp_atol:
        failures.append(f"{d}: warp max |diff| in confident pixels {wm.max():.4f} > {warp_atol}")
    if np.percentile(rm, 99) > precision_rtol:
        failures.append(f"{d}: precision rel diff p99 in confident pixels "
                        f"{np.percentile(rm, 99):.3f} > {precision_rtol}")
    return failures


def _require_finite(engine: str, names, arrays) -> None:
    """NaN/inf never pass: every masked check below is a `>` comparison, which is
    False for NaN, so a model emitting NaN would otherwise 'pass'."""
    bad = [n for n, a in zip(names, arrays) if not np.isfinite(a).all()]
    if bad:
        raise SystemExit(f"Validation FAILED: non-finite values in {engine} outputs {bad}")


def validate(onnx_path: str, setting: str = "fast", *,
             img_a=DEFAULT_PAIR[0], img_b=DEFAULT_PAIR[1],
             warp_atol: float = 0.05, overlap_atol: float = 0.01,
             precision_rtol: float = 0.3) -> None:
    """Compare an exported model with the PyTorch reference on a real image pair.

    Max-abs over the whole output is the wrong metric for this model: wherever
    the two images do not overlap, the matcher's softmax (temperature 0.1)
    turns float32 rounding into arbitrarily different warps and precisions --
    PyTorch CPU vs PyTorch CUDA differ just as much there.  Random inputs are
    unmatched everywhere, so they cannot be used either.  The check therefore
    runs the bundled Toronto pair, resized exactly like RoMaV2.match() does,
    and compares per direction:

      * warp       max |diff| where PyTorch overlap > 0.5    <= warp_atol
      * overlap    mean |diff| over the whole map            <= overlap_atol
      * precision  p99 relative diff where overlap > 0.5     <= precision_rtol

    Reference (precise, CPU): warp max 0.017, overlap mean 0.0002, precision
    p99 0.12.  A broken export (wrong resize, swapped outputs, ...) is off by
    0.3+ on the warp, so the limits leave a wide margin either way.
    """
    import time

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime is required for validation: pip install onnxruntime")

    # ── Step 1: build PyTorch model on CPU ───────────────────────────────────
    # ONNX Runtime runs on CPU here, so the reference pass runs on CPU too:
    # MPS/CUDA vs CPU float32 already diverge far beyond the tolerances.
    print("[1/5] Building PyTorch model on CPU (downloads weights on first run) ...")
    t0 = time.time()
    wrapper = build_model(setting, force_cpu=True)
    print(f"      Done in {time.time() - t0:.1f}s  (model device: cpu)")

    input_names, output_names, _, _ = io_spec(wrapper)
    inputs = load_pair(wrapper.input_hw, img_a, img_b)
    print(f"      Inputs: {Path(img_a).name}, {Path(img_b).name} -> "
          + ", ".join(f"{n}{tuple(x.shape)}" for n, x in zip(input_names, inputs)))

    # ── Step 2: PyTorch forward pass ─────────────────────────────────────────
    print("[2/5] Running PyTorch forward pass on CPU ...")
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
    ort_shape = [i.shape for i in sess.get_inputs()]
    print(f"      ORT inputs:  {list(zip(ort_in, ort_shape))}")
    print(f"      ORT outputs: {ort_out}")
    expected_shape = ["batch", 3, *wrapper.input_hw]
    if ort_in != input_names or ort_out != output_names or any(s != expected_shape for s in ort_shape):
        raise SystemExit(
            f"ONNX model IO {list(zip(ort_in, ort_shape))} -> {ort_out} does not match setting "
            f"'{setting}' ({input_names} @ {expected_shape} -> {output_names}); wrong --setting, "
            "or a model exported before the unified interface?"
        )

    # ── Step 4: ONNX inference ────────────────────────────────────────────────
    print("[4/5] Running ONNX inference (CPU) ...")
    t0 = time.time()
    onnx_outs = sess.run(None, {n: x.numpy() for n, x in zip(input_names, inputs)})
    print(f"      Done in {time.time() - t0:.1f}s")
    for name, arr in zip(output_names, onnx_outs):
        print(f"      onnx {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")

    # ── Step 5: compare outputs ───────────────────────────────────────────────
    print("[5/5] Comparing outputs (confidence-masked, see validate.__doc__) ...")
    _require_finite("PyTorch", output_names, pt_outs)
    _require_finite("ONNX", output_names, onnx_outs)
    pt = {n: a[0] for n, a in zip(output_names, pt_outs)}
    ox = {n: a[0] for n, a in zip(output_names, onnx_outs)}
    failures = []
    for d in ("AB", "BA"):
        failures += _compare_direction(d, pt, ox, warp_atol=warp_atol,
                                       overlap_atol=overlap_atol,
                                       precision_rtol=precision_rtol)
    if failures:
        raise SystemExit("Validation FAILED:\n  " + "\n  ".join(failures))
    print("Validation passed — PyTorch and ONNX agree wherever the images overlap.")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export or validate RoMaV2 ONNX model")
    parser.add_argument("--validate", metavar="ONNX_PATH",
                        help="path to .onnx file to validate (skips export)")
    parser.add_argument("--output",   default=None,
                        help="output .onnx path (default: the dense model's version "
                             "directory in triton/model_repository)")
    parser.add_argument("--setting",  default="base",
                        choices=["turbo", "fast", "base", "precise"],
                        help="model setting (fixes the input size; precise resizes "
                             "its 800x800 low-res pass in-graph from the 1280 input)")
    parser.add_argument("--opset",    type=int, default=18,
                        help="ONNX opset version")
    parser.add_argument("--trt", action="store_true",
                        help="bake RoPE to constants so the graph is "
                             "TensorRT-parseable (removes If/Range)")
    parser.add_argument("--triton-name", default=DEFAULT_NAME,
                        help=f"Triton module name: models <name>_dense and <name>_sampled "
                             f"(default {DEFAULT_NAME})")
    cfg = parser.add_mutually_exclusive_group()
    cfg.add_argument("--triton-configs", action="store_true",
                     help="(re)write the Triton config.pbtxt files even when --output "
                          "puts the model outside the repository")
    cfg.add_argument("--no-triton-configs", action="store_true",
                     help="export only; leave the Triton config.pbtxt files untouched")
    parser.add_argument("--img-a", default=str(DEFAULT_PAIR[0]),
                        help="(--validate) image A, resized like RoMaV2.match() does")
    parser.add_argument("--img-b", default=str(DEFAULT_PAIR[1]),
                        help="(--validate) image B")
    parser.add_argument("--warp-atol", type=float, default=0.05,
                        help="(--validate) max warp |diff| allowed where "
                             "PyTorch overlap > 0.5 (normalized coords)")
    # Every export now carries both directions and precision; these are kept so
    # older command lines and docs keep working.
    parser.add_argument("--bidirectional", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--include-precision", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.bidirectional or args.include_precision:
        print("note: --bidirectional / --include-precision are implied by the unified "
              "interface (every export has warp/overlap/precision for AB and BA).")

    if args.validate:
        validate(args.validate, setting=args.setting,
                 img_a=args.img_a, img_b=args.img_b, warp_atol=args.warp_atol)
    else:
        export(args.output, setting=args.setting, opset=args.opset, trt=args.trt,
               triton_name=args.triton_name,
               triton_configs=True if args.triton_configs else False if args.no_triton_configs else None)
