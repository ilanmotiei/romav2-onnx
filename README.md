# RoMa v2 — ONNX & Triton Deployment

Export, validate, and serve the [RoMa v2](https://github.com/Parskatt/RoMaV2) dense image matcher as an ONNX model, with full support for local inference and NVIDIA Triton Inference Server deployment.

RoMa v2 produces a **dense warp** and **overlap confidence map** between any two images — useful for visual localization, 3D reconstruction, and image alignment.

---

## Example results

Input images (University of Toronto, large viewpoint change):

| Image A | Image B |
|---------|---------|
| ![img_A](assets/toronto_A.jpg) | ![img_B](assets/toronto_B.jpg) |

**Matching output** (PyTorch / ONNX / Triton — all identical):

![match result](assets/match_result.png)

> **Top-left:** Image A · **Top-center:** Image B · **Top-right:** Image B warped into A's frame
> **Bottom-left:** Overlap confidence (bright = high) · **Bottom-center:** Alpha blend · **Bottom-right:** Dense correspondences

---

## Requirements

```bash
git clone https://github.com/ilanmotiei/romav2-onnx
cd romav2-onnx
pip install ".[all]"
```

Or without the Triton client:

```bash
pip install .
```

The modified `romav2` source is embedded directly in `src/` — no separate clone or install step needed. It is a fork of [Parskatt/RoMaV2](https://github.com/Parskatt/RoMaV2) with the following changes required for ONNX export:
- `enable_amp` flag added to `Matcher`, `Refiners`, `FineFeatures`, `DPTHead` and `VGG` so AMP/bfloat16 can be disabled at export time
- `pos_embed_rope_dtype` added to `Matcher.Cfg` and `vit_from_name` to keep RoPE in float32
- `native_torch_local_corr` vectorised (removed Python `for` loop over batch) for dynamic-batch ONNX compatibility
- `@torch.inference_mode()` removed from `RoMaV2.forward` (blocks the JIT tracer)

**For Triton server:**
- Docker with the `nvcr.io/nvidia/tritonserver:23.12-py3` image (≈12 GB)

---

## 1 — Export to ONNX

Every setting exports the **same module**: one image per side in, both directions plus precision out. Settings differ only in the input size `S`.

```bash
python scripts/export_onnx.py --setting fast    --output romav2_fast.onnx      # 512×512
python scripts/export_onnx.py --setting base    --output romav2_base.onnx      # 640×640
python scripts/export_onnx.py --setting precise --output romav2_precise.onnx   # 1280×1280
```

| Setting   | Input size `S` | What runs                                                     |
|-----------|----------------|---------------------------------------------------------------|
| `turbo`   | 320×320        | DINOv3 + matcher + one refinement stage                       |
| `fast`    | 512×512        | same                                                          |
| `base`    | 640×640        | same                                                          |
| `precise` | 1280×1280      | 800×800 low-res pass (resized **in-graph**) + 1280 refinement |

**Inputs:** `img_A`, `img_B` — `float32 [B, 3, S, S]`, values in `[0, 1]`, resized from the original with an antialiased bicubic filter (PIL `BICUBIC`, or torch `bicubic` with `antialias=True`; that is what upstream `RoMaV2.match()` does).

**Outputs** (all at `S×S`):

| Name | Shape | Meaning |
|------|-------|---------|
| `warp_AB`, `warp_BA` | `[B, S, S, 2]` | dense warp, normalised coords in `[-1, 1]` |
| `overlap_AB`, `overlap_BA` | `[B, S, S, 1]` | overlap probability in `[0, 1]` |
| `precision_AB`, `precision_BA` | `[B, S, S, 2, 2]` | precision matrices consumed by RoMaV2's `sample()` |

The script:
- Disables AMP / bfloat16 everywhere so the full graph stays in float32 (required for ORT compatibility)
- Forces RoPE embeddings to float32
- Switches the B→A pass on for every setting (upstream only does so for `precise`)
- Uses the classic TorchScript-based JIT tracer (`dynamo=False`) at opset 18
- Bakes `S` into the graph; only the batch dimension is dynamic
- For `precise`, traces the 1280→800 resize as a plain bicubic `Resize` and then patches those two nodes to the antialiased PIL-style cubic (`antialias=1`, coefficient −0.5), which ONNX Runtime reproduces to ~1e-5 of torch's. This is why opset 18 is required.

Add `--trt` to bake the RoPE tables to constants so TensorRT can parse the backbone (see `bake_rope_for_trt` in the script). `--bidirectional` and `--include-precision` are still accepted for old command lines but are implied.

---

## 2 — Validate

Runs the PyTorch model and the exported ONNX model on the bundled Toronto pair (resized exactly like `RoMaV2.match()` does, on CPU for both) and compares them where the comparison is meaningful. Pass the same `--setting` used at export; the check refuses models with a different interface.

```bash
python scripts/export_onnx.py --validate romav2_fast.onnx    --setting fast
python scripts/export_onnx.py --validate romav2_precise.onnx --setting precise
# other images: --img-a path --img-b path; stricter/looser warp limit: --warp-atol
```

Per direction (A→B and B→A) the check asserts:

| quantity | region | limit |
|---|---|---|
| warp max abs diff | PyTorch `overlap > 0.5` | 0.05 (`--warp-atol`, normalized coords) |
| overlap mean abs diff | whole map | 0.01 |
| precision relative diff, p99 | PyTorch `overlap > 0.5` | 0.3 |

Expected output (precise, laptop CPU):
```
[1/5] Building PyTorch model on CPU ...       Done in 5.2s
      Inputs: toronto_A.jpg, toronto_B.jpg -> img_A(1, 3, 1280, 1280), img_B(1, 3, 1280, 1280)
[2/5] Running PyTorch forward pass on CPU ... Done in 56.2s
[3/5] Loading ONNX model from romav2_precise.onnx ... Done in 3.6s
[4/5] Running ONNX inference (CPU) ...        Done in 37.3s
[5/5] Comparing outputs (confidence-masked, see validate.__doc__) ...
      AB: PyTorch overlap > 0.5 on 54.6% of pixels (894780)
        warp      |diff| unmasked: median 0.00000  p99 0.0001  max 0.1559  (informational)
        overlap   |diff| mean 0.00016  max 0.0325  (mean limit 0.01)
        warp      |diff| in overlap>0.5: p99 0.0001  max 0.0121  (max limit 0.05)
        precision rel diff in overlap>0.5: median 2.11e-04  p99 0.020  max 0.166  (p99 limit 0.3)
      BA: PyTorch overlap > 0.5 on 1.4% of pixels (23203)
        ... warp max 0.0164, precision p99 0.119 ...
Validation passed — PyTorch and ONNX agree wherever the images overlap.
```

> **Why not a global tolerance?** Wherever the two images do not overlap, the matcher's softmax at temperature 0.1 turns float32 rounding into arbitrarily different warps and precisions: on real images the unmasked warp max is ~0.15–0.7 while the confident pixels agree to ~0.01, and PyTorch CPU vs PyTorch CUDA differ just as much in those regions. Random inputs are unmatched everywhere, so they cannot be used either. The precise export's in-graph antialiased Resize matches torch's to 1e-4 in the interior (2e-2 on the 4-pixel border, a known ORT-vs-torch edge-handling difference) and does not change the picture.
>
> **GPU memory.** The native local-correlation path used to gather all 49 window offsets of the patch-4 refiner in one tensor; at the precise input size that is two 3.85 GB intermediates, a ~20 GB standalone peak, and a CUDA out-of-memory inside Triton next to the other models on a 24 GB card. `src/romav2/local_correlation.py` now samples one offset at a time (bit-identical result, K static so the batch axis stays dynamic); standalone the ORT CUDA peak drops to 9.0 GB with arena shrinkage enabled as in the Triton config (14.7 GB without it, because ORT adds its memory-pattern buffer on the second run), and with the other dense model loaded Triton settles at about 14.6 GB after precise requests.
>
> Both passes run on CPU so the comparison is numerically equivalent; MPS vs CPU diverges by ~0.23 in warp coords for a 24-layer ViT. `scripts/benchmark.py --report` applies the same overlap mask across engines (CPU, MPS, CUDA, ONNX).

---

## 3 — Visualise (direct ONNX)

```bash
# Uses sample images included in this repo
python scripts/visualize.py --onnx romav2_fast.onnx --out result.png
python scripts/visualize.py --onnx romav2_precise.onnx --setting precise --out precise_result.png

# Or PyTorch model
python scripts/visualize.py --out result.png

# Custom images
python scripts/visualize.py \
    --img-a path/to/image_A.jpg \
    --img-b path/to/image_B.jpg \
    --onnx romav2_fast.onnx \
    --out result.png
```

The output stacks one 2×3 block per direction plus an error block:

| | Left | Center | Right |
|---|---|---|---|
| A→B row 1 | Image A | Image B | Image B warped into A |
| A→B row 2 | Overlap heatmap | Alpha blend | Dense correspondences (overlap > 0.5) |
| B→A | same, roles swapped | | |
| error | expected error A→B (px, from precision) | expected error B→A | legend |

![bidirectional ONNX result](assets/bidir_onnx_result.png)

---

## 4 — Benchmark

`scripts/benchmark.py` times one engine per process on four Toronto-derived pairs and saves outputs so engines can be diffed with an overlap mask:

```bash
python scripts/benchmark.py --engine torch-cpu --setting precise --out-dir bench
python scripts/benchmark.py --engine onnx --onnx romav2_precise.onnx --setting precise --out-dir bench
python scripts/benchmark.py --engine onnx --onnx romav2_precise.onnx --provider CUDAExecutionProvider --setting precise --out-dir bench
python scripts/benchmark.py --report bench
```

Measured for `precise`, batch 1: RTX 3090 — ORT CUDA 1.0 s/pair, PyTorch CUDA 1.1 s/pair; M1 Max CPU — ORT 50 s, PyTorch 66 s. `fast` on the M1 Max CPU: 6.9 s/pair.

---

## 5 — Triton Inference Server

### 5.1 Model repository

Two Triton modules serve every setting, and their `config.pbtxt` files are **generated from one template** so they differ only in name and input size:

| Setting | Dense model (ONNX)            | Sampled ensemble               | `S`  |
|---------|-------------------------------|--------------------------------|------|
| fast    | `romav2`                      | `romav2_sampled`               | 512  |
| base    | `romav2_bidirectional_dense`  | `romav2_bidirectional_sampled` | 640  |
| precise | `romav2_precise_dense`        | `romav2_precise_sampled`       | 1280 |

```bash
python scripts/gen_triton_configs.py          # regenerate after editing the template
python scripts/gen_triton_configs.py --check  # CI-style drift check
```

- **Dense model:** `img_A`, `img_B` `[-1, 3, S, S]` → the six outputs above. GPU instance, bounded CUDA arena and arena shrinkage so it coexists with other models on one GPU.
- **Sampled ensemble:** same inputs plus `num_corresp` and `seed` (`INT64 [1]`); chains the dense model into `romav2_sampler`, a Triton Python backend implementation of RoMaV2's `sample()` (CuPy on GPU, NumPy fallback). Returns sparse correspondences instead of dense `S×S` tensors — use this from clients that just want matches.

Place the export as `model.onnx` in the dense model's version directory:

```bash
cp romav2_fast.onnx    triton/model_repository/romav2/1/model.onnx
cp romav2_precise.onnx triton/model_repository/romav2_precise_dense/1/model.onnx
```

The generated configs target a GPU. For a CPU-only Docker run, change `kind: KIND_GPU` to `KIND_CPU` and delete the `optimization` and `parameters` blocks.

### 5.2 Start the server

```bash
docker run --rm -d \
  --name romav2-triton --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.07-py3 \
  tritonserver --model-repository=/models
```

Wait for the models to load, then verify:

```bash
curl http://localhost:8000/v2/health/ready               # → HTTP 200
curl -X POST http://localhost:8000/v2/repository/index    # → every model READY
```

### 5.3 Run inference

```bash
# dense outputs + visualisation; --setting fixes the input size
python scripts/triton_client.py assets/toronto_A.jpg assets/toronto_B.jpg \
    --url localhost:8000 --model romav2 --setting fast --out result.png
python scripts/triton_client.py assets/toronto_A.jpg assets/toronto_B.jpg \
    --url localhost:8000 --model romav2_precise_dense --setting precise --out precise.png

# sparse correspondences
python scripts/triton_sampled_client.py assets/toronto_A.jpg assets/toronto_B.jpg \
    --url localhost:8000 --model romav2_precise_sampled --setting precise --num-corresp 5000
```

The sampled ensemble returns:

```text
sampled_matches      (N, 4)       # x_A, y_A, x_B, y_B in normalized coordinates
sampled_confidence   (N,)
sampled_precision_A  (N, 2, 2)
sampled_precision_B  (N, 2, 2)
```

Convert to pixels with `(p + 1) / 2 * S - 0.5`.

### 5.4 Use the client in Python

```python
from scripts.triton_client import infer

outs = infer("path/to/image_A.jpg", "path/to/image_B.jpg",
             url="localhost:8000", model_name="romav2_precise_dense", setting="precise")
outs["warp_AB"]       # numpy (1280, 1280, 2)     normalised coords in [-1, 1]
outs["overlap_AB"]    # numpy (1280, 1280, 1)     probability in [0, 1]
outs["precision_AB"]  # numpy (1280, 1280, 2, 2)
# ... and warp_BA / overlap_BA / precision_BA
```

**Triton result** (identical to direct ONNX):

![triton result](assets/triton_result.png)

---

## Repository structure

```
romav2-onnx/
├── scripts/
│   ├── export_onnx.py           # Export + validate (unified interface)
│   ├── gen_triton_configs.py    # Generates triton/model_repository/*/config.pbtxt
│   ├── benchmark.py             # PyTorch vs ONNX Runtime timing + overlap-masked diffs
│   ├── visualize.py             # Composite visualisation (ONNX or PyTorch); shared image helpers
│   ├── triton_client.py         # Triton HTTP client for the dense models
│   └── triton_sampled_client.py # Triton HTTP client for the sampled ensembles
├── triton/
│   └── model_repository/
│       ├── romav2/                        # dense, fast/512       (config generated)
│       ├── romav2_sampled/                # ensemble, fast/512    (config generated)
│       ├── romav2_bidirectional_dense/    # dense, base/640       (config generated)
│       ├── romav2_bidirectional_sampled/  # ensemble, base/640    (config generated)
│       ├── romav2_precise_dense/          # dense, precise/1280   (config generated)
│       ├── romav2_precise_sampled/        # ensemble, precise/1280 (config generated)
│       └── romav2_sampler/                # shared Python-backend sampler (model.py, sampler.py)
│           (each */1/ holds model.onnx, gitignored, or an empty .gitkeep)
└── assets/
    ├── toronto_A.jpg          # Sample input A
    ├── toronto_B.jpg          # Sample input B
    ├── match_result.png       # Sample ONNX output
    └── triton_result.png      # Sample Triton output
```

---

## Notes

- **Float32 only** — all AMP/bfloat16 paths are disabled at export time; the full graph runs in float32 for ORT compatibility.
- **Static `S`** — the input size is baked into the ONNX graph per setting. Export a separate `.onnx` per setting if you need multiple resolutions.
- **Batch dimension** — the Triton configs use `max_batch_size: 0` with an explicit batch dim in `dims`, matching the ONNX model's dynamic batch axis. Pass batches of size ≥ 1 from your client.
- **Precise resize** — the 1280 input is downscaled to 800 inside the graph with an antialiased bicubic. Deriving the low-res pass from the 1280 image instead of the original changes the outputs by less than the CPU-vs-GPU noise of the same graph.
- **Python from the repo root** — the repo's `triton/` folder shadows the `triton` package that torch's dynamo probes on import, which breaks `import torchvision` in an interactive `python` started at the repo root. Run the scripts as `python scripts/…`, or start Python from another directory.
