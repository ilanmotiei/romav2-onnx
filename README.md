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

```bash
# Fast setting (512×512 input, ~350 MB model)
python scripts/export_onnx.py --output romav2_fast.onnx --setting fast

# Fast setting with both A→B and B→A dense outputs
python scripts/export_onnx.py \
    --output romav2_fast_bidir.onnx \
    --setting fast \
    --bidirectional

# Base setting (640×640)
python scripts/export_onnx.py --output romav2_base.onnx --setting base
```

Available settings:

| Setting | Input size | Notes |
|---------|-----------|-------|
| `turbo` | 320×320 | Fastest, least accurate |
| `fast`  | 512×512 | Good balance |
| `base`  | 640×640 | Higher accuracy |

The script:
- Disables AMP / bfloat16 everywhere so the full graph stays in float32 (required for ORT compatibility)
- Forces RoPE embeddings to float32
- Uses the classic TorchScript-based JIT tracer (`dynamo=False`)
- Bakes H/W into the graph; only the batch dimension is dynamic

**Inputs:** `img_A`, `img_B` — `float32 [B, 3, H, W]`, values in `[0, 1]`
**Outputs:** `warp_AB` — `float32 [B, H, W, 2]` (normalised coords in `[-1, 1]`), `overlap_AB` — `float32 [B, H, W, 1]` (probability in `[0, 1]`)

Add `--bidirectional` to also export `warp_BA` and `overlap_BA` with the same shapes. This is slower because the matcher/refiners also compute the B→A direction, but it is useful when downstream sampling or geometry wants both directions.

Add `--include-precision` for models that will feed the Triton sampled ensemble. This also exports `precision_AB` and, with `--bidirectional`, `precision_BA`, matching the precision matrices consumed by RoMaV2's `sample()` flow:

```bash
python scripts/export_onnx.py \
    --output romav2_fast_bidir_precision.onnx \
    --setting fast \
    --bidirectional \
    --include-precision
```

---

## 2 — Validate

Runs both the PyTorch model and the exported ONNX model on identical CPU inputs and asserts their outputs match within tolerance.

```bash
python scripts/export_onnx.py --validate romav2_fast.onnx --setting fast

# Validate a bidirectional export
python scripts/export_onnx.py \
    --validate romav2_fast_bidir.onnx \
    --setting fast \
    --bidirectional
```

Expected output:
```
[1/5] Building PyTorch model on CPU ...       Done in 8.2s
[2/5] Running PyTorch forward pass on CPU ... Done in 7.1s
      pt_warp:    shape=(1, 512, 512, 2), min=-0.9123, max=0.9087
      pt_overlap: shape=(1, 512, 512, 1), min=0.0213, max=0.8315
[3/5] Loading ONNX model from romav2_fast.onnx ... Done in 3.4s
[4/5] Running ONNX inference (CPU) ...        Done in 7.4s
      onnx_warp:    shape=(1, 512, 512, 2), min=-0.9123, max=0.9087
[5/5] Comparing outputs (atol=0.02) ...
Validation passed — PyTorch and ONNX outputs match.
```

> Both passes run on CPU so the comparison is numerically equivalent. MPS vs CPU diverges by ~0.23 in warp coords for a 24-layer ViT; always validate CPU-vs-CPU.

---

## 3 — Visualise (direct ONNX)

```bash
# Uses sample images included in this repo
python scripts/visualize.py --onnx romav2_fast.onnx --out result.png

# Bidirectional ONNX exports are detected automatically and produce both directions
python scripts/visualize.py --onnx romav2_fast_bidir.onnx --out bidir_result.png

# Or PyTorch model
python scripts/visualize.py --out result.png

# Custom images
python scripts/visualize.py \
    --img-a path/to/image_A.jpg \
    --img-b path/to/image_B.jpg \
    --onnx romav2_fast.onnx \
    --out result.png
```

The output is a 6-panel composite (2×3 grid):

| Top-left | Top-center | Top-right |
|----------|------------|-----------|
| Image A | Image B | Image B warped into A |

| Bottom-left | Bottom-center | Bottom-right |
|-------------|---------------|--------------|
| Confidence heatmap | Alpha blend | Dense correspondences |

For bidirectional ONNX exports, the image stacks two 6-panel composites: A→B first, then B→A.

![bidirectional ONNX result](assets/bidir_onnx_result.png)

---

## 4 — Triton Inference Server

### 4.1 Set up the model repository

The `triton/model_repository/romav2/config.pbtxt` is already configured. You only need to place the exported ONNX file:

```bash
mkdir -p triton/model_repository/romav2/1
cp romav2_fast.onnx triton/model_repository/romav2/1/model.onnx
```

For a bidirectional export, use the separate config:

```bash
mkdir -p triton/model_repository/romav2_bidirectional/1
cp romav2_fast_bidir.onnx triton/model_repository/romav2_bidirectional/1/model.onnx
```

To avoid returning dense `H×W×...` tensors over the network, use the sampled ensemble. It chains a precision-exported bidirectional ONNX model into `romav2_sampler`, a Triton Python backend implementation of RoMaV2's sampling flow:

```bash
mkdir -p triton/model_repository/romav2_bidirectional_dense/1
cp romav2_fast_bidir_precision.onnx \
  triton/model_repository/romav2_bidirectional_dense/1/model.onnx
```

Then call the user-facing ensemble model `romav2_bidirectional_sampled`, not the dense model. The dense tensors stay inside Triton.

If you export `turbo` or `base`, update the `img_A`/`img_B` input dimensions in the relevant `config.pbtxt` to `320×320` or `640×640`.

The checked-in Triton configs keep output dimensions fully dynamic for compatibility with `nvcr.io/nvidia/tritonserver:23.12-py3`. If a newer Triton version reports that the model expects concrete output dimensions, set warp outputs to `[ -1, -1, -1, 2 ]`, overlap outputs to `[ -1, -1, -1, 1 ]`, and precision outputs to `[ -1, -1, -1, 2, 2 ]` in that environment.

### 4.2 Start the server

```bash
docker run --rm -d \
  --name romav2-triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.12-py3 \
  tritonserver --model-repository=/models
```

Add `--gpus all` if you have a CUDA GPU. Wait ~10 seconds, then verify:

```bash
curl http://localhost:8000/v2/health/ready        # → HTTP 200
curl http://localhost:8000/v2/models/romav2        # → model metadata JSON
```

### 4.3 Run inference

```bash
python scripts/triton_client.py \
    assets/toronto_A.jpg assets/toronto_B.jpg \
    --url localhost:8000 \
    --out result.png
```

For the sampled ensemble:

```bash
python scripts/triton_sampled_client.py \
    assets/toronto_A.jpg assets/toronto_B.jpg \
    --url localhost:8000 \
    --model romav2_bidirectional_sampled \
    --num-corresp 5000
```

The sampled ensemble returns:

```text
sampled_matches      (N, 4)       # x_A, y_A, x_B, y_B in normalized coordinates
sampled_confidence   (N,)
sampled_precision_A  (N, 2, 2)
sampled_precision_B  (N, 2, 2)
```

### 4.4 Use the client in Python

```python
from scripts.triton_client import infer

warp_AB, overlap_AB = infer(
    "path/to/image_A.jpg",
    "path/to/image_B.jpg",
    url="localhost:8000",       # or remote host:port
    model_name="romav2",
)
# warp_AB:    numpy (512, 512, 2)  normalised coords in [-1, 1]
# overlap_AB: numpy (512, 512, 1)  probability in [0, 1]
```

**Triton result** (identical to direct ONNX):

![triton result](assets/triton_result.png)

---

## Repository structure

```
romav2-onnx/
├── scripts/
│   ├── export_onnx.py        # Export + validate
│   ├── visualize.py          # 6-panel visualisation (ONNX or PyTorch)
│   ├── triton_client.py      # Triton HTTP client + visualisation
│   └── triton_sampled_client.py
├── triton/
│   └── model_repository/
│       ├── romav2/
│           ├── config.pbtxt  # Triton model config
│           └── 1/            # Place model.onnx here (gitignored)
│       ├── romav2_bidirectional/
│           ├── config.pbtxt  # Triton config for --bidirectional exports
│           └── 1/            # Place bidirectional model.onnx here (gitignored)
│       ├── romav2_bidirectional_dense/
│       │   ├── config.pbtxt  # Internal dense ONNX model for the sampled ensemble
│       │   └── 1/            # Place precision-exported bidirectional model.onnx here
│       ├── romav2_sampler/
│       │   ├── config.pbtxt  # Triton Python backend sampler
│       │   └── 1/
│       └── romav2_bidirectional_sampled/
│           ├── config.pbtxt  # User-facing ensemble returning sparse samples
│           └── 1/            # Empty version directory required by Triton
└── assets/
    ├── toronto_A.jpg          # Sample input A
    ├── toronto_B.jpg          # Sample input B
    ├── match_result.png       # Sample ONNX output
    └── triton_result.png      # Sample Triton output
```

---

## Notes

- **Float32 only** — all AMP/bfloat16 paths are disabled at export time; the full graph runs in float32 for ORT compatibility.
- **Static H/W** — height and width are baked into the ONNX graph per setting. Export a separate `.onnx` per setting if you need multiple resolutions.
- **Batch dimension** — the Triton config uses `max_batch_size: 0` with an explicit batch dim in `dims`, matching the ONNX model's fully-dynamic shape annotation. Pass batches of size ≥ 1 from your client.
- **GPU** — the exported model runs on CPU by default in both ORT and Triton. For CUDA GPU inference in Triton, set `kind: KIND_GPU` in `config.pbtxt` and pass `--gpus all` to Docker.
