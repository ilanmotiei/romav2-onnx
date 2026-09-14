"""Triton config templates for the RoMaV2 module.

One dense ONNX model plus one sampled ensemble (on top of the shared python
sampler) serve RoMaV2; the module is the same for every setting and only the
input size S differs.  `scripts/export_onnx.py` writes both config.pbtxt files
in the same pass as the model, so the repository always describes the model
that was exported last:

    python scripts/export_onnx.py --setting base            # model + configs
    python scripts/triton_configs.py --setting base          # configs only
    python scripts/triton_configs.py --setting base --check  # exit 1 if out of date

Interface (identical for every setting, see scripts/export_onnx.py):
    inputs   img_A, img_B            FP32 [batch, 3, S, S]
    outputs  warp_AB/BA              FP32 [batch, S, S, 2]
             overlap_AB/BA           FP32 [batch, S, S, 1]
             precision_AB/BA         FP32 [batch, S, S, 2, 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "triton" / "model_repository"
SAMPLER = "romav2_sampler"          # shared Python-backend model, hand-written (not generated)
DEFAULT_NAME = "romav2_bidirectional"   # dense model <name>_dense, ensemble <name>_sampled
SIZES = {"turbo": 320, "fast": 512, "base": 640, "precise": 1280}   # input size S per setting
NOTES = {
    "precise": (
        "# Precise: the graph resizes this 1280x1280 input down to the 800x800 low-res pass\n"
        "# itself (antialiased bicubic as a constant-tap gather), then runs the 1280 refinement stage, so the\n"
        "# client sends one image per side exactly like the other settings.\n"
    ),
}

DENSE = '''name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: 0

# RoMaV2 dense matcher, "{setting}" setting. Same module for every setting; only the
# input size differs. Written by the export together with the model:
#   python scripts/export_onnx.py --setting {setting}
{note}input [
  {{
    name: "img_A"
    data_type: TYPE_FP32
    dims: [ -1, 3, {size}, {size} ]
  }},
  {{
    name: "img_B"
    data_type: TYPE_FP32
    dims: [ -1, 3, {size}, {size} ]
  }}
]

output [
  {{
    name: "warp_AB"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 2 ]
  }},
  {{
    name: "overlap_AB"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 1 ]
  }},
  {{
    name: "precision_AB"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1, -1 ]
  }},
  {{
    name: "warp_BA"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 2 ]
  }},
  {{
    name: "overlap_BA"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 1 ]
  }},
  {{
    name: "precision_BA"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, -1, -1 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

# Bound the ORT GPU arena: kSameAsRequested extends by exactly the requested size instead of
# the default power-of-two doubling. The dense matcher emits huge 5-D warp/precision
# volumes (a 1280 precise export needed two 3.9 GB local-correlation buffers before the
# per-offset rewrite in src/romav2/local_correlation.py),
# so the default strategy over-allocates and the arena grows to swallow the shared 3090
# (it never releases), starving the co-resident SAM3 encoder. This matches the SAM3
# ONNX models' setting. (Distinct from the disabled TensorRT block below.)
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [ {{
      name : "cuda"
      parameters {{ key: "arena_extend_strategy" value: "kSameAsRequested" }}
    }} ]
  }}
}}

# TensorRT FP32 via the ONNX Runtime TensorRT accelerator — built at load time
# and cached to /trt-cache. FP32 (NOT FP16): RoMaV2's ViT/matcher correlations
# overflow FP16's ~65k range and the engine produces all-NaN output, so FP32 is
# the only numerically-correct precision for this model.
#
# PREREQUISITE 1 — the model.onnx in 1/ must be the BAKED-RoPE export
# (romav2-onnx: `python scripts/export_onnx.py --setting {setting} --trt`). RoMaV2's
# DINOv3 RoPE traces the angle .tile() to an `If` with no static shape that TRT's
# parser rejects; the bake replaces the per-resolution sin/cos with constants (the
# export pins the resolution) so the backbone is TRT-parseable.
#
# PREREQUISITE 2 — host RAM. This load-time ORT-TensorRT-EP build is extremely
# memory-heavy: ORT keeps the full ~1.4GB graph resident, re-serializes the TRT
# subgraph, AND TensorRT builds its own network — 2-3 simultaneous copies of the
# weights plus build overhead. The FP16 build of the 512 variant already
# OOM-killed a 24Gi pod; FP32 is HEAVIER (its TRT network is 2x the FP16 one), so
# the at-load path needs a box with ~30Gi+ FREE host RAM. It will NOT build on the
# homelab node. 8GiB workspace is GPU scratch (the 3090 has 24GB) — fixes the earlier
# "insufficient workspace at refiners.4/Concat" failure; it does not affect host RAM.
#
# On a memory-constrained node use the PREBUILT-PLAN path instead: build the .plan
# offline with trtexec (FP32 peaks only ~10-11GiB host, vs >24GiB for ORT-EP) and
# deploy as platform "tensorrt_plan" — Triton then just deserializes it at load.
#
# DISABLED — against a plain (non-baked-RoPE) export ORT hands TRT a subgraph whose
# input `/model/f/rope_embed/Sub_1_output_0` has a dynamic shape with no profile, so
# the load fails outright with:
#   "Following input(s) has no associated shape profiles provided: /model/f/rope_embed/..."
# and — because strict_readiness gates the pod on ALL models loading — that takes the whole
# Triton server down, not just this model. Re-enable only together with the baked-RoPE
# export, and only on a node that satisfies PREREQUISITE 2. Until then the model runs on
# the plain CUDA EP, which is what serves today.
#
# optimization {{
#   execution_accelerators {{
#     gpu_execution_accelerator: [
#       {{
#         name: "tensorrt"
#         parameters {{ key: "precision_mode" value: "FP32" }}
#         parameters {{ key: "max_workspace_size_bytes" value: "8589934592" }}
#         parameters {{ key: "trt_builder_optimization_level" value: "1" }}
#         parameters {{ key: "trt_engine_cache_enable" value: "true" }}
#         parameters {{ key: "trt_engine_cache_path" value: "/trt-cache/{name}" }}
#         parameters {{ key: "trt_profile_min_shapes" value: "img_A:1x3x{size}x{size},img_B:1x3x{size}x{size}" }}
#         parameters {{ key: "trt_profile_opt_shapes" value: "img_A:1x3x{size}x{size},img_B:1x3x{size}x{size}" }}
#         parameters {{ key: "trt_profile_max_shapes" value: "img_A:2x3x{size}x{size},img_B:2x3x{size}x{size}" }}
#       }}
#     ]
#   }}
# }}

# Release this model's CUDA activation arena to the driver after each run so SAM3 and
# romav2 hand the GPU back to each other between sequential uses (they never inference at
# the same time). See sam3_unified_encoder for the full rationale.
parameters {{ key: "memory.enable_memory_arena_shrinkage" value: {{ string_value: "gpu:0" }} }}
parameters {{ key: "session.use_device_allocator_for_initializers" value: {{ string_value: "1" }} }}
'''

SAMPLED = '''name: "{sampled}"
platform: "ensemble"
max_batch_size: 0
# RoMaV2 "{setting}" dense matcher followed by the shared romav2_sampler (RoMaV2.sample()).
# Generated by scripts/gen_triton_configs.py; same pipeline for every setting.

input [
  {{
    name: "img_A"
    data_type: TYPE_FP32
    dims: [ -1, 3, {size}, {size} ]
  }},
  {{
    name: "img_B"
    data_type: TYPE_FP32
    dims: [ -1, 3, {size}, {size} ]
  }},
  {{
    name: "num_corresp"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }},
  {{
    name: "seed"
    data_type: TYPE_INT64
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "sampled_matches"
    data_type: TYPE_FP32
    dims: [ -1, 4 ]
  }},
  {{
    name: "sampled_confidence"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }},
  {{
    name: "sampled_precision_A"
    data_type: TYPE_FP32
    dims: [ -1, 2, 2 ]
  }},
  {{
    name: "sampled_precision_B"
    data_type: TYPE_FP32
    dims: [ -1, 2, 2 ]
  }}
]

ensemble_scheduling {{
  step [
    {{
      model_name: "{name}"
      model_version: -1
      input_map {{
        key: "img_A"
        value: "img_A"
      }}
      input_map {{
        key: "img_B"
        value: "img_B"
      }}
      output_map {{
        key: "warp_AB"
        value: "dense_warp_AB"
      }}
      output_map {{
        key: "overlap_AB"
        value: "dense_overlap_AB"
      }}
      output_map {{
        key: "precision_AB"
        value: "dense_precision_AB"
      }}
      output_map {{
        key: "warp_BA"
        value: "dense_warp_BA"
      }}
      output_map {{
        key: "overlap_BA"
        value: "dense_overlap_BA"
      }}
      output_map {{
        key: "precision_BA"
        value: "dense_precision_BA"
      }}
    }},
    {{
      model_name: "romav2_sampler"
      model_version: -1
      input_map {{
        key: "warp_AB"
        value: "dense_warp_AB"
      }}
      input_map {{
        key: "overlap_AB"
        value: "dense_overlap_AB"
      }}
      input_map {{
        key: "precision_AB"
        value: "dense_precision_AB"
      }}
      input_map {{
        key: "warp_BA"
        value: "dense_warp_BA"
      }}
      input_map {{
        key: "overlap_BA"
        value: "dense_overlap_BA"
      }}
      input_map {{
        key: "precision_BA"
        value: "dense_precision_BA"
      }}
      input_map {{
        key: "num_corresp"
        value: "num_corresp"
      }}
      input_map {{
        key: "seed"
        value: "seed"
      }}
      output_map {{
        key: "sampled_matches"
        value: "sampled_matches"
      }}
      output_map {{
        key: "sampled_confidence"
        value: "sampled_confidence"
      }}
      output_map {{
        key: "sampled_precision_A"
        value: "sampled_precision_A"
      }}
      output_map {{
        key: "sampled_precision_B"
        value: "sampled_precision_B"
      }}
    }}
  ]
}}
'''


def model_names(name: str = DEFAULT_NAME) -> tuple[str, str]:
    """(dense model name, sampled ensemble name) for a Triton module name."""
    return f"{name}_dense", f"{name}_sampled"


def render(setting: str, size: int | None = None, name: str = DEFAULT_NAME,
           repo: Path = REPO) -> dict[Path, str]:
    """config.pbtxt path -> text for the dense model and the sampled ensemble."""
    size = SIZES[setting] if size is None else size
    dense, sampled = model_names(name)
    return {
        repo / dense / "config.pbtxt": DENSE.format(
            name=dense, setting=setting, size=size, note=NOTES.get(setting, "")),
        repo / sampled / "config.pbtxt": SAMPLED.format(
            name=dense, sampled=sampled, setting=setting, size=size),
    }


def write_triton_configs(setting: str, size: int | None = None, name: str = DEFAULT_NAME,
                         repo: Path = REPO) -> list[Path]:
    """Write both configs (and the version-dir placeholders); returns the paths written."""
    written = []
    for path, text in render(setting, size, name, repo).items():
        (path.parent / "1").mkdir(parents=True, exist_ok=True)
        (path.parent / "1" / ".gitkeep").touch()
        path.write_text(text)
        written.append(path)
    return written


def check_triton_configs(setting: str, size: int | None = None, name: str = DEFAULT_NAME,
                         repo: Path = REPO) -> list[str]:
    """Problems that make the repository differ from what an export would write."""
    problems = []
    for path, text in render(setting, size, name, repo).items():
        if not path.exists():
            problems.append(f"missing: {path.relative_to(repo.parent.parent)}")
        elif path.read_text() != text:
            problems.append(f"out of date: {path.relative_to(repo.parent.parent)}")
    expected = set(model_names(name)) | {SAMPLER}
    for d in sorted(repo.iterdir()):
        if d.is_dir() and d.name not in expected:
            problems.append(f"unexpected model directory: {d.name}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--setting", required=True, choices=list(SIZES),
                        help="setting the repository should describe (fixes the input size)")
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help=f"Triton module name; models are <name>_dense and <name>_sampled "
                             f"(default {DEFAULT_NAME})")
    parser.add_argument("--check", action="store_true",
                        help="verify the repository instead of writing it")
    args = parser.parse_args()
    if args.check:
        problems = check_triton_configs(args.setting, name=args.name)
        if problems:
            print("\n".join(problems))
            sys.exit(1)
        print(f"Triton configs up to date for setting '{args.setting}'.")
        return
    for path in write_triton_configs(args.setting, name=args.name):
        print(f"wrote {path.relative_to(REPO.parent.parent)}")


if __name__ == "__main__":
    main()
