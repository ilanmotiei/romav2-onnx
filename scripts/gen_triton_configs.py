"""Generate the Triton model repository from one template per module.

Every RoMaV2 setting is served through the same two Triton modules -- a dense
ONNX model and a sampled ensemble on top of the shared python sampler -- so the
per-setting config.pbtxt files differ only in their name and input size.  They
are generated here rather than hand-edited so they cannot drift apart.

    python scripts/gen_triton_configs.py            # (re)write triton/model_repository/*/config.pbtxt
    python scripts/gen_triton_configs.py --check    # exit 1 if any file is out of date

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
SAMPLER = "romav2_sampler"   # shared Python-backend model, hand-written (not generated)

# setting -> (dense model name, sampled ensemble name, input size S, note for the dense config)
# Only the setting deployed on modelhub is kept here (base, 640). The turbo/fast/precise
# exports still work but have no Triton entry; add a row here to deploy one.
MODELS = {
    "base": ("romav2_bidirectional_dense", "romav2_bidirectional_sampled", 640, ""),
}

DENSE = '''name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: 0

# RoMaV2 dense matcher, "{setting}" setting. Same module for every setting; only the
# input size differs (see scripts/gen_triton_configs.py). Export with
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


def render() -> dict[Path, str]:
    files: dict[Path, str] = {}
    for setting, (name, sampled, size, note) in MODELS.items():
        files[REPO / name / "config.pbtxt"] = DENSE.format(name=name, setting=setting, size=size, note=note)
        files[REPO / sampled / "config.pbtxt"] = SAMPLED.format(name=name, sampled=sampled, setting=setting, size=size)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify files are up to date instead of writing")
    args = parser.parse_args()

    stale = []
    for path, text in render().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "1").mkdir(exist_ok=True)
        (path.parent / "1" / ".gitkeep").touch()
        path.write_text(text)
        print(f"wrote {path.relative_to(REPO.parents[1])}")
    if args.check:
        expected = {p.parent.name for p in render()} | {SAMPLER}
        extra = sorted(d.name for d in REPO.iterdir() if d.is_dir() and d.name not in expected)
        if extra:
            print("not generated by this script (delete or add to MODELS):\n  " + "\n  ".join(extra))
            stale.extend(REPO / d for d in extra)
        if stale:
            print("out of date:\n  " + "\n  ".join(str(p.relative_to(REPO.parents[1])) for p in stale))
            sys.exit(1)
        print("all Triton configs up to date")


if __name__ == "__main__":
    main()
