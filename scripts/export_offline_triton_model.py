"""Export the Triton RoMaV2 dense ONNX model from local offline assets.

This script is meant for closed-network machines. It prevents RoMaV2 from
downloading weights by redirecting the model weight URL and DINOv3 torch-hub
load to local files.

Expected local assets:
    weights/romav2.pt          RoMaV2 release checkpoint
    third_party/dinov3/        local clone/copy of facebookresearch/dinov3

Default output:
    triton/model_repository/romav2_bidirectional_dense/1/model.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "triton" / "model_repository" / "romav2_bidirectional_dense" / "1" / "model.onnx"


def _patch_torch_hub(*, roma_weights: Path, dinov3_repo: Path) -> None:
    original_load = torch.hub.load

    def load_state_dict_from_local_url(url: str, *args: Any, **kwargs: Any) -> Any:
        if "romav2.pt" not in url:
            raise RuntimeError(
                f"Offline export blocked an unexpected checkpoint download: {url}"
            )
        map_location = kwargs.get("map_location", "cpu")
        try:
            return torch.load(roma_weights, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(roma_weights, map_location=map_location)

    def load_local_hub(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if "facebookresearch/dinov3" not in repo_or_dir:
            raise RuntimeError(
                f"Offline export blocked an unexpected torch.hub.load call: {repo_or_dir}"
            )
        kwargs["source"] = "local"
        return original_load(str(dinov3_repo), model, *args, **kwargs)

    torch.hub.load_state_dict_from_url = load_state_dict_from_local_url
    torch.hub.load = load_local_hub


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing {description}: {path}")


def _require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"Missing {description}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the offline Triton RoMaV2 bidirectional precision ONNX"
    )
    parser.add_argument(
        "--roma-weights",
        default=ROOT / "weights" / "romav2.pt",
        type=Path,
        help="local romav2.pt checkpoint",
    )
    parser.add_argument(
        "--dinov3-repo",
        default=ROOT / "third_party" / "dinov3",
        type=Path,
        help="local facebookresearch/dinov3 repo copy",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="output ONNX path",
    )
    parser.add_argument(
        "--setting",
        choices=["turbo", "fast", "base"],
        default="fast",
        help="RoMaV2 export setting; Triton configs in this repo expect fast/512",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    roma_weights = args.roma_weights.expanduser().resolve()
    dinov3_repo = args.dinov3_repo.expanduser().resolve()
    output = args.output.expanduser().resolve()

    _require_file(roma_weights, "RoMaV2 checkpoint")
    _require_dir(dinov3_repo, "DINOv3 torch-hub repo")
    _require_file(dinov3_repo / "hubconf.py", "DINOv3 hubconf.py")

    os.environ.setdefault("TORCH_HOME", str((ROOT / "weights" / "torch").resolve()))
    _patch_torch_hub(roma_weights=roma_weights, dinov3_repo=dinov3_repo)

    sys.path.insert(0, str(ROOT))
    from scripts.export_onnx import export

    output.parent.mkdir(parents=True, exist_ok=True)
    export(
        output_path=str(output),
        setting=args.setting,
        opset=args.opset,
        bidirectional=True,
        include_precision=True,
    )

    print("\nOffline export complete.")
    print(f"ONNX: {output}")
    print("Use it with model: romav2_bidirectional_fast_sampled")


if __name__ == "__main__":
    main()
