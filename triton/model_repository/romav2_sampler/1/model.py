from __future__ import annotations

import numpy as np
import triton_python_backend_utils as pb_utils

from sampler import sample_roma_outputs, sample_roma_outputs_torch

try:
    import torch
    from torch.utils.dlpack import from_dlpack, to_dlpack
except ImportError:  # Plain Triton py3 images do not ship PyTorch.
    torch = None
    from_dlpack = None
    to_dlpack = None


def _input_as_torch(request, name: str) -> torch.Tensor:
    if from_dlpack is None:
        raise RuntimeError("PyTorch/DLPack path is unavailable")
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    return from_dlpack(tensor.to_dlpack())


def _output_from_torch(name: str, tensor: torch.Tensor):
    if to_dlpack is None:
        raise RuntimeError("PyTorch/DLPack path is unavailable")
    return pb_utils.Tensor.from_dlpack(name, to_dlpack(tensor.contiguous()))


def _execute_torch(request, num_corresp: int, seed: int):
    matches, confidence, precision_a, precision_b = sample_roma_outputs_torch(
        warp_ab=_input_as_torch(request, "warp_AB"),
        overlap_ab=_input_as_torch(request, "overlap_AB"),
        precision_ab=_input_as_torch(request, "precision_AB"),
        warp_ba=_input_as_torch(request, "warp_BA"),
        overlap_ba=_input_as_torch(request, "overlap_BA"),
        precision_ba=_input_as_torch(request, "precision_BA"),
        num_corresp=num_corresp,
        seed=seed,
    )
    return [
        _output_from_torch("sampled_matches", matches),
        _output_from_torch("sampled_confidence", confidence),
        _output_from_torch("sampled_precision_A", precision_a),
        _output_from_torch("sampled_precision_B", precision_b),
    ]


def _execute_numpy(request, num_corresp: int, seed: int):
    warp_ab = pb_utils.get_input_tensor_by_name(request, "warp_AB").as_numpy()
    overlap_ab = pb_utils.get_input_tensor_by_name(request, "overlap_AB").as_numpy()
    precision_ab = pb_utils.get_input_tensor_by_name(request, "precision_AB").as_numpy()
    warp_ba = pb_utils.get_input_tensor_by_name(request, "warp_BA").as_numpy()
    overlap_ba = pb_utils.get_input_tensor_by_name(request, "overlap_BA").as_numpy()
    precision_ba = pb_utils.get_input_tensor_by_name(request, "precision_BA").as_numpy()

    matches, confidence, precision_a, precision_b = sample_roma_outputs(
        warp_ab=warp_ab,
        overlap_ab=overlap_ab,
        precision_ab=precision_ab,
        warp_ba=warp_ba,
        overlap_ba=overlap_ba,
        precision_ba=precision_ba,
        num_corresp=num_corresp,
        seed=seed,
    )
    return [
        pb_utils.Tensor("sampled_matches", matches),
        pb_utils.Tensor("sampled_confidence", confidence),
        pb_utils.Tensor("sampled_precision_A", precision_a),
        pb_utils.Tensor("sampled_precision_B", precision_b),
    ]


class TritonPythonModel:
    def initialize(self, args):
        self._warned_numpy_fallback = False

    def execute(self, requests):
        responses = []
        for request in requests:
            num_corresp = int(pb_utils.get_input_tensor_by_name(request, "num_corresp").as_numpy().reshape(-1)[0])
            seed = int(pb_utils.get_input_tensor_by_name(request, "seed").as_numpy().reshape(-1)[0])

            try:
                output_tensors = _execute_torch(request, num_corresp, seed)
            except Exception as exc:
                if not self._warned_numpy_fallback:
                    print(
                        "RoMaV2 sampler falling back to NumPy CPU path: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    self._warned_numpy_fallback = True
                output_tensors = _execute_numpy(request, num_corresp, seed)
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=output_tensors
                )
            )
        return responses

    def finalize(self):
        pass
