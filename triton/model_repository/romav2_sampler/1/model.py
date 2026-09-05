from __future__ import annotations

import numpy as np
import triton_python_backend_utils as pb_utils

from sampler import sample_roma_outputs, sample_roma_outputs_cupy

try:
    import cupy as cp
except ImportError:  # CuPy ships in nvcr.io/nvidia/tritonserver:*-py3; NumPy path covers its absence.
    cp = None

_DENSE_INPUTS = (
    "warp_AB",
    "overlap_AB",
    "precision_AB",
    "warp_BA",
    "overlap_BA",
    "precision_BA",
)


def _input_as_cupy(request, name: str):
    """The named input as a CuPy array, zero-copy when Triton hands over a GPU buffer.

    With FORCE_CPU_ONLY_INPUT_TENSORS=no on a KIND_GPU instance the tensor already lives
    in device memory, so from_dlpack just adopts the pointer. If it arrives on the host
    anyway (that parameter unset, or a CPU instance) we upload it -- correct either way.
    """
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    if tensor.is_cpu():
        return cp.asarray(tensor.as_numpy())
    return cp.from_dlpack(tensor.to_dlpack())


def _execute_cupy(request, num_corresp: int, seed: int):
    matches, confidence, precision_a, precision_b = sample_roma_outputs_cupy(
        **{name.lower(): _input_as_cupy(request, name) for name in _DENSE_INPUTS},
        num_corresp=num_corresp,
        seed=seed,
    )
    # The sampled outputs are tiny (num_corresp rows) and are about to be serialized to
    # the wire regardless, so bring them home rather than plumbing DLPack back out.
    return [
        pb_utils.Tensor("sampled_matches", cp.asnumpy(matches)),
        pb_utils.Tensor("sampled_confidence", cp.asnumpy(confidence)),
        pb_utils.Tensor("sampled_precision_A", cp.asnumpy(precision_a)),
        pb_utils.Tensor("sampled_precision_B", cp.asnumpy(precision_b)),
    ]


# The NumPy kde() materializes an N x N x 4 temporary with N = 4 * num_corresp, so its
# peak is 64 * num_corresp^2 bytes: fine at 1k (0.24 GiB), 25 GiB at 10k. Falling back to
# it at a large num_corresp would not degrade gracefully -- it would swap the pod to death
# and hang the request forever, which is strictly worse than failing. Cap the fallback and
# surface a real error instead.
_NUMPY_FALLBACK_MAX_BYTES = 2 << 30


def _numpy_fallback_is_safe(num_corresp: int) -> bool:
    return 64 * num_corresp * num_corresp <= _NUMPY_FALLBACK_MAX_BYTES


def _execute_numpy(request, num_corresp: int, seed: int):
    if not _numpy_fallback_is_safe(num_corresp):
        raise RuntimeError(
            f"refusing the NumPy fallback at num_corresp={num_corresp}: its kde() would "
            f"allocate ~{64 * num_corresp**2 / 2**30:.1f} GiB and hang the server. "
            "Fix the CuPy path rather than degrading to this."
        )
    arrays = {
        name.lower(): pb_utils.get_input_tensor_by_name(request, name).as_numpy()
        for name in _DENSE_INPUTS
    }
    matches, confidence, precision_a, precision_b = sample_roma_outputs(
        **arrays,
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
        self._use_cupy = cp is not None
        if not self._use_cupy:
            print("RoMaV2 sampler: CuPy unavailable -- using the NumPy CPU path", flush=True)
            return

        # Pin this instance to the device Triton assigned it.
        device_id = int(args.get("model_instance_device_id", 0) or 0)
        cp.cuda.Device(device_id).use()

        # Warm up by running the REAL sampling path once on a tiny synthetic input. CuPy
        # NVRTC-compiles each kernel on first use, so a cold first request otherwise pays
        # seconds of compilation (it dominated the measured latency when this only warmed
        # a matmul). Every kernel the request path touches -- grid, grid_sample, the KDE
        # GEMM, Gumbel-top-k's argpartition/RNG -- must be exercised here, not just some.
        h = w = 8
        sample_roma_outputs_cupy(
            warp_ab=cp.zeros((1, h, w, 2), cp.float32),
            overlap_ab=cp.ones((1, h, w, 1), cp.float32),
            precision_ab=cp.zeros((1, h, w, 2, 2), cp.float32),
            warp_ba=cp.zeros((1, h, w, 2), cp.float32),
            overlap_ba=cp.ones((1, h, w, 1), cp.float32),
            precision_ba=cp.zeros((1, h, w, 2, 2), cp.float32),
            num_corresp=8,
            seed=0,
        )
        cp.cuda.Stream.null.synchronize()
        print(f"RoMaV2 sampler: CuPy {cp.__version__} on device {device_id} (warmed)", flush=True)

    def execute(self, requests):
        responses = []
        for request in requests:
            num_corresp = int(pb_utils.get_input_tensor_by_name(request, "num_corresp").as_numpy().reshape(-1)[0])
            seed = int(pb_utils.get_input_tensor_by_name(request, "seed").as_numpy().reshape(-1)[0])

            if self._use_cupy:
                try:
                    output_tensors = _execute_cupy(request, num_corresp, seed)
                except Exception as exc:
                    if not self._warned_numpy_fallback:
                        print(
                            "RoMaV2 sampler falling back to NumPy CPU path: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        self._warned_numpy_fallback = True
                    output_tensors = _execute_numpy(request, num_corresp, seed)
            else:
                output_tensors = _execute_numpy(request, num_corresp, seed)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=output_tensors
                )
            )

        # Return CuPy's cached GPU blocks to the driver after each batch so another model
        # (SAM3) can reuse the memory when romav2 isn't sampling -- the CuPy analogue of the
        # `memory.enable_memory_arena_shrinkage` we set on every ONNX model, since that ORT
        # option can't reach this python/CuPy backend. Safe here: _execute_cupy already copies
        # every output to host (cp.asnumpy), so no live device array is referenced by now.
        if self._use_cupy:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        return responses

    def finalize(self):
        pass
