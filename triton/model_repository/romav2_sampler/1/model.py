from __future__ import annotations

import numpy as np
import triton_python_backend_utils as pb_utils

from sampler import sample_roma_outputs


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            warp_ab = pb_utils.get_input_tensor_by_name(request, "warp_AB").as_numpy()
            overlap_ab = pb_utils.get_input_tensor_by_name(request, "overlap_AB").as_numpy()
            precision_ab = pb_utils.get_input_tensor_by_name(request, "precision_AB").as_numpy()
            warp_ba = pb_utils.get_input_tensor_by_name(request, "warp_BA").as_numpy()
            overlap_ba = pb_utils.get_input_tensor_by_name(request, "overlap_BA").as_numpy()
            precision_ba = pb_utils.get_input_tensor_by_name(request, "precision_BA").as_numpy()
            num_corresp = int(pb_utils.get_input_tensor_by_name(request, "num_corresp").as_numpy().reshape(-1)[0])
            seed = int(pb_utils.get_input_tensor_by_name(request, "seed").as_numpy().reshape(-1)[0])

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

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("sampled_matches", matches),
                        pb_utils.Tensor("sampled_confidence", confidence),
                        pb_utils.Tensor("sampled_precision_A", precision_a),
                        pb_utils.Tensor("sampled_precision_B", precision_b),
                    ]
                )
            )
        return responses

    def finalize(self):
        pass
