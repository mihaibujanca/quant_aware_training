"""Canonical space error analysis for quantized neural networks.

Maps quantization errors to a common coordinate system (input space) and
decomposes them into local vs propagated components.  All operations use
PyTorch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


@dataclass
class ForwardTrace:
    """Pre- and post-activation tensors collected from a forward pass.

    Framework-agnostic: build this however you like (manual loops,
    nn.Module hooks, etc.).
    """

    pre_acts: List[torch.Tensor]  # z_i at each layer (before activation)
    post_acts: List[torch.Tensor]  # a_i at each layer (after activation)


class CanonicalSpaceTracker:
    """Map errors to input space via cumulative transform pseudoinverse.

    Computes T_L = W_L ... W_1 for each layer and caches pinv(T_L).
    Errors at layer L are mapped to the input space via T_L^+, giving a
    consistent coordinate system for comparing errors across layers of
    different widths.
    """

    def __init__(self, weights: List[torch.Tensor]):
        self._T: List[torch.Tensor] = []
        self._T_inv: List[torch.Tensor] = []
        T = torch.eye(weights[0].shape[1], dtype=weights[0].dtype)
        for W in weights:
            T = W @ T
            self._T.append(T.clone())
            self._T_inv.append(torch.linalg.pinv(T))

    def cumulative_transform(self, layer_idx: int) -> torch.Tensor:
        return self._T[layer_idx]

    def to_input_space(self, error: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Map error vector(s) to canonical (input) space via T_L^+."""
        return F.linear(error, self._T_inv[layer_idx])

    def cumulative_amplification(self, layer_idx: int) -> float:
        return float(torch.linalg.norm(self._T[layer_idx], ord=2))

    def decompose_error(
        self,
        layer_idx: int,
        total_output: torch.Tensor,
        local_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map total and local error to canonical space.

        Returns (local_canonical, propagated_canonical, total_canonical).
        Propagated = total - local (exact, no pseudoinverse identities).
        """
        total_can = self.to_input_space(total_output, layer_idx)
        local_can = self.to_input_space(local_output, layer_idx)
        return local_can, total_can - local_can, total_can


class ReLUDisagreementTracker:
    """Track where float and quantized networks make different ReLU decisions.

    Only examines hidden layers (all but the last in the trace).
    """

    def __init__(self, float_trace: ForwardTrace, quant_trace: ForwardTrace):
        n_hidden = len(float_trace.pre_acts) - 1
        self.disagreements: List[torch.Tensor] = []
        self.fractions: List[float] = []
        for i in range(n_hidden):
            mask = (float_trace.pre_acts[i] > 0) != (quant_trace.pre_acts[i] > 0)
            self.disagreements.append(mask)
            self.fractions.append(float(mask.float().mean()))

    def any_disagreement(self, layer_idx: int) -> torch.Tensor:
        """True for each sample where at least one neuron disagrees."""
        m = self.disagreements[layer_idx]
        return m.any() if m.ndim == 1 else m.any(dim=-1)


def error_attribution(
    x: torch.Tensor,
    weights: List[torch.Tensor],
    weights_q: List[torch.Tensor],
    float_trace: ForwardTrace,
    quant_trace: ForwardTrace,
    tracker: CanonicalSpaceTracker,
) -> List[Dict[str, Any]]:
    """Per-layer error decomposition into local and propagated components.

    Local error:  E_L @ a_hat_{L-1}  (this layer's quantization).
    Propagated:   total - local       (errors from all previous layers).
    Both are also projected to canonical (input) space via *tracker*.
    """
    results: List[Dict[str, Any]] = []
    for i in range(len(weights)):
        E = weights_q[i] - weights[i]
        a_prev = quant_trace.post_acts[i - 1] if i > 0 else x
        total_out = quant_trace.pre_acts[i] - float_trace.pre_acts[i]
        local_out = F.linear(a_prev, E)
        local_can, prop_can, total_can = tracker.decompose_error(
            i, total_out, local_out
        )
        results.append(
            {
                "layer": i,
                "local_output": local_out,
                "propagated_output": total_out - local_out,
                "total_output": total_out,
                "local_canonical": local_can,
                "propagated_canonical": prop_can,
                "total_canonical": total_can,
            }
        )
    return results
