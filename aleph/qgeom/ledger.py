"""Exact per-sample error creation ledger, including shared float biases."""
from dataclasses import dataclass
from typing import Callable

import torch

from aleph.qgeom.map_decomposition import MapDecomposition


@dataclass(frozen=True)
class ErrorLedger:
    """Three telescoping counterfactual differences; total is Q minus FP.

    All comparisons use the SAME input. Define A(v)=vR+b(I-R).
    h=phi(zq), a=phi(A(zq)), y=phi(x W+b).
    Then uncomp=h-A(h), nonlinear=A(h)-a, range=a-y.
    In the forward repair account, c_R=-uncomp, c_NL=-nonlinear,
    c_U=-range and c_total=-total. The range field keeps its legacy name.
    This affine extension reduces to the plan's h-hR and hR-phi(zq R)
    for zero bias. With an identity activation, nonlinear is exactly zero.
    The oracle residual restoring y is -total. Cross terms matter: the
    three component energies generally do not sum to total energy.
    """

    uncomp: torch.Tensor
    nonlinear: torch.Tensor
    range: torch.Tensor
    total: torch.Tensor
    reconstruction_error: torch.Tensor
    fp_pre: torch.Tensor
    quant_pre: torch.Tensor
    fp_post: torch.Tensor
    quant_post: torch.Tensor
    recoverable_pre: torch.Tensor
    recoverable_post: torch.Tensor
    transported_post: torch.Tensor

    @classmethod
    @torch.no_grad()
    def compute(cls, x: torch.Tensor, decomposition: MapDecomposition, *,
                bias: torch.Tensor | None = None,
                activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
                atol: float | None = None, rtol: float | None = None) -> "ErrorLedger":
        W, Wq, R = decomposition.weight, decomposition.quantized_weight, decomposition.R
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != W.shape[0]:
            raise ValueError("x must be a nonempty [samples, input] matrix")
        if x.dtype != W.dtype or x.device != W.device or not torch.isfinite(x).all():
            raise ValueError("x must be finite and share weights' dtype and device")
        b = torch.zeros(W.shape[1], dtype=W.dtype, device=W.device) if bias is None else bias
        if b.shape != (W.shape[1],) or b.dtype != W.dtype or b.device != W.device or not torch.isfinite(b).all():
            raise ValueError("bias must be a finite output vector sharing weights' dtype and device")
        zq, zfp = x @ Wq + b, x @ W + b
        # Transporting the representation also transports b; restore the shared bias.
        offset = b - b @ R
        recoverable_pre = zq @ R + offset
        h, y, a = activation(zq), activation(zfp), activation(recoverable_pre)
        if any(t.shape != zq.shape for t in (h, y, a)):
            raise ValueError("activation must preserve shape")
        transported = h @ R + offset
        uncomp, nonlinear, range_error = h - transported, transported - a, a - y
        total = h - y
        residual = uncomp + nonlinear + range_error - total
        tolerance = 1e-5 if W.dtype == torch.float32 else 1e-10
        atol = tolerance if atol is None else atol
        rtol = tolerance if rtol is None else rtol
        # Check every sample/coordinate, not just an average norm.
        torch.testing.assert_close(uncomp + nonlinear + range_error, total, atol=atol, rtol=rtol)
        torch.testing.assert_close(recoverable_pre + x @ decomposition.U, zfp, atol=atol, rtol=rtol)
        return cls(uncomp, nonlinear, range_error, total, residual,
                   zfp, zq, y, h, recoverable_pre, a, transported)

    def components(self) -> dict[str, torch.Tensor]:
        return {"uncomp": self.uncomp, "nonlinear": self.nonlinear, "range": self.range}

    def gram(self) -> torch.Tensor:
        """Mean inner products in component order uncomp, nonlinear, range.

        sum(G) is mean squared total norm; off-diagonals expose cancellation
        and reinforcement. This is an interaction summary, not propagation.
        """
        vectors = torch.stack(list(self.components().values()), dim=1)
        return torch.einsum("nif,njf->ij", vectors, vectors) / vectors.shape[0]
