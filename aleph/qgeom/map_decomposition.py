"""Network-independent recoverable/range split in row-vector convention."""
from dataclasses import dataclass

import torch

from aleph.qgeom.weight_quantizer import validate_matrix


@dataclass(frozen=True)
class MapDecomposition:
    """W = Wq R + U for weights shaped [input, output].

    R = I + pinv(Wq) (W-Wq); U is the remainder outside the retained
    column space of Wq. Near-null singular values are governed by rtol.
    For an exact inverse, U is orthogonal to the full column space of Wq.
    Truncation instead makes U orthogonal only to the retained column space;
    discarded nonzero directions may still be recovered from x Wq by a
    high-gain decoder. A truncation residual is not proof of information loss.
    """

    weight: torch.Tensor
    quantized_weight: torch.Tensor
    R: torch.Tensor
    U: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    rtol: float

    @classmethod
    @torch.no_grad()
    def from_weights(cls, weight: torch.Tensor, quantized_weight: torch.Tensor,
                     *, rtol: float | None = None) -> "MapDecomposition":
        validate_matrix(weight)
        validate_matrix(quantized_weight, "quantized_weight")
        if weight.shape != quantized_weight.shape:
            raise ValueError("weight matrices must have the same shape")
        if weight.dtype != quantized_weight.dtype or weight.device != quantized_weight.device:
            raise ValueError("weight matrices must share dtype and device")
        if rtol is None:
            rtol = max(weight.shape) * torch.finfo(weight.dtype).eps
        if not 0 <= rtol < 1:
            raise ValueError("rtol must be finite and in [0, 1)")
        W, Wq = weight.detach().clone(), quantized_weight.detach().clone()
        difference = W - Wq
        pinv = torch.linalg.pinv(Wq, rtol=rtol)
        R = torch.eye(W.shape[1], dtype=W.dtype, device=W.device) + pinv @ difference
        U = difference - Wq @ (pinv @ difference)
        tolerance = 1e-5 if W.dtype == torch.float32 else 1e-10
        torch.testing.assert_close(Wq @ R + U, W, atol=tolerance, rtol=tolerance)
        spectra = torch.linalg.svdvals(Wq)
        rank = int((spectra > spectra.max() * rtol).sum())
        return cls(W, Wq, R, U, spectra, rank, rtol)

    def diagnostics(self) -> dict:
        """Absolute diagnostics; conditioning uses retained singular values."""
        retained = self.singular_values[:self.rank]
        return {
            "rank": self.rank,
            "rtol": self.rtol,
            "condition_retained": (float(retained[0] / retained[-1])
                                   if self.rank else float("inf")),
            "rank_deficient": self.rank < min(self.weight.shape),
            "reconstruction_max_abs": float((self.weight - self.quantized_weight @ self.R - self.U).abs().max()),
            "range_orthogonality_max_abs": float((self.quantized_weight.T @ self.U).abs().max()),
            "Wq_spectrum": self.singular_values.tolist(),
            "R_spectrum": torch.linalg.svdvals(self.R).tolist(),
            "U_spectrum": torch.linalg.svdvals(self.U).tolist(),
            # Scale the threshold to the original maps, not tiny roundoff in U.
            "U_rank": int(torch.linalg.matrix_rank(
                self.U, atol=self.rtol * max(float(torch.linalg.norm(self.weight, ord=2)),
                                            float(torch.linalg.norm(self.quantized_weight, ord=2))),
                rtol=0)),
        }
