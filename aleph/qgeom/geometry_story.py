"""Geometric teaching examples: move the same grid, circle and named points.

Uses the exact toolkit rather than a second implementation of the algebra.
All weights use row-vector convention; all examples have zero float bias.
"""
from dataclasses import dataclass

import torch

from aleph.qgeom.ledger import ErrorLedger
from aleph.qgeom.map_decomposition import MapDecomposition
from aleph.qgeom.weight_quantizer import WeightQuantizer


@dataclass(frozen=True)
class GeometryStory:
    key: str
    title: str
    labels: tuple[str, ...]
    points: torch.Tensor
    path_lengths: tuple[int, ...]
    states: tuple[torch.Tensor, ...]
    reference_indices: tuple[int | None, ...]
    storyboard_indices: tuple[int, ...]
    decomposition: MapDecomposition
    ledger: ErrorLedger

    def payload(self):
        def values(t):
            return [[round(float(v), 6) for v in row] for row in t]
        return {
            "key": self.key, "title": self.title, "labels": self.labels,
            "pathLengths": self.path_lengths, "states": [values(s) for s in self.states],
            "references": self.reference_indices,
            "W": self.decomposition.weight.tolist(), "Wq": self.decomposition.quantized_weight.tolist(),
            "R": self.decomposition.R.tolist(), "U": self.decomposition.U.tolist(),
        }


def _geometry(probes):
    dense = torch.linspace(-1, 1, 41, dtype=torch.float64)
    paths = []
    for position in torch.linspace(-1, 1, 9, dtype=torch.float64):
        paths.extend([torch.stack((dense, torch.full_like(dense, position)), dim=1),
                      torch.stack((torch.full_like(dense, position), dense), dim=1)])
    theta = torch.linspace(0, 2 * torch.pi, 121, dtype=torch.float64)
    paths.append(torch.stack((.9*theta.cos(), .9*theta.sin()), dim=1))
    return torch.cat(paths + [torch.tensor(probes, dtype=torch.float64)]), tuple(len(p) for p in paths)


def build_geometry_stories() -> tuple[GeometryStory, ...]:
    results = []
    specifications = [
        ("deformation", "Two different maps; the quantized route can be corrected", [[1., .65], [-.35, .8]], [[-.6, .9], [.7, -.4]]),
        ("relu", "ReLU can erase what the correction needs", [[1., -.4], [.4, 1.]], [[-.8, 0.], [0., -.8]]),
        ("loss", "A flattened shape has lost a direction", [[1., .3], [.25, .25]], [[.4, -.8], [.4, .8]]),
    ]
    for key, title, weights, probes in specifications:
        x, lengths = _geometry(probes)
        W = torch.tensor(weights, dtype=torch.float64)
        Wq = WeightQuantizer(2)(W).quantized
        decomp = MapDecomposition.from_weights(W, Wq)
        ledger = ErrorLedger.compute(x, decomp)
        if key == "deformation":
            states = (x, x @ W, x @ Wq, x @ Wq @ decomp.R)
            labels = ("Input x", "FP target: xW", "Apply rounded weights: xWq",
                      "Apply correction: (xWq)R = xW")
            references, indices = (None, None, None, None), (0, 1, 2, 3)
        elif key == "relu":
            states = (x @ Wq, x @ Wq @ decomp.R, ledger.fp_post,
                      ledger.quant_post, ledger.transported_post)
            labels = ("Quantized preactivation", "Correct before ReLU", "Then ReLU: FP recovered",
                      "Instead: ReLU first", "Correct after ReLU: mismatch")
            references, indices = (None, None, None, 2, 2), (0, 2, 3, 4)
        else:
            states = (x, x @ W, x @ Wq, x @ Wq @ decomp.R,
                      x @ Wq @ decomp.R + x @ decomp.U, ledger.fp_post)
            labels = ("Input", "Full-precision map", "Weights snapped: line", "Correct map: still a line",
                      "Add input-dependent xU", "Then ReLU: FP restored")
            references, indices = (None, None, 1, 1, 1, None), (1, 2, 3, 4)
        results.append(GeometryStory(key, title, labels, x, lengths, states,
                                     references, indices, decomp, ledger))
    return tuple(results)
