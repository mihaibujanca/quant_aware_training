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
class ScalarMeasurementStory:
    """One scalar measurement changes direction without changing its rank."""

    w: torch.Tensor
    wq: torch.Tensor
    r: torch.Tensor
    u: torch.Tensor
    points: torch.Tensor


def build_scalar_measurement_story() -> ScalarMeasurementStory:
    W = torch.tensor([[1.], [.55]], dtype=torch.float64)
    Wq = WeightQuantizer(2)(W).quantized
    decomposition = MapDecomposition.from_weights(W, Wq)
    return ScalarMeasurementStory(W[:, 0], Wq[:, 0], decomposition.R[0, 0],
                                  decomposition.U[:, 0],
                                  torch.tensor([[.8, -.2], [-.2, .8]], dtype=W.dtype))


@dataclass(frozen=True)
class RectangularMeasurementStory:
    decomposition: MapDecomposition
    points: torch.Tensor
    fiber_direction: torch.Tensor


def build_rectangular_measurement_story() -> RectangularMeasurementStory:
    """A 3→2 measurement keeps rank 2 but changes its blind direction."""
    W = torch.tensor([[1., .25], [.75, 1.], [.25, .8]], dtype=torch.float64)
    Wq = WeightQuantizer(2)(W).quantized
    direction = torch.tensor([1., -1., 1.], dtype=W.dtype)
    # C and D share the Q preactivation (0.6, -0.3); they sit on either side of the measured plane.
    center = torch.tensor([.6, -.3], dtype=W.dtype) @ torch.linalg.pinv(Wq)
    return RectangularMeasurementStory(MapDecomposition.from_weights(W, Wq),
                                       torch.stack((center + .3*direction, center - .3*direction)),
                                       direction)


@dataclass(frozen=True)
class TypicalLedgerStory:
    """The typical 3→2 map through ReLU: one shared Q route, two FP targets.

    The grid/circle is parameterized by its Q preactivation p; each sheet
    x = p Wq^+ + t d has the same Q route and a different FP target.
    """

    decomposition: MapDecomposition
    preactivations: torch.Tensor
    path_lengths: tuple[int, ...]
    offsets: tuple[float, float]
    sheets: tuple[ErrorLedger, ErrorLedger]
    pair: torch.Tensor
    ledger: ErrorLedger


def build_typical_ledger_story() -> TypicalLedgerStory:
    rectangular = build_rectangular_measurement_story()
    decomposition, d = rectangular.decomposition, rectangular.fiber_direction
    p, lengths = _geometry([[.6, -.3], [.6, -.3]])
    offsets = (.3, -.3)
    lift = p @ torch.linalg.pinv(decomposition.quantized_weight)
    sheets = tuple(ErrorLedger.compute(lift + t*d, decomposition) for t in offsets)
    return TypicalLedgerStory(decomposition, p, lengths, offsets, sheets, rectangular.points,
                              ErrorLedger.compute(rectangular.points, decomposition))


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
            "inputs": values(self.points),
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
    # Follow one map and the same A/B inputs from deformation through ReLU.
    repair_weights = [[1., -.4], [.65, .85]]
    repair_probes = [[-.8, -.1], [-.6, -.2]]
    specifications = [
        ("deformation", "The quantized layer deforms the output; a linear map undoes it", repair_weights, repair_probes),
        ("relu", "ReLU first merges points that correcting first keeps apart", repair_weights, repair_probes),
        ("loss", "Quantization collapses a dimension", [[1., .3], [.25, .25]], [[.4, -.8], [.4, .8]]),
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
            labels = ("Apply rounded weights: xWq", "Correct before ReLU: (xWq)R = xW",
                      "Then ReLU: distinct FP outputs", "Instead: ReLU first merges A and B",
                      "Correct after ReLU: A and B stay merged")
            references, indices = (None, None, None, 2, 2), (0, 2, 3, 4)
        else:
            states = (x, x @ W, x @ Wq, x @ Wq @ decomp.R,
                      x @ Wq @ decomp.R + x @ decomp.U, ledger.fp_post)
            labels = ("Input", "Full-precision map", "Apply rounded weights: xWq is a line", "Correct map: still a line",
                      "Add input-dependent xU", "Then ReLU: FP restored")
            references, indices = (None, None, 1, 1, 1, None), (1, 2, 3, 4)
        results.append(GeometryStory(key, title, labels, x, lengths, states,
                                     references, indices, decomp, ledger))
    return tuple(results)
