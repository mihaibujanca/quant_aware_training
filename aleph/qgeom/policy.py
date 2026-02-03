"""Policy scoring and simulation for geometry-guided correction placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from aleph.qgeom.core import RunGeometryReport


@dataclass
class CorrectionPointScore:
    """Ranked score for a candidate correction point."""

    layer_index: int
    score: float
    predicted_gain: float
    reason: str


@dataclass
class PolicySimulationResult:
    """Output of policy simulation with predicted and observed gains."""

    policy_name: str
    selected_points: List[int]
    predicted_gain: float
    actual_gain: float | None
    details: Dict[str, Any] = field(default_factory=dict)


def _normalize_constraints(constraints: Dict[str, Any] | None) -> Dict[str, Any]:
    if constraints is None:
        constraints = {}
    return {
        "min_gap": int(constraints.get("min_gap", 1)),
        "allowed_points": set(constraints.get("allowed_points", [])),
        "blocked_points": set(constraints.get("blocked_points", [])),
    }


def _candidate_score(linear_error_norm: float, correctability_score: float, entropy_proxy: float, anisotropy_ratio: float) -> float:
    entropy_term = np.log1p(max(entropy_proxy, 0.0))
    anis_term = 1.0 / (1.0 + np.log1p(max(anisotropy_ratio - 1.0, 0.0)))
    return float(linear_error_norm * correctability_score * (1.0 + 0.2 * entropy_term) * (0.5 + 0.5 * anis_term))


def evenly_spaced_points(layer_indices: Sequence[int], budget: int) -> List[int]:
    """Select evenly spaced indices across available layers."""

    uniq = sorted(set(int(i) for i in layer_indices))
    if budget <= 0 or not uniq:
        return []
    if budget >= len(uniq):
        return uniq

    pos = np.linspace(0, len(uniq) - 1, budget)
    picked = sorted({uniq[int(round(p))] for p in pos})

    # If collisions caused fewer picks, fill deterministically from left to right.
    for idx in uniq:
        if len(picked) >= budget:
            break
        if idx not in picked:
            picked.append(idx)
    return sorted(picked[:budget])


def score_correction_points(
    run_report: RunGeometryReport,
    budget: int,
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Rank and pick correction points from a run geometry report."""

    if budget < 0:
        raise ValueError("budget must be >= 0")

    cfg = _normalize_constraints(constraints)
    min_gap = cfg["min_gap"]
    allowed = cfg["allowed_points"]
    blocked = cfg["blocked_points"]

    ranking: List[CorrectionPointScore] = []
    for report in run_report.layer_reports:
        idx = int(report.layer_index)
        if allowed and idx not in allowed:
            continue
        if idx in blocked:
            continue

        score = _candidate_score(
            report.linear_error_norm,
            report.correctability_score,
            report.entropy_proxy,
            report.anisotropy_ratio,
        )
        ranking.append(
            CorrectionPointScore(
                layer_index=idx,
                score=score,
                predicted_gain=score,
                reason=(
                    f"linear={report.linear_error_norm:.4f}, "
                    f"corr={report.correctability_score:.3f}, "
                    f"flip={report.relu_flip_rate:.3f}"
                ),
            )
        )

    ranking.sort(key=lambda r: (-r.score, r.layer_index))

    selected: List[int] = []
    for candidate in ranking:
        if len(selected) >= budget:
            break
        if any(abs(candidate.layer_index - s) < min_gap for s in selected):
            continue
        selected.append(candidate.layer_index)

    return {
        "selected_points": selected,
        "ranking": [
            {
                "layer_index": r.layer_index,
                "score": r.score,
                "predicted_gain": r.predicted_gain,
                "reason": r.reason,
            }
            for r in ranking
        ],
    }


def build_baseline_policy(run_report: RunGeometryReport, budget: int) -> Dict[str, Any]:
    """Create evenly-spaced baseline policy for comparison."""

    layers = [r.layer_index for r in run_report.layer_reports]
    selected = evenly_spaced_points(layers, budget)
    return {
        "selected_points": selected,
        "ranking": [
            {"layer_index": i, "score": 0.0, "predicted_gain": 0.0, "reason": "evenly_spaced"}
            for i in selected
        ],
    }


def simulate_policy(
    policy_spec: Dict[str, Any],
    model: Any,
    data: Any,
    quant_cfg: Dict[str, Any],
) -> PolicySimulationResult:
    """Evaluate policy with predicted gain and optional observed gain callback."""

    selected_points = [int(i) for i in policy_spec.get("selected_points", [])]
    ranking = policy_spec.get("ranking", [])

    score_map = {int(row["layer_index"]): float(row.get("predicted_gain", row.get("score", 0.0))) for row in ranking}
    predicted_gain = float(sum(score_map.get(i, 0.0) for i in selected_points))

    evaluator: Callable[..., Dict[str, Any] | float] | None = quant_cfg.get("policy_evaluator")
    actual_gain: float | None = None
    details: Dict[str, Any] = {}

    if evaluator is not None:
        result = evaluator(
            model=model,
            data=data,
            quant_cfg=quant_cfg,
            selected_points=selected_points,
        )
        if isinstance(result, dict):
            actual_gain = float(result.get("actual_gain", np.nan))
            details.update(result)
        else:
            actual_gain = float(result)
    else:
        observed_gain_by_layer = quant_cfg.get("observed_gain_by_layer")
        if observed_gain_by_layer is not None:
            actual_gain = float(sum(float(observed_gain_by_layer.get(i, 0.0)) for i in selected_points))
            details["source"] = "observed_gain_by_layer"

    return PolicySimulationResult(
        policy_name=str(policy_spec.get("name", "unnamed_policy")),
        selected_points=selected_points,
        predicted_gain=predicted_gain,
        actual_gain=actual_gain,
        details=details,
    )
