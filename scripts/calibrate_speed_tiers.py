#!/usr/bin/env python
"""Benchmark-driven calibration for geometric compression speed tiers."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.geometric_compression import GeometricCompressionConfig, run_pipeline


EXPLORE_METHODS = ("source_mlp", "geometric_kan", "small_mlp")
CONFIRM_METHODS = ("source_mlp", "geometric_kan", "distilled_kan", "small_mlp", "kan_scratch")


EXPLORE_CANDIDATES = [
    {
        "name": "explore_fast",
        "overrides": {
            "n_points": 40_000,
            "epochs": 240,
            "kan_fit_steps": 120,
            "lbfgs_max_iter": 12,
            "n_profile": 4_000,
            "n_fit": 2_500,
            "n_perturbations": 12,
            "kmeans_n_init": 8,
            "eikonal_every_n_steps": 1,
            "eikonal_subsample": None,
            "methods": EXPLORE_METHODS,
        },
    },
    {
        "name": "explore_quality",
        "overrides": {
            "n_points": 45_000,
            "epochs": 260,
            "kan_fit_steps": 130,
            "lbfgs_max_iter": 14,
            "n_profile": 4_500,
            "n_fit": 2_800,
            "n_perturbations": 14,
            "kmeans_n_init": 8,
            "eikonal_every_n_steps": 1,
            "eikonal_subsample": None,
            "methods": EXPLORE_METHODS,
        },
    },
    {
        "name": "explore_reference",
        "overrides": {
            "n_points": 50_000,
            "epochs": 300,
            "kan_fit_steps": 150,
            "lbfgs_max_iter": 15,
            "n_profile": 5_000,
            "n_fit": 3_000,
            "n_perturbations": 16,
            "kmeans_n_init": 10,
            "eikonal_every_n_steps": 1,
            "eikonal_subsample": None,
            "methods": EXPLORE_METHODS,
        },
    },
]


CONFIRM_CANDIDATES = [
    {
        "name": "confirm_quality",
        "overrides": {
            "n_points": 45_000,
            "epochs": 260,
            "kan_fit_steps": 130,
            "lbfgs_max_iter": 14,
            "n_profile": 4_500,
            "n_fit": 2_800,
            "n_perturbations": 14,
            "kmeans_n_init": 8,
            "eikonal_every_n_steps": 1,
            "eikonal_subsample": None,
            "methods": CONFIRM_METHODS,
        },
    },
    {
        "name": "confirm_reference",
        "overrides": {
            "n_points": 50_000,
            "epochs": 300,
            "kan_fit_steps": 150,
            "lbfgs_max_iter": 15,
            "n_profile": 5_000,
            "n_fit": 3_000,
            "n_perturbations": 16,
            "kmeans_n_init": 10,
            "eikonal_every_n_steps": 1,
            "eikonal_subsample": None,
            "methods": CONFIRM_METHODS,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate speed tiers from benchmark runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["torus", "sphere"],
        help="One or more shapes to quality-gate candidate settings on.",
    )
    parser.add_argument(
        "--shape",
        default=None,
        help="Deprecated single-shape alias. Overrides --shapes when provided.",
    )
    parser.add_argument("--embed-dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-seconds", type=float, default=90.0)
    parser.add_argument("--target-seconds-confirm", type=float, default=150.0)
    parser.add_argument("--surface-metrics", choices=["on", "off"], default="on")
    parser.add_argument("--quality-gate", choices=["on", "off"], default="on")
    parser.add_argument(
        "--quality-method",
        default="geometric_kan",
        help="Method whose surface metrics are used for quality gating.",
    )
    parser.add_argument(
        "--quality-chamfer-max",
        type=float,
        default=0.055,
        help="Reject candidates whose quality method has higher Chamfer on any shape.",
    )
    parser.add_argument(
        "--quality-hausdorff-max",
        type=float,
        default=0.25,
        help="Reject candidates whose quality method has higher Hausdorff on any shape.",
    )
    parser.add_argument(
        "--fail-on-no-quality",
        choices=["on", "off"],
        default="on",
        help="Exit non-zero when no candidate passes quality gate for a tier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/geometric_compression/calibration"),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap per tier to shorten calibration runs.",
    )
    return parser.parse_args()


def _is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _extract_core_metrics(metrics: dict) -> tuple[dict, bool]:
    core = {}
    finite = True
    for method in ("source_mlp", "geometric_kan", "small_mlp"):
        m = metrics.get(method, {})
        core[method] = {
            "test_mse": m.get("test_mse"),
            "jacobian_mse": m.get("jacobian_mse"),
        }
        if core[method]["test_mse"] is None:
            continue
        finite = finite and _is_finite_number(core[method]["test_mse"])
        finite = finite and _is_finite_number(core[method]["jacobian_mse"])
    return core, finite


def _extract_surface_metrics(metrics: dict, quality_method: str) -> dict:
    out = {}
    for method in ("source_mlp", quality_method):
        m = metrics.get(method, {})
        out[method] = {
            "chamfer": m.get("chamfer"),
            "hausdorff": m.get("hausdorff"),
        }
    return out


def _assess_quality(
    per_shape: dict,
    quality_method: str,
    chamfer_max: float,
    hausdorff_max: float,
) -> dict:
    reasons = []
    for shape, row in per_shape.items():
        quality_metrics = row.get("surface_metrics", {}).get(quality_method, {})
        chamfer = quality_metrics.get("chamfer")
        hausdorff = quality_metrics.get("hausdorff")

        if not _is_finite_number(chamfer):
            reasons.append(f"{shape}: {quality_method}.chamfer is non-finite ({chamfer})")
            continue
        if not _is_finite_number(hausdorff):
            reasons.append(f"{shape}: {quality_method}.hausdorff is non-finite ({hausdorff})")
            continue
        if float(chamfer) > chamfer_max:
            reasons.append(
                f"{shape}: {quality_method}.chamfer={float(chamfer):.5f} > {chamfer_max:.5f}"
            )
        if float(hausdorff) > hausdorff_max:
            reasons.append(
                f"{shape}: {quality_method}.hausdorff={float(hausdorff):.5f} > {hausdorff_max:.5f}"
            )

    return {
        "pass": len(reasons) == 0,
        "method": quality_method,
        "chamfer_max": chamfer_max,
        "hausdorff_max": hausdorff_max,
        "reasons": reasons,
    }


def _run_candidate(
    base_cfg: GeometricCompressionConfig,
    candidate: dict,
    shapes: list[str],
    quality_method: str,
    chamfer_max: float,
    hausdorff_max: float,
    quality_gate: bool,
) -> dict:
    cfg = replace(base_cfg, **candidate["overrides"])
    per_shape = {}
    runtime_sum = 0.0
    finite_core = True

    for shape in shapes:
        cfg_shape = replace(cfg, mesh_name=shape)

        # Warm-up run.
        run_pipeline(cfg_shape)

        t0 = time.perf_counter()
        result = run_pipeline(cfg_shape)
        runtime = time.perf_counter() - t0
        runtime_sum += runtime

        metrics = result["metrics"]
        core, finite = _extract_core_metrics(metrics)
        finite_core = finite_core and finite
        surface = _extract_surface_metrics(metrics, quality_method)

        per_shape[shape] = {
            "runtime_seconds": runtime,
            "core_metrics": core,
            "surface_metrics": surface,
        }

    quality = _assess_quality(
        per_shape=per_shape,
        quality_method=quality_method,
        chamfer_max=chamfer_max,
        hausdorff_max=hausdorff_max,
    )
    if not quality_gate:
        quality["pass"] = True
        quality["reasons"] = []

    return {
        "name": candidate["name"],
        "runtime_seconds_sum": runtime_sum,
        "runtime_seconds_mean": runtime_sum / max(len(shapes), 1),
        "finite_core_metrics": finite_core,
        "overrides": candidate["overrides"],
        "per_shape": per_shape,
        "quality": quality,
        "resolved_config": asdict(cfg),
    }


def _select_candidate(results: list[dict], target_seconds: float, quality_gate: bool) -> dict | None:
    finite = [r for r in results if r["finite_core_metrics"]]
    if quality_gate:
        finite = [r for r in finite if r["quality"]["pass"]]
    if not finite:
        return None
    eligible = [r for r in finite if r["runtime_seconds_mean"] <= target_seconds]
    if eligible:
        return min(eligible, key=lambda r: r["runtime_seconds_mean"])
    return min(finite, key=lambda r: r["runtime_seconds_mean"])


def _write_markdown_report(path: Path, payload: dict) -> None:
    lines = []
    lines.append("# Speed Tier Calibration")
    lines.append("")
    lines.append(f"- Timestamp: {payload['timestamp']}")
    lines.append(f"- Device: `{payload['meta']['device']}`")
    lines.append(f"- Shapes: `{payload['meta']['shapes']}`")
    lines.append(f"- Embed dim: `{payload['meta']['embed_dim']}`")
    lines.append(f"- Surface metrics: `{payload['meta']['surface_metrics']}`")
    lines.append(f"- Quality gate: `{payload['meta']['quality_gate']}`")
    lines.append(
        f"- Quality thresholds ({payload['meta']['quality_method']}): "
        f"chamfer <= `{payload['meta']['quality_chamfer_max']}`, "
        f"hausdorff <= `{payload['meta']['quality_hausdorff_max']}`"
    )
    lines.append("")
    for tier in ("explore", "confirm"):
        lines.append(f"## {tier.title()} Candidates")
        lines.append("")
        lines.append("| Name | Mean Runtime (s/shape) | Finite Core | Quality Pass |")
        lines.append("|---|---:|:---:|:---:|")
        for row in payload["results"][tier]:
            lines.append(
                f"| {row['name']} | {row['runtime_seconds_mean']:.2f} | "
                f"{'yes' if row['finite_core_metrics'] else 'no'} | "
                f"{'yes' if row['quality']['pass'] else 'no'} |"
            )
            for shape in payload["meta"]["shapes"]:
                shape_row = row["per_shape"].get(shape, {})
                q = shape_row.get("surface_metrics", {}).get(payload["meta"]["quality_method"], {})
                ch = q.get("chamfer")
                hd = q.get("hausdorff")
                lines.append(
                    f"  - `{shape}` runtime={shape_row.get('runtime_seconds', float('nan')):.2f}s, "
                    f"chamfer={ch if ch is not None else 'n/a'}, "
                    f"hausdorff={hd if hd is not None else 'n/a'}"
                )
            if row["quality"]["reasons"]:
                lines.append("  - gate failure reasons:")
                for reason in row["quality"]["reasons"]:
                    lines.append(f"    - {reason}")
        sel = payload["selected"][tier]
        lines.append("")
        if sel is None:
            lines.append("Selected: `none` (no candidate passed quality gate).")
        else:
            lines.append(
                f"Selected: `{sel['name']}` at `{sel['runtime_seconds_mean']:.2f}s/shape` "
                f"(target `{payload['targets'][tier]:.2f}s/shape`)."
            )
        lines.append("")

    lines.append("## Suggested _SPEED_TIERS Overrides")
    lines.append("")
    lines.append("```python")
    lines.append("_SPEED_TIERS = {")
    for tier in ("explore", "confirm"):
        sel = payload["selected"][tier]
        if sel is None:
            lines.append(f"    '{tier}': None,  # no passing candidate")
        else:
            lines.append(f"    '{tier}': {sel['overrides']},")
    lines.append("}")
    lines.append("```")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = list(args.shapes)
    if args.shape is not None:
        print("Warning: --shape is deprecated; using single-shape calibration.")
        shapes = [args.shape]
    if len(shapes) == 0:
        raise ValueError("No shapes provided. Use --shapes.")

    embed_dim = None if args.embed_dim == 3 else args.embed_dim
    surface_on = args.surface_metrics == "on"
    quality_gate = args.quality_gate == "on"
    fail_on_no_quality = args.fail_on_no_quality == "on"
    quality_method = args.quality_method

    if quality_gate and not surface_on:
        raise ValueError(
            "Quality gate requires surface metrics. Use --surface-metrics on or --quality-gate off."
        )
    if quality_gate and len(shapes) < 2:
        print(
            "Warning: quality gate is enabled with a single shape. "
            "Use at least two shapes for robust defaults."
        )

    base_cfg = GeometricCompressionConfig(
        mesh_name=shapes[0],
        embed_dim=embed_dim,
        device=args.device,
        seed=args.seed,
        speed_tier=None,
        compute_surface_metrics=surface_on,
    )

    candidates = {
        "explore": EXPLORE_CANDIDATES,
        "confirm": CONFIRM_CANDIDATES,
    }
    if args.max_candidates is not None:
        for tier in candidates:
            candidates[tier] = candidates[tier][: max(1, args.max_candidates)]

    results: dict[str, list[dict]] = {"explore": [], "confirm": []}
    targets = {
        "explore": float(args.target_seconds),
        "confirm": float(args.target_seconds_confirm),
    }

    for tier in ("explore", "confirm"):
        print(f"\n=== Calibrating {tier} tier ===")
        for candidate in candidates[tier]:
            print(f"Running {candidate['name']} ...")
            row = _run_candidate(
                base_cfg=base_cfg,
                candidate=candidate,
                shapes=shapes,
                quality_method=quality_method,
                chamfer_max=float(args.quality_chamfer_max),
                hausdorff_max=float(args.quality_hausdorff_max),
                quality_gate=quality_gate,
            )
            results[tier].append(row)

            shape_summaries = []
            for shape in shapes:
                sr = row["per_shape"][shape]
                q = sr["surface_metrics"].get(quality_method, {})
                shape_summaries.append(
                    f"{shape}: rt={sr['runtime_seconds']:.2f}s "
                    f"ch={q.get('chamfer')} hd={q.get('hausdorff')}"
                )
            print(
                f"  mean_runtime={row['runtime_seconds_mean']:.2f}s/shape "
                f"finite_core={row['finite_core_metrics']} "
                f"quality_pass={row['quality']['pass']}"
            )
            for summary in shape_summaries:
                print(f"    {summary}")
            if row["quality"]["reasons"]:
                for reason in row["quality"]["reasons"]:
                    print(f"    gate_fail: {reason}")

    selected = {
        "explore": _select_candidate(results["explore"], targets["explore"], quality_gate=quality_gate),
        "confirm": _select_candidate(results["confirm"], targets["confirm"], quality_gate=quality_gate),
    }

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "meta": {
            "device": args.device,
            "shapes": shapes,
            "embed_dim": args.embed_dim,
            "surface_metrics": args.surface_metrics,
            "quality_gate": args.quality_gate,
            "quality_method": quality_method,
            "quality_chamfer_max": float(args.quality_chamfer_max),
            "quality_hausdorff_max": float(args.quality_hausdorff_max),
            "seed": args.seed,
        },
        "targets": targets,
        "results": results,
        "selected": selected,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"calibration_{ts}.json"
    md_path = output_dir / f"calibration_{ts}.md"
    latest_path = output_dir / "latest_calibration.json"

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(latest_path, "w") as f:
        json.dump(payload, f, indent=2)
    _write_markdown_report(md_path, payload)

    print("\n=== Selected Candidates ===")
    missing_quality = []
    for tier in ("explore", "confirm"):
        s = selected[tier]
        if s is None:
            print(f"{tier}: none (no candidate passed quality gate)")
            missing_quality.append(tier)
        else:
            print(
                f"{tier}: {s['name']} "
                f"(runtime={s['runtime_seconds_mean']:.2f}s/shape, target={targets[tier]:.2f}s/shape)"
            )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {latest_path}")

    if missing_quality and fail_on_no_quality:
        missing = ", ".join(missing_quality)
        raise SystemExit(
            f"No quality-passing candidate for tier(s): {missing}. "
            "See calibration report for gate failure reasons."
        )


if __name__ == "__main__":
    main()
