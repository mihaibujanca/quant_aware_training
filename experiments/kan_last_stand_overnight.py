#!/usr/bin/env python
"""Decisive overnight KAN viability sweep across 3D shapes.

Goal: give a clear keep/pivot signal by training under favorable settings
while keeping KAN capacity low.

Default behavior:
- Shapes: torus, sphere, capsule, box, cylinder, bunny mesh
- Methods: source_mlp, geometric_kan, distilled_kan, kan_scratch, small_mlp
- Capacity: KAN width [3, 8, 1] (~335 params), grid=8
- Strong training budget (no speed tier): many points + long training

Outputs:
- CSV with per-shape metrics
- JSON summary with KEEP/PIVOT verdict
- Markdown summary for quick morning read
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aleph.geometric_compression import GeometricCompressionConfig, run_pipeline


DEFAULT_SHAPES = ["torus", "sphere", "capsule", "box", "cylinder", "notebooks/meshes/bunny.ply"]
METHODS = ("source_mlp", "geometric_kan", "distilled_kan", "kan_scratch", "small_mlp")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_run_dir = project_root / "runs" / f"kan_last_stand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Decisive overnight KAN viability sweep.")
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, help="Shape names/mesh paths.")
    parser.add_argument("--epochs", type=int, default=800, help="Source/baseline Adam epochs.")
    parser.add_argument("--kan-steps", type=int, default=800, help="Geometric KAN L-BFGS steps.")
    parser.add_argument("--n-points", type=int, default=120_000)
    parser.add_argument("--n-fit", type=int, default=20_000)
    parser.add_argument("--n-profile", type=int, default=20_000)
    parser.add_argument("--jacobian-eval-points", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(default_run_dir),
        help="Base directory for run artifacts when output paths are not explicitly set.",
    )
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-summary", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    parser.add_argument(
        "--decision-ratio",
        type=float,
        default=0.95,
        help="A KAN method counts as a win if mse <= decision_ratio * small_mlp_mse.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device.",
    )
    return parser.parse_args()


def _safe_float(value):
    if isinstance(value, (float, int)):
        return float(value)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _geometric_mean(values: list[float]) -> float:
    vals = [v for v in values if v > 0 and math.isfinite(v)]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main() -> None:
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"
    run_dir = Path(args.run_dir)
    output_csv = Path(args.output_csv) if args.output_csv else run_dir / "kan_last_stand_results.csv"
    output_summary = Path(args.output_summary) if args.output_summary else run_dir / "kan_last_stand_summary.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "kan_last_stand_summary.md"

    shapes = []
    for s in args.shapes:
        p = Path(s)
        if p.suffix.lower() in {".ply", ".obj", ".stl"} and not p.exists():
            print(f"[skip] mesh path not found: {s}")
            continue
        shapes.append(s)

    if not shapes:
        raise RuntimeError("No valid shapes to run.")

    base_cfg = GeometricCompressionConfig(
        mesh_name="torus",
        n_points=args.n_points,
        epochs=args.epochs,
        kan_fit_steps=args.kan_steps,
        n_fit=args.n_fit,
        n_profile=args.n_profile,
        jacobian_eval_points=args.jacobian_eval_points,
        batch_size=args.batch_size,
        methods=METHODS,
        compute_surface_metrics=False,
        speed_tier=None,
        embed_dim=None,
        n_clusters=8,
        kan_grid=8,
        device=device,
    )

    rows: list[dict] = []
    print(f"Device: {device}")
    print("Decision rule:")
    print(
        f"  win if method_mse <= {args.decision_ratio:.2f} * small_mlp_mse "
        "(per shape)"
    )

    for idx, shape in enumerate(shapes, start=1):
        print(f"\n[{idx}/{len(shapes)}] shape={shape}")
        cfg = replace(base_cfg, mesh_name=shape)
        try:
            result = run_pipeline(cfg)
            metrics = result["metrics"]

            row = {
                "shape": shape,
                "error": "",
            }
            for method in METHODS:
                m = metrics.get(method, {})
                row[f"{method}_mse"] = _safe_float(m.get("test_mse"))
                row[f"{method}_jac_mse"] = _safe_float(m.get("jacobian_mse"))
                row[f"{method}_params"] = int(m.get("params", 0) or 0)
                row[f"{method}_compression"] = _safe_float(m.get("compression_ratio"))

            small = row["small_mlp_mse"]
            for method in ("geometric_kan", "distilled_kan", "kan_scratch"):
                m = row[f"{method}_mse"]
                row[f"{method}_to_small_ratio"] = (
                    m / small if small > 0 and math.isfinite(small) and math.isfinite(m) else float("nan")
                )
                row[f"{method}_win"] = (
                    int(row[f"{method}_to_small_ratio"] <= args.decision_ratio)
                    if math.isfinite(row[f"{method}_to_small_ratio"])
                    else 0
                )

            # Best KAN per shape: strongest possible case for keeping KANs.
            best_kan = min(
                row["geometric_kan_mse"],
                row["distilled_kan_mse"],
                row["kan_scratch_mse"],
            )
            row["best_kan_mse"] = best_kan
            row["best_kan_to_small_ratio"] = best_kan / small if small > 0 and math.isfinite(small) else float("nan")
            row["best_kan_win"] = (
                int(row["best_kan_to_small_ratio"] <= args.decision_ratio)
                if math.isfinite(row["best_kan_to_small_ratio"])
                else 0
            )

            rows.append(row)
            print(
                "  mse: geo={:.6g} distill={:.6g} scratch={:.6g} small={:.6g} "
                "best/small={:.3f}".format(
                    row["geometric_kan_mse"],
                    row["distilled_kan_mse"],
                    row["kan_scratch_mse"],
                    row["small_mlp_mse"],
                    row["best_kan_to_small_ratio"],
                )
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append({"shape": shape, "error": str(exc)})

    if not rows:
        raise RuntimeError("No results were produced.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]

    ratios_best = [r.get("best_kan_to_small_ratio", float("nan")) for r in valid]
    ratios_geo = [r.get("geometric_kan_to_small_ratio", float("nan")) for r in valid]

    wins_best = sum(int(r.get("best_kan_win", 0)) for r in valid)
    wins_geo = sum(int(r.get("geometric_kan_win", 0)) for r in valid)
    need_wins = math.ceil(len(valid) / 2) if valid else 0

    median_best = float(torch.tensor([x for x in ratios_best if math.isfinite(x)]).median().item()) if any(
        math.isfinite(x) for x in ratios_best
    ) else float("nan")
    geom_mean_best = _geometric_mean([x for x in ratios_best if math.isfinite(x)])
    median_geo = float(torch.tensor([x for x in ratios_geo if math.isfinite(x)]).median().item()) if any(
        math.isfinite(x) for x in ratios_geo
    ) else float("nan")

    verdict = "PIVOT"
    reason = "Best KAN does not beat small MLP on enough shapes."
    if valid and wins_best >= need_wins and median_best <= args.decision_ratio:
        verdict = "KEEP"
        reason = "Best KAN beats small MLP on majority of shapes with favorable settings."

    summary = {
        "verdict": verdict,
        "reason": reason,
        "decision_ratio": args.decision_ratio,
        "n_shapes_total": len(rows),
        "n_shapes_success": len(valid),
        "n_shapes_failed": len(failed),
        "required_wins_for_keep": need_wins,
        "wins_geometric_kan": wins_geo,
        "wins_best_kan": wins_best,
        "median_best_kan_to_small_ratio": median_best,
        "geometric_mean_best_kan_to_small_ratio": geom_mean_best,
        "median_geometric_kan_to_small_ratio": median_geo,
        "output_csv": str(output_csv),
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, indent=2))

    lines = [
        "# KAN Last-Stand Overnight Summary",
        "",
        f"- Verdict: **{summary['verdict']}**",
        f"- Reason: {summary['reason']}",
        f"- Shapes (success/total): {summary['n_shapes_success']}/{summary['n_shapes_total']}",
        f"- Decision ratio threshold: {summary['decision_ratio']:.2f}",
        f"- Required wins for KEEP: {summary['required_wins_for_keep']}",
        f"- Geometric KAN wins vs small MLP: {summary['wins_geometric_kan']}",
        f"- Best-KAN wins vs small MLP: {summary['wins_best_kan']}",
        f"- Median(best_kan/small_mlp): {summary['median_best_kan_to_small_ratio']:.4f}",
        f"- Geometric mean(best_kan/small_mlp): {summary['geometric_mean_best_kan_to_small_ratio']:.4f}",
        f"- Median(geometric_kan/small_mlp): {summary['median_geometric_kan_to_small_ratio']:.4f}",
        "",
        f"CSV: `{output_csv}`",
        f"JSON: `{output_summary}`",
    ]
    output_md.write_text("\n".join(lines) + "\n")

    print("\n=== FINAL SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nWrote: {output_csv}")
    print(f"Wrote: {output_summary}")
    print(f"Wrote: {output_md}")


if __name__ == "__main__":
    main()
