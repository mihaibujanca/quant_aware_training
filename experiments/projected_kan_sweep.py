#!/usr/bin/env python
"""Projected KAN sweep: decoupled dimensionality reduction + KAN compression.

Tests whether extracting a low-rank projection from the source MLP Jacobian
allows a small KAN to match source MLP accuracy at high input dimensions.

Usage:
    python experiments/projected_kan_sweep.py
    python experiments/projected_kan_sweep.py --dims 3 10 100 --ranks 2 3 5
"""

import argparse
import csv
import sys
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.geometric_compression import (
    GeometricCompressionConfig,
    run_pipeline,
)

DIMS = [3, 10, 50, 100]
RANKS = [2, 3, 5]


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    default_run_dir = project_root / "runs" / f"projected_kan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Projected KAN sweep")
    parser.add_argument("--dims", nargs="+", type=int, default=DIMS)
    parser.add_argument("--ranks", nargs="+", type=int, default=RANKS)
    parser.add_argument("--steps", type=int, default=100, help="KAN L-BFGS fit steps")
    parser.add_argument("--epochs", type=int, default=300, help="Source MLP epochs")
    parser.add_argument("--n-fit", type=int, default=5000)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(default_run_dir),
        help="Base directory for run artifacts when --output is not set.",
    )
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output) if args.output else Path(args.run_dir) / "projected_kan_results.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rows = []

    for dim in args.dims:
        embed_dim = dim if dim > 3 else None

        base_cfg = GeometricCompressionConfig(
            mesh_name="torus",
            epochs=args.epochs,
            kan_fit_steps=args.steps,
            n_fit=args.n_fit,
            embed_dim=embed_dim,
            device=device,
            methods=("source_mlp", "geometric_kan", "projected_kan", "small_mlp"),
            speed_tier="explore",
            compute_surface_metrics=False,
        )

        for rank in args.ranks:
            cfg = replace(base_cfg, projection_rank=rank)
            label = f"d={dim} k={rank}"
            print(f"\n--- {label} ---")

            try:
                result = run_pipeline(cfg)
                metrics = result["metrics"]
                profile = result["geometric_profile"]

                row = {
                    "dim": dim,
                    "rank": rank,
                    "label": label,
                }

                for method in ("source_mlp", "geometric_kan", "projected_kan", "small_mlp"):
                    m = metrics.get(method, {})
                    row[f"{method}_mse"] = m.get("test_mse", float("nan"))
                    row[f"{method}_jac_mse"] = m.get("jacobian_mse", float("nan"))
                    row[f"{method}_params"] = m.get("params", 0)
                    row[f"{method}_compression"] = m.get("compression_ratio", float("nan"))

                row["source_effective_rank"] = profile.get("kan_width", [None])[0]
                row["input_dim"] = profile.get("input_dim", dim)

                rows.append(row)

                # Print summary
                for method in ("source_mlp", "geometric_kan", "projected_kan", "small_mlp"):
                    m = metrics.get(method, {})
                    if m:
                        print(f"  {method:20s}  MSE={m['test_mse']:.6f}  "
                              f"params={m['params']:6d}  ratio={m['compression_ratio']:.1f}x")

            except Exception as e:
                print(f"  FAILED: {e}")
                rows.append({"dim": dim, "rank": rank, "label": label, "error": str(e)})

    # Write CSV
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Label':<20} {'src_MSE':>10} {'geo_MSE':>10} {'proj_MSE':>10} {'mlp_MSE':>10} {'proj_params':>12}")
    for row in rows:
        if "error" in row:
            continue
        print(f"{row['label']:<20} "
              f"{row.get('source_mlp_mse', float('nan')):>10.6f} "
              f"{row.get('geometric_kan_mse', float('nan')):>10.6f} "
              f"{row.get('projected_kan_mse', float('nan')):>10.6f} "
              f"{row.get('small_mlp_mse', float('nan')):>10.6f} "
              f"{row.get('projected_kan_params', 0):>12d}")


if __name__ == "__main__":
    main()
