#!/usr/bin/env python
"""Jacobian lambda annealing sweep for high-D embedding experiments.

Tests whether annealing the Jacobian constraint (warmup/cooldown) helps
geometric KAN achieve better MSE without sacrificing structural fidelity.

Usage:
    python experiments/jacobian_annealing_sweep.py
    python experiments/jacobian_annealing_sweep.py --dims 3 100 --steps 100
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
    jacobian_rank_profile,
    run_pipeline,
)

CONFIGS = [
    {"jacobian_lambda": 0.0, "jacobian_schedule": "constant", "label": "lambda=0 (MSE only)"},
    {"jacobian_lambda": 1.0, "jacobian_schedule": "constant", "label": "lambda=1 constant"},
    {"jacobian_lambda": 1.0, "jacobian_schedule": "warmup", "label": "lambda=1 warmup"},
    {"jacobian_lambda": 1.0, "jacobian_schedule": "cooldown", "label": "lambda=1 cooldown"},
]


def parse_args():
    project_root = Path(__file__).resolve().parent.parent
    default_run_dir = project_root / "runs" / f"annealing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Jacobian annealing sweep")
    parser.add_argument("--dims", nargs="+", type=int, default=[3, 100])
    parser.add_argument("--steps", type=int, default=100, help="KAN L-BFGS fit steps")
    parser.add_argument("--n-fit", type=int, default=5000, help="Number of fitting points")
    parser.add_argument("--epochs", type=int, default=300, help="Source MLP epochs")
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
    out_path = Path(args.output) if args.output else Path(args.run_dir) / "annealing_results.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_cfg = GeometricCompressionConfig(
        mesh_name="torus",
        epochs=args.epochs,
        kan_fit_steps=args.steps,
        n_fit=args.n_fit,
        device=device,
        methods=("source_mlp", "geometric_kan"),
        speed_tier="explore",
    )

    rows = []
    for embed_dim in args.dims:
        cfg_dim = replace(base_cfg, embed_dim=None if embed_dim == 3 else embed_dim)
        for variant in CONFIGS:
            cfg = replace(cfg_dim,
                          jacobian_lambda=variant["jacobian_lambda"],
                          jacobian_schedule=variant["jacobian_schedule"])
            label = variant["label"]
            dim_label = f"d={embed_dim}"
            print(f"\n{'='*60}")
            print(f"  {dim_label} | {label}")
            print(f"{'='*60}")

            results = run_pipeline(cfg)
            m = results["metrics"]["geometric_kan"]

            # Jacobian SVD analysis on geometric KAN
            # Reconstruct model from state dict to compute Jacobian profile
            from aleph.geometric_compression import _make_kan
            cfg_used = results["config"]
            input_dim = 3 if embed_dim == 3 else embed_dim
            kan_width = [input_dim, int(cfg_used["n_clusters"]), 1]
            kan_model = _make_kan(
                kan_width,
                int(cfg_used["kan_grid"]),
                int(cfg_used["seed"]),
                device,
                kan_k=int(cfg_used.get("kan_k", 3)),
                rbf_width_scale=float(cfg_used.get("rbf_width_scale", 1.0)),
            )
            kan_model.load_state_dict(
                {k: v.to(device) for k, v in results["kan_state_dict"].items()}
            )

            X_eval = results["X_profile"].to(device)
            prof = jacobian_rank_profile(kan_model, X_eval, chunk_size=512)
            energy = prof["energy_ratio"].numpy()

            row = {
                "embed_dim": embed_dim,
                "label": label,
                "jacobian_lambda": variant["jacobian_lambda"],
                "jacobian_schedule": variant["jacobian_schedule"],
                "test_mse": m["test_mse"],
                "jacobian_mse": m["jacobian_mse"],
                "effective_rank": float(prof["effective_rank"]),
                "top3_energy": float(energy[:3].sum()),
                "rank_95": int(prof["rank_95"]),
            }
            rows.append(row)
            print(f"  test_mse={row['test_mse']:.6f}  jac_mse={row['jacobian_mse']:.6f}  "
                  f"eff_rank={row['effective_rank']:.2f}  top3={row['top3_energy']:.4f}")

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'dim':>4} {'schedule':<20} {'test_mse':>10} {'jac_mse':>10} {'eff_rank':>9} {'top3':>7} {'r95':>4}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['embed_dim']:>4} {r['label']:<20} {r['test_mse']:>10.6f} {r['jacobian_mse']:>10.6f} "
              f"{r['effective_rank']:>9.2f} {r['top3_energy']:>7.4f} {r['rank_95']:>4}")

    # Save CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
