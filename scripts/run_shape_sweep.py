#!/usr/bin/env python
"""Headless multi-shape geometric compression sweep."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch

# Make sure the package is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.geometric_compression import GeometricCompressionConfig, run_pipeline, save_results

DEFAULT_SHAPES = ["sphere", "torus", "capsule", "box", "cylinder"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-shape geometric compression sweep")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=DEFAULT_SHAPES,
        help=f"Shape names or mesh file paths. Default: {DEFAULT_SHAPES}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/geometric_compression"),
        help="Root output directory. Each shape gets its own subdirectory.",
    )
    parser.add_argument(
        "--speed-tier",
        choices=["explore", "confirm", "none"],
        default="explore",
        help="Runtime preset. Use 'none' to disable speed-tier overrides.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=(
            "Comma-separated methods to run, e.g. "
            "'source_mlp,geometric_kan,small_mlp'. "
            "If omitted, speed tier/default config decides."
        ),
    )
    parser.add_argument(
        "--compute-surface-metrics",
        choices=["on", "off"],
        default="on",
        help="Compute Chamfer/Hausdorff metrics.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs over shapes.")

    parser.add_argument("--n-points", type=int, default=100_000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--eikonal-lambda", type=float, default=0.1)
    parser.add_argument("--eikonal-every-n-steps", type=int, default=1)
    parser.add_argument("--eikonal-subsample", type=int, default=None)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--kan-grid", type=int, default=8)
    parser.add_argument("--kan-k", type=int, default=3)
    parser.add_argument("--rbf-width-scale", type=float, default=1.0)
    parser.add_argument("--jacobian-lambda", type=float, default=1.0)
    parser.add_argument("--jacobian-schedule", choices=["constant", "warmup", "cooldown"], default="constant")
    parser.add_argument("--kan-fit-steps", type=int, default=500)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--n-perturbations", type=int, default=32)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Embed 3D points into this many dimensions (None = native 3D).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _parse_methods(arg: Optional[str]) -> Optional[tuple[str, ...]]:
    if arg is None:
        return None
    methods = tuple(s.strip() for s in arg.split(",") if s.strip())
    return methods or None


def _shape_seed(base_seed: int, shape: str) -> int:
    digest = hashlib.blake2b(shape.encode("utf-8"), digest_size=8).digest()
    seed = int(base_seed) ^ int.from_bytes(digest, byteorder="little", signed=False)
    return int(seed % (2**32 - 1))


def _shape_dir_name(shape: str) -> str:
    return (
        shape.replace("/", "_")
        .replace("\\", "_")
        .replace(".ply", "")
        .replace(".obj", "")
    )


def _format_float(x: object, digits: int = 6) -> str:
    try:
        xf = float(x)
    except Exception:
        return "n/a"
    if math.isnan(xf):
        return "n/a"
    return f"{xf:.{digits}f}"


def _run_shape_job(job: dict) -> dict:
    shape = job["shape"]
    shape_dir = Path(job["shape_dir"])
    torch_threads = int(job["torch_threads"])
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    try:
        cfg = GeometricCompressionConfig(**job["cfg_kwargs"])
        result = run_pipeline(cfg)
        save_results(result, shape_dir)

        p = result["geometric_profile"]
        m = result["metrics"]
        ratio = p["n_params_source"] / p["n_params_kan"]
        geom_metrics = m.get("geometric_kan", {})
        row = {
            "shape": shape,
            "status": "ok",
            "source_mse": p.get("source_test_mse"),
            "geom_kan_mse": geom_metrics.get("test_mse"),
            "chamfer": geom_metrics.get("chamfer", float("nan")),
            "hausdorff": geom_metrics.get("hausdorff", float("nan")),
            "compression_ratio": ratio,
            "watertight": p.get("mesh_watertight"),
            "saved_to": str(shape_dir),
            "metric_keys": sorted(m.keys()),
        }
        return row
    except Exception as exc:
        return {
            "shape": shape,
            "status": f"failed: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()

    if args.kan_k != 3:
        print(
            "Warning: --kan-k is deprecated with FastKAN. "
            "It is mapped to an RBF width scale proxy (clamped k/3)."
        )

    methods = _parse_methods(args.methods)
    speed_tier = None if args.speed_tier == "none" else args.speed_tier
    compute_surface_metrics = args.compute_surface_metrics == "on"
    jobs = max(1, int(args.jobs))

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    total_threads = torch.get_num_threads()
    worker_threads = max(1, total_threads // jobs)

    print(f"Device: {args.device}")
    print(f"Output: {output_root}")
    print(f"Shapes: {args.shapes}")
    print(f"Speed tier: {speed_tier!r}")
    print(f"Methods: {methods if methods is not None else '(tier/default)'}")
    print(f"Surface metrics: {'on' if compute_surface_metrics else 'off'}")
    print(f"Jobs: {jobs} (torch threads/job={worker_threads}, total={total_threads})")
    print()

    jobs_payload: list[dict] = []
    for shape in args.shapes:
        shape_dir = output_root / _shape_dir_name(shape)
        shape_seed = _shape_seed(args.seed, shape)
        cfg_kwargs = {
            "mesh_name": shape,
            "n_points": args.n_points,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "lr": 1e-3,
            "epochs": args.epochs,
            "eikonal_lambda": args.eikonal_lambda,
            "eikonal_every_n_steps": args.eikonal_every_n_steps,
            "eikonal_subsample": args.eikonal_subsample,
            "n_clusters": args.n_clusters,
            "kan_grid": args.kan_grid,
            "kan_k": args.kan_k,
            "rbf_width_scale": args.rbf_width_scale,
            "jacobian_lambda": args.jacobian_lambda,
            "jacobian_schedule": args.jacobian_schedule,
            "kan_fit_steps": args.kan_fit_steps,
            "lbfgs_max_iter": args.lbfgs_max_iter,
            "embed_dim": args.embed_dim,
            "n_perturbations": args.n_perturbations,
            "kmeans_n_init": args.kmeans_n_init,
            "compute_surface_metrics": compute_surface_metrics,
            "speed_tier": speed_tier,
            "device": args.device,
            "seed": shape_seed,
        }
        if methods is not None:
            cfg_kwargs["methods"] = methods
            if compute_surface_metrics:
                cfg_kwargs["surface_metric_methods"] = methods

        jobs_payload.append(
            {
                "shape": shape,
                "shape_dir": str(shape_dir),
                "cfg_kwargs": cfg_kwargs,
                "torch_threads": worker_threads,
            }
        )

    summary_by_shape: dict[str, dict] = {}
    if jobs == 1:
        for payload in jobs_payload:
            shape = payload["shape"]
            print(f"{'='*60}")
            print(f"Shape: {shape}")
            print(f"{'='*60}")
            row = _run_shape_job(payload)
            summary_by_shape[shape] = row
            if row["status"] == "ok":
                print(f"  source_mse   = {_format_float(row['source_mse'])}")
                print(f"  geom_kan_mse = {_format_float(row['geom_kan_mse'])}")
                print(f"  chamfer      = {_format_float(row['chamfer'], 5)}")
                print(f"  hausdorff    = {_format_float(row['hausdorff'], 5)}")
                print(f"  ratio        = {_format_float(row['compression_ratio'], 2)}x")
                print(f"  saved to     = {row['saved_to']}")
                print(f"  metrics      = {row['metric_keys']}")
            else:
                print(f"  FAILED: {row['status']}")
                if row.get("traceback"):
                    print(row["traceback"])
            print()
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_run_shape_job, payload): payload["shape"] for payload in jobs_payload}
            for future in as_completed(futures):
                shape = futures[future]
                row = future.result()
                summary_by_shape[shape] = row
                if row["status"] == "ok":
                    print(
                        f"[ok] {shape}: mse={_format_float(row['geom_kan_mse'])} "
                        f"ch={_format_float(row['chamfer'], 5)} "
                        f"ratio={_format_float(row['compression_ratio'], 2)}x"
                    )
                else:
                    print(f"[failed] {shape}: {row['status']}")
                    if row.get("traceback"):
                        print(row["traceback"])

    summary = [summary_by_shape[s] for s in args.shapes if s in summary_by_shape]

    print("\n" + "=" * 110)
    print(
        f"{'Shape':20s} {'Status':8s} {'Source MSE':>12s} {'GeoKAN MSE':>12s} "
        f"{'Chamfer':>10s} {'Ratio':>8s}"
    )
    print("-" * 110)
    for row in summary:
        if row["status"] == "ok":
            print(
                f"{row['shape']:20s} {'ok':8s} "
                f"{_format_float(row['source_mse']):>12s} "
                f"{_format_float(row['geom_kan_mse']):>12s} "
                f"{_format_float(row['chamfer'], 5):>10s} "
                f"{_format_float(row['compression_ratio'], 1):>7s}x"
            )
        else:
            print(f"{row['shape']:20s} {row['status']}")
    print("=" * 110)
    print(f"\nResults saved under: {output_root}")


if __name__ == "__main__":
    main()
