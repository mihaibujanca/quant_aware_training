#!/usr/bin/env python3
"""Curvature-adaptive boundary compression runner."""

from __future__ import annotations

import argparse
import json
import os
import time

from aleph.boundary_geometric_compression import (
    BoundaryCompressionConfig,
    run_boundary_compression_program,
    save_boundary_compression_results,
)


def parse_csv(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_int_csv(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def summarize_program_runs(results_by_method: dict[str, dict[str, dict]]) -> dict[str, dict]:
    metric_paths = {
        "source_acc": ("metrics", "source_mlp", "test_accuracy"),
        "small_acc": ("metrics", "small_relu_32x4", "test_accuracy"),
        "hybrid_acc": ("metrics", "hybrid_boundary", "test_accuracy"),
        "ambient_tiny_acc": ("metrics", "ambient_tiny_relu", "test_accuracy"),
        "projected_tiny_acc": ("metrics", "projected_tiny_relu", "test_accuracy"),
        "polyline_acc": ("metrics", "projected_global_polyline", "test_accuracy"),
        "spline_acc": ("metrics", "projected_global_spline", "test_accuracy"),
        "hybrid_trainable": ("metrics", "hybrid_boundary", "trainable_params"),
        "hybrid_stored": ("metrics", "hybrid_boundary", "stored_numbers"),
        "hybrid_ratio_vs_small_32x4": (
            "metrics",
            "hybrid_boundary",
            "compression_ratio_vs_small_32x4",
        ),
        "projection_alignment": ("teacher_profile", "projection_subspace_alignment"),
        "complexity_correlation": ("teacher_profile", "complexity_correlation"),
    }

    def get_path(item: dict, path: tuple[str, ...]) -> float:
        cur = item
        for key in path:
            cur = cur[key]
        return float(cur)

    summary: dict[str, dict] = {}
    for method, runs in results_by_method.items():
        all_datasets = sorted({dataset for run in runs.values() for dataset in run["gate_results"].keys()})
        method_summary = {
            "retained_datasets": [],
            "datasets": {},
        }
        for dataset in all_datasets:
            gate_pass_count = 0
            gate_acc_gaps: list[float] = []
            gate_bce_ratios: list[float] = []
            retained_seeds: list[int] = []
            dims: dict[str, dict[str, list[float]]] = {}
            for seed_text, run in runs.items():
                gate = run["gate_results"][dataset]["gate"]
                if gate["passed"]:
                    gate_pass_count += 1
                gate_acc_gaps.append(float(gate["source_minus_small_accuracy"]))
                gate_bce_ratios.append(float(gate["source_over_small_bce_ratio"]))
                if dataset not in run["results"]:
                    continue
                retained_seeds.append(int(seed_text))
                for dim, item in run["results"][dataset].items():
                    bucket = dims.setdefault(dim, {name: [] for name in metric_paths})
                    for name, path in metric_paths.items():
                        bucket[name].append(get_path(item, path))
            if retained_seeds:
                method_summary["retained_datasets"].append(dataset)
            dim_summary = {
                dim: {
                    name: sum(values) / len(values)
                    for name, values in metrics.items()
                    if values
                }
                for dim, metrics in dims.items()
            }
            method_summary["datasets"][dataset] = {
                "gate_pass_count": gate_pass_count,
                "gate_runs": len(runs),
                "mean_gate_accuracy_gap": sum(gate_acc_gaps) / len(gate_acc_gaps),
                "mean_gate_bce_ratio": sum(gate_bce_ratios) / len(gate_bce_ratios),
                "retained_seeds": retained_seeds,
                "dims": dim_summary,
            }
        summary[method] = method_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Curvature-adaptive boundary compression")
    parser.add_argument("--datasets", type=str, default="windowed_ripple_ring,serpentine_tube,corrugated_dumbbell")
    parser.add_argument("--observed-dims", type=str, default="2,100,1000")
    parser.add_argument("--gate-observed-dim", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=2400)
    parser.add_argument("--source-epochs", type=int, default=1000)
    parser.add_argument("--baseline-epochs", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--projection-method", type=str, default="manifold_pca")
    parser.add_argument("--projection-methods", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    methods = parse_csv(args.projection_methods) if args.projection_methods else (args.projection_method,)
    seeds = parse_int_csv(args.seeds) if args.seeds else (args.seed,)

    if len(methods) == 1 and len(seeds) == 1:
        cfg = BoundaryCompressionConfig(
            datasets=parse_csv(args.datasets),
            observed_dims=parse_int_csv(args.observed_dims),
            gate_observed_dim=args.gate_observed_dim,
            n_samples=args.n_samples,
            source_epochs=args.source_epochs,
            baseline_epochs=args.baseline_epochs,
            grid_size=args.grid_size,
            projection_method=methods[0],
            seed=seeds[0],
            device=args.device,
        )
        results = run_boundary_compression_program(cfg)

        output = args.output or os.path.join(
            os.path.dirname(__file__),
            "..",
            "runs",
            f"curvature_boundary_compression_{seeds[0]}.json",
        )
        os.makedirs(os.path.dirname(output), exist_ok=True)
        save_boundary_compression_results(results, output)

        summary = {}
        for dataset, gate_result in results["gate_results"].items():
            summary[dataset] = {"gate_passed": gate_result["passed"]}
            if dataset not in results["results"]:
                continue
            summary[dataset]["dims"] = {}
            for dim, vals in results["results"][dataset].items():
                metrics = vals["metrics"]
                summary[dataset]["dims"][dim] = {
                    "source_acc": metrics["source_mlp"]["test_accuracy"],
                    "small_acc": metrics["small_relu_32x4"]["test_accuracy"],
                    "hybrid_acc": metrics["hybrid_boundary"]["test_accuracy"],
                    "hybrid_trainable": metrics["hybrid_boundary"]["trainable_params"],
                    "polyline_acc": metrics["projected_global_polyline"]["test_accuracy"],
                    "spline_acc": metrics["projected_global_spline"]["test_accuracy"],
                }
        print(json.dumps(summary, indent=2))
        print(f"Saved results to {output}")
        return

    base_kwargs = {
        "datasets": parse_csv(args.datasets),
        "observed_dims": parse_int_csv(args.observed_dims),
        "gate_observed_dim": args.gate_observed_dim,
        "n_samples": args.n_samples,
        "source_epochs": args.source_epochs,
        "baseline_epochs": args.baseline_epochs,
        "grid_size": args.grid_size,
        "device": args.device,
    }
    all_results: dict[str, dict[str, dict]] = {}
    method_times: dict[str, float] = {}
    total_start = time.perf_counter()
    for method in methods:
        method_start = time.perf_counter()
        per_seed: dict[str, dict] = {}
        for seed in seeds:
            print(f"=== {method} seed {seed} start ===", flush=True)
            run_start = time.perf_counter()
            cfg = BoundaryCompressionConfig(
                **base_kwargs,
                projection_method=method,
                seed=seed,
            )
            per_seed[str(seed)] = run_boundary_compression_program(cfg)
            elapsed = time.perf_counter() - run_start
            print(f"=== {method} seed {seed} done in {elapsed:.2f}s ===", flush=True)
        all_results[method] = per_seed
        method_times[method] = time.perf_counter() - method_start

    payload = {
        "experiment": "curvature_boundary_compression_compare",
        "methods": list(methods),
        "seeds": list(seeds),
        "wall_time_seconds": time.perf_counter() - total_start,
        "per_method_wall_time_seconds": method_times,
        "summary": summarize_program_runs(all_results),
        "results_by_method": all_results,
    }
    output = args.output or os.path.join(
        os.path.dirname(__file__),
        "..",
        "runs",
        "curvature_boundary_compression_compare.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({
        "wall_time_seconds": payload["wall_time_seconds"],
        "per_method_wall_time_seconds": payload["per_method_wall_time_seconds"],
        "summary": payload["summary"],
    }, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
