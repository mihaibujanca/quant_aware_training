#!/usr/bin/env python3
"""High-D-first geometric compression for ReLU MLPs.

Runs the fresh classification-oriented compression program on embedded
low-dimensional datasets. Native 2D is kept as a reference lane only.
"""

from __future__ import annotations

import argparse
import json
import os

from aleph.relu_geometric_compression import (
    HighDReluCompressionConfig,
    run_high_d_first_program,
    save_high_d_program_results,
)


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="High-D-first ReLU geometric compression")
    parser.add_argument("--dataset", type=str, default="spirals", choices=["spirals", "rough_spirals", "moons"])
    parser.add_argument("--observed-dims", type=str, default="2,100,1000")
    parser.add_argument("--n-samples", type=int, default=2400)
    parser.add_argument("--source-hidden", type=int, default=64)
    parser.add_argument("--source-depth", type=int, default=4)
    parser.add_argument("--source-epochs", type=int, default=800)
    parser.add_argument("--baseline-epochs", type=int, default=240)
    parser.add_argument("--cleanup-epochs", type=int, default=180)
    parser.add_argument("--trace-every-n-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = HighDReluCompressionConfig(
        dataset=args.dataset,
        observed_dims=parse_int_list(args.observed_dims),
        n_samples=args.n_samples,
        source_hidden=args.source_hidden,
        source_depth=args.source_depth,
        source_epochs=args.source_epochs,
        baseline_epochs=args.baseline_epochs,
        cleanup_epochs=args.cleanup_epochs,
        trace_every_n_steps=args.trace_every_n_steps,
        seed=args.seed,
        device=args.device,
    )
    results = run_high_d_first_program(cfg)

    output = args.output or os.path.join(
        os.path.dirname(__file__),
        "..",
        "runs",
        f"high_d_relu_compression_{args.dataset}_{args.seed}.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_high_d_program_results(results, output)

    summary = {
        dim: {
            "source_acc": vals["metrics"]["source_mlp"]["test_accuracy"],
            "geometric_acc": vals["metrics"]["geometric_relu"]["test_accuracy"],
            "distilled_acc": vals["metrics"]["distilled_relu"]["test_accuracy"],
            "small_acc": vals["metrics"]["small_relu"]["test_accuracy"],
            "pruned_acc": vals["metrics"]["pruned_source"]["test_accuracy"],
            "regions": vals["atlas"]["n_regions"],
            "macro_regions": vals["atlas"]["n_macro_regions"],
        }
        for dim, vals in results["results"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
