#!/usr/bin/env python3
"""Adaptive-width structured pruning on rough spirals."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aleph.datasets import make_embedded_classification_benchmark
from aleph.models import MLP
from aleph.relu_geometric_compression import (
    _device_for,
    _eval_classifier,
    _importance_scores_from_source,
    _linear_layers,
    _make_classifier,
    _mlp_param_count,
    _set_seed,
    _targets,
    _tensor,
    HighDReluCompressionConfig,
)


class VariableWidthMLP(nn.Module):
    def __init__(self, input_dim: int, widths: list[int], output_dim: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in widths:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def param_count(input_dim: int, widths: list[int], output_dim: int = 1) -> int:
    total = 0
    prev = input_dim
    for width in widths:
        total += prev * width + width
        prev = width
    total += prev * output_dim + output_dim
    return total


def fit_best_checkpoint(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_bce = float("inf")
    best_state = None
    best_test = None
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        opt.step()
        val = _eval_classifier(model, X_val, y_val)
        if val["bce"] < best_bce:
            best_bce = float(val["bce"])
            best_state = copy.deepcopy(model.state_dict())
            best_test = _eval_classifier(model, X_test, y_test)
    if best_state is not None:
        model.load_state_dict(best_state)
    test = best_test or _eval_classifier(model, X_test, y_test)
    val = _eval_classifier(model, X_val, y_val)
    return {
        "val_bce": float(val["bce"]),
        "val_accuracy": float(val["accuracy"]),
        "test_bce": float(test["bce"]),
        "test_accuracy": float(test["accuracy"]),
        "test_margin": float(test["mean_margin"]),
    }


def derive_adaptive_widths(
    importance_scores: list[torch.Tensor],
    *,
    input_dim: int,
    target_params: int,
    min_width: int = 4,
    max_width: int = 96,
) -> list[int]:
    masses = np.asarray([float(score.sum().item()) for score in importance_scores], dtype=np.float64)
    masses = np.maximum(masses, 1e-6)
    masses = masses / masses.mean()

    low = 0.1
    high = float(max_width)
    best = [min_width] * len(masses)
    for _ in range(60):
        mid = 0.5 * (low + high)
        widths = np.clip(np.round(mid * masses), min_width, max_width).astype(int).tolist()
        params = param_count(input_dim, widths)
        if params <= target_params:
            best = widths
            low = mid
        else:
            high = mid
    return best


def build_adaptive_pruned_student(
    source_model: MLP,
    X_train: torch.Tensor,
    *,
    target_params: int,
) -> tuple[VariableWidthMLP, list[int]]:
    source_linears = _linear_layers(source_model)
    importance = _importance_scores_from_source(source_model, X_train)
    widths = derive_adaptive_widths(
        importance,
        input_dim=source_linears[0].in_features,
        target_params=target_params,
        max_width=source_linears[0].out_features,
    )
    student = VariableWidthMLP(source_linears[0].in_features, widths, output_dim=1)

    with torch.no_grad():
        dst_linears = [layer for layer in student.layers if isinstance(layer, nn.Linear)]
        prev_keep = None
        for layer_idx, keep_count in enumerate(widths):
            scores = importance[layer_idx]
            keep = torch.topk(scores, k=keep_count).indices.sort().values
            if prev_keep is None:
                dst_linears[layer_idx].weight.copy_(source_linears[layer_idx].weight[keep])
            else:
                dst_linears[layer_idx].weight.copy_(source_linears[layer_idx].weight[keep][:, prev_keep])
            dst_linears[layer_idx].bias.copy_(source_linears[layer_idx].bias[keep])
            prev_keep = keep
        dst_linears[-1].weight.copy_(source_linears[-1].weight[:, prev_keep])
        dst_linears[-1].bias.copy_(source_linears[-1].bias)
    return student, widths

def run_single(
    seed: int,
    observed_dim: int,
    source_hidden: int,
    source_depth: int,
    *,
    epochs: int,
    source_lr: float,
    finetune_lr: float,
) -> dict[str, object]:
    cfg = HighDReluCompressionConfig(
        dataset="rough_spirals",
        observed_dims=(observed_dim,),
        n_samples=2400,
        source_hidden=source_hidden,
        source_depth=source_depth,
        source_epochs=epochs,
        baseline_epochs=epochs,
        cleanup_epochs=epochs,
        device="auto",
        seed=seed,
    )
    _set_seed(seed)
    device = _device_for(cfg)
    benchmark = make_embedded_classification_benchmark(
        dataset="rough_spirals",
        observed_dim=observed_dim,
        n_samples=cfg.n_samples,
        random_state=seed,
    )
    X_train = _tensor(benchmark.train.X_high, device)
    y_train = _targets(benchmark.train.y, device)
    X_val = _tensor(benchmark.val.X_high, device)
    y_val = _targets(benchmark.val.y, device)
    X_test = _tensor(benchmark.test.X_high, device)
    y_test = _targets(benchmark.test.y, device)

    source = _make_classifier(observed_dim, source_hidden, source_depth).to(device)
    source_metrics = {
        "params": float(sum(p.numel() for p in source.parameters())),
        **fit_best_checkpoint(
            source,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            epochs=epochs,
            lr=source_lr,
        ),
    }

    target_params = _mlp_param_count(observed_dim, 32, source_depth)
    uniform = _make_classifier(observed_dim, 32, source_depth).to(device)
    uniform_metrics = {
        "params": float(sum(p.numel() for p in uniform.parameters())),
        **fit_best_checkpoint(
            uniform,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            epochs=epochs,
            lr=source_lr,
        ),
    }

    adaptive, widths = build_adaptive_pruned_student(source, X_train, target_params=target_params)
    adaptive = adaptive.to(device)
    adaptive_metrics = fit_best_checkpoint(
        adaptive,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        lr=finetune_lr,
    )
    adaptive_metrics["params"] = float(param_count(observed_dim, widths))
    return {
        "source": source_metrics,
        "uniform_small": uniform_metrics,
        "adaptive_pruned": adaptive_metrics,
        "adaptive_widths": widths,
        "target_params": target_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive structured pruning on rough spirals")
    parser.add_argument("--observed-dims", type=str, default="100,1000")
    parser.add_argument("--source-hidden", type=int, default=96)
    parser.add_argument("--source-depth", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    observed_dims = [int(x) for x in args.observed_dims.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    output = args.output or os.path.join(
        os.path.dirname(__file__),
        "..",
        "runs",
        "rough_spirals_structured_pruning.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    start = time.perf_counter()
    results = {}
    for dim in observed_dims:
        rows = []
        for seed in seeds:
            print(f"=== dim {dim} seed {seed} start ===", flush=True)
            rows.append(
                run_single(
                    seed,
                    dim,
                    args.source_hidden,
                    args.source_depth,
                    epochs=args.epochs,
                    source_lr=args.source_lr,
                    finetune_lr=args.finetune_lr,
                )
            )
            print(f"=== dim {dim} seed {seed} done ===", flush=True)
        results[str(dim)] = {
            "rows": rows,
            "mean_source_acc": float(np.mean([r["source"]["test_accuracy"] for r in rows])),
            "mean_uniform_small_acc": float(np.mean([r["uniform_small"]["test_accuracy"] for r in rows])),
            "mean_adaptive_pruned_acc": float(np.mean([r["adaptive_pruned"]["test_accuracy"] for r in rows])),
            "mean_gap_adaptive_vs_uniform": float(
                np.mean([r["adaptive_pruned"]["test_accuracy"] - r["uniform_small"]["test_accuracy"] for r in rows])
            ),
            "mean_gap_source_vs_adaptive": float(
                np.mean([r["source"]["test_accuracy"] - r["adaptive_pruned"]["test_accuracy"] for r in rows])
            ),
        }

    payload = {
        "experiment": "rough_spirals_structured_pruning",
        "wall_time_seconds": time.perf_counter() - start,
        "results": results,
    }
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
