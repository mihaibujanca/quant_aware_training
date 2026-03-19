#!/usr/bin/env python3
"""Headroom benchmark for embedded multi-label shape classification."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from aleph.datasets import make_embedded_multilabel_shape_benchmark
from aleph.models import MLP


def parse_csv(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_int_csv(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    logits = model(X)
    bce = float(F.binary_cross_entropy_with_logits(logits, y).item())
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    y_np = y.detach().cpu().numpy().astype(np.float32)
    preds_np = preds.detach().cpu().numpy().astype(np.float32)

    exact_match = float(np.all(preds_np == y_np, axis=1).mean())
    label_accuracy = float((preds_np == y_np).mean())

    tp = (preds_np * y_np).sum(axis=0)
    fp = (preds_np * (1.0 - y_np)).sum(axis=0)
    fn = ((1.0 - preds_np) * y_np).sum(axis=0)
    macro_f1 = float(np.mean((2.0 * tp) / np.maximum(2.0 * tp + fp + fn, 1e-8)))

    tp_all = float(tp.sum())
    fp_all = float(fp.sum())
    fn_all = float(fn.sum())
    micro_f1 = float((2.0 * tp_all) / max(2.0 * tp_all + fp_all + fn_all, 1e-8))

    return {
        "bce": bce,
        "exact_match": exact_match,
        "label_accuracy": label_accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }


def train_with_best_checkpoint(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    lr: float,
    trace_every_n_steps: int = 0,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace: list[dict[str, float]] = []
    best_val_bce = float("inf")
    best_state = None
    best_metrics: dict[str, float] | None = None

    for step in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, X_val, y_val)
        if val_metrics["bce"] < best_val_bce:
            best_val_bce = val_metrics["bce"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                "step": float(step + 1),
                "val_bce": float(val_metrics["bce"]),
                "val_exact_match": float(val_metrics["exact_match"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                **{
                    f"test_{k}": float(v)
                    for k, v in evaluate(model, X_test, y_test).items()
                },
            }

        if trace_every_n_steps > 0 and ((step + 1) % trace_every_n_steps == 0 or step == 0 or step + 1 == epochs):
            test_metrics = evaluate(model, X_test, y_test)
            trace.append(
                {
                    "step": float(step + 1),
                    "train_loss": float(loss.detach().item()),
                    "val_bce": float(val_metrics["bce"]),
                    "val_exact_match": float(val_metrics["exact_match"]),
                    "val_macro_f1": float(val_metrics["macro_f1"]),
                    "test_bce": float(test_metrics["bce"]),
                    "test_exact_match": float(test_metrics["exact_match"]),
                    "test_macro_f1": float(test_metrics["macro_f1"]),
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return trace, best_metrics or {}


def run_single(
    *,
    dataset: str,
    observed_dim: int,
    n_samples: int,
    source_hidden: int,
    source_depth: int,
    small_hidden: int,
    small_depth: int,
    epochs: int,
    lr: float,
    seed: int,
    device_name: str,
    trace_every_n_steps: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device_for(device_name)
    benchmark = make_embedded_multilabel_shape_benchmark(
        dataset=dataset,
        observed_dim=observed_dim,
        n_samples=n_samples,
        random_state=seed,
    )

    X_train = tensor(benchmark.train.X_high, device)
    X_val = tensor(benchmark.val.X_high, device)
    X_test = tensor(benchmark.test.X_high, device)
    y_train = tensor(benchmark.train.y, device)
    y_val = tensor(benchmark.val.y, device)
    y_test = tensor(benchmark.test.y, device)

    output_dim = benchmark.train.y.shape[1]
    source = MLP(input_size=benchmark.observed_dim, hidden_size=source_hidden, output_size=output_dim, depth=source_depth).to(device)
    small = MLP(input_size=benchmark.observed_dim, hidden_size=small_hidden, output_size=output_dim, depth=small_depth).to(device)

    source_trace, source_best = train_with_best_checkpoint(
        source,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        trace_every_n_steps=trace_every_n_steps,
    )
    small_trace, small_best = train_with_best_checkpoint(
        small,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        epochs=epochs,
        lr=lr,
        trace_every_n_steps=trace_every_n_steps,
    )

    source_test = evaluate(source, X_test, y_test)
    small_test = evaluate(small, X_test, y_test)
    source_val = evaluate(source, X_val, y_val)
    small_val = evaluate(small, X_val, y_val)

    return {
        "benchmark": {
            "name": benchmark.name,
            "observed_dim": benchmark.observed_dim,
            "intrinsic_dim": benchmark.intrinsic_dim,
            "n_labels": int(output_dim),
            "label_names": benchmark.label_names,
            "train_size": int(len(benchmark.train.y)),
            "val_size": int(len(benchmark.val.y)),
            "test_size": int(len(benchmark.test.y)),
            "train_label_prevalence": benchmark.train.y.mean(axis=0).astype(float).tolist(),
        },
        "source": {
            "params": float(num_params(source)),
            "val": source_val,
            "test": source_test,
            "best_checkpoint": source_best,
        },
        "small": {
            "params": float(num_params(small)),
            "val": small_val,
            "test": small_test,
            "best_checkpoint": small_best,
        },
        "headroom": {
            "val_exact_match_gap": float(source_val["exact_match"] - small_val["exact_match"]),
            "val_macro_f1_gap": float(source_val["macro_f1"] - small_val["macro_f1"]),
            "val_bce_ratio": float(source_val["bce"] / max(small_val["bce"], 1e-8)),
            "test_exact_match_gap": float(source_test["exact_match"] - small_test["exact_match"]),
            "test_macro_f1_gap": float(source_test["macro_f1"] - small_test["macro_f1"]),
        },
        "training_traces": {
            "source": source_trace,
            "small": small_trace,
        },
    }


def summarize(results_by_seed: dict[str, dict[str, dict[str, object]]]) -> dict[str, dict[str, dict[str, float]]]:
    datasets = sorted({dataset for seed_results in results_by_seed.values() for dataset in seed_results.keys()})
    dims = sorted({dim for seed_results in results_by_seed.values() for ds in seed_results.values() for dim in ds.keys()}, key=int)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        out[dataset] = {}
        for dim in dims:
            rows = []
            for seed_results in results_by_seed.values():
                if dataset in seed_results and dim in seed_results[dataset]:
                    rows.append(seed_results[dataset][dim])
            if not rows:
                continue
            out[dataset][dim] = {
                "source_test_exact_match": float(np.mean([row["source"]["test"]["exact_match"] for row in rows])),
                "small_test_exact_match": float(np.mean([row["small"]["test"]["exact_match"] for row in rows])),
                "source_test_macro_f1": float(np.mean([row["source"]["test"]["macro_f1"] for row in rows])),
                "small_test_macro_f1": float(np.mean([row["small"]["test"]["macro_f1"] for row in rows])),
                "test_exact_match_gap": float(np.mean([row["headroom"]["test_exact_match_gap"] for row in rows])),
                "test_macro_f1_gap": float(np.mean([row["headroom"]["test_macro_f1_gap"] for row in rows])),
                "val_bce_ratio": float(np.mean([row["headroom"]["val_bce_ratio"] for row in rows])),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedded multi-label shape headroom benchmark")
    parser.add_argument("--datasets", type=str, default="overlap_triplet,rough_overlap_triplet")
    parser.add_argument("--observed-dims", type=str, default="2,100,1000")
    parser.add_argument("--n-samples", type=int, default=2400)
    parser.add_argument("--source-hidden", type=int, default=64)
    parser.add_argument("--source-depth", type=int, default=4)
    parser.add_argument("--small-hidden", type=int, default=32)
    parser.add_argument("--small-depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--trace-every-n-steps", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    datasets = parse_csv(args.datasets)
    observed_dims = parse_int_csv(args.observed_dims)
    seeds = parse_int_csv(args.seeds)

    output = args.output or os.path.join(
        os.path.dirname(__file__),
        "..",
        "runs",
        "multilabel_shape_headroom.json",
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    results_by_seed: dict[str, dict[str, dict[str, object]]] = {}
    start = time.perf_counter()
    for seed in seeds:
        print(f"=== seed {seed} start ===", flush=True)
        per_dataset: dict[str, dict[str, object]] = {}
        for dataset in datasets:
            per_dim: dict[str, object] = {}
            for observed_dim in observed_dims:
                per_dim[str(observed_dim)] = run_single(
                    dataset=dataset,
                    observed_dim=observed_dim,
                    n_samples=args.n_samples,
                    source_hidden=args.source_hidden,
                    source_depth=args.source_depth,
                    small_hidden=args.small_hidden,
                    small_depth=args.small_depth,
                    epochs=args.epochs,
                    lr=args.lr,
                    seed=seed,
                    device_name=args.device,
                    trace_every_n_steps=args.trace_every_n_steps,
                )
            per_dataset[dataset] = per_dim
        results_by_seed[str(seed)] = per_dataset
        payload = {
            "experiment": "multilabel_shape_headroom",
            "seeds_completed": [int(s) for s in results_by_seed.keys()],
            "results_by_seed": results_by_seed,
        }
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"=== seed {seed} done ===", flush=True)

    payload = {
        "experiment": "multilabel_shape_headroom",
        "datasets": list(datasets),
        "observed_dims": list(observed_dims),
        "seeds": list(seeds),
        "wall_time_seconds": time.perf_counter() - start,
        "summary": summarize(results_by_seed),
        "results_by_seed": results_by_seed,
    }
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"wall_time_seconds": payload["wall_time_seconds"], "summary": payload["summary"]}, indent=2))
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
