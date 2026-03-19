#!/usr/bin/env python3
"""Projected tiny-MLP + local RBF hybrid on rough spirals."""

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
    _make_classifier,
    _set_seed,
    _targets,
    _tensor,
    HighDReluCompressionConfig,
)
from experiments.rough_spirals_local_patch_hybrid import (
    _fit_classifier_best,
    _fit_tiny_relu_under_budget,
    _normalize,
    _projection_from_teacher,
    _subspace_alignment_score,
    _teacher_complexity,
)
from experiments.rough_spirals_local_rbf_hybrid import _bucket_metrics, _build_atom_specs


class LocalRBFTinyMLPHybrid(nn.Module):
    def __init__(
        self,
        *,
        centers: np.ndarray,
        scales: np.ndarray,
        global_hidden: int,
        global_depth: int,
    ):
        super().__init__()
        self.global_model = MLP(input_size=2, hidden_size=global_hidden, output_size=1, depth=global_depth)
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.local_weights = nn.Parameter(torch.zeros(len(centers), 3))

    @property
    def trainable_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    @property
    def stored_numbers(self) -> int:
        return int(self.trainable_params + self.centers.numel() + self.scales.numel())

    def local_features(self, z: torch.Tensor) -> torch.Tensor:
        if self.centers.numel() == 0:
            return torch.zeros(len(z), 0, device=z.device, dtype=z.dtype)
        dx = z[:, None, :] - self.centers[None, :, :]
        scales = self.scales[None, :, None].clamp_min(1e-4)
        local = dx / scales
        rbf = torch.exp(-0.5 * (local ** 2).sum(dim=-1))
        feat = torch.stack([rbf, rbf * local[..., 0], rbf * local[..., 1]], dim=-1)
        return feat.reshape(len(z), -1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.global_model(z)
        local = self.local_features(z)
        if local.shape[1] > 0:
            logits = logits + (local @ self.local_weights.reshape(-1)).unsqueeze(1)
        return logits


@torch.no_grad()
def _fit_local_residual_ridge(
    model: LocalRBFTinyMLPHybrid,
    z_train: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    ridge: float = 1e-2,
) -> None:
    local = model.local_features(z_train)
    if local.shape[1] == 0:
        return
    residual = (teacher_logits.squeeze(1) - model.global_model(z_train).squeeze(1)).detach()
    local_cpu = local.detach().cpu()
    residual_cpu = residual.detach().cpu().unsqueeze(1)
    gram = local_cpu.T @ local_cpu
    rhs = local_cpu.T @ residual_cpu
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    sol = torch.linalg.solve(gram + ridge * eye, rhs).squeeze(1).to(z_train.device)
    model.local_weights.copy_(sol.view_as(model.local_weights))


def _train_hybrid(
    model: LocalRBFTinyMLPHybrid,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
    teacher_logits: torch.Tensor,
    epochs: int,
    lr: float,
    sample_weights: torch.Tensor | None = None,
    distill_weight: float = 0.2,
    weight_decay: float = 5e-4,
) -> dict[str, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_bce = float("inf")
    best_state = None
    best_test = None
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(z_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        if sample_weights is not None:
            loss = loss * sample_weights
        loss = loss.mean() + distill_weight * F.mse_loss(logits, teacher_logits)
        loss.backward()
        opt.step()
        val = _eval_classifier(model, z_val, y_val)
        if val["bce"] < best_bce:
            best_bce = float(val["bce"])
            best_state = copy.deepcopy(model.state_dict())
            best_test = _eval_classifier(model, z_test, y_test)
    if best_state is not None:
        model.load_state_dict(best_state)
    val = _eval_classifier(model, z_val, y_val)
    test = best_test or _eval_classifier(model, z_test, y_test)
    return {
        "val_bce": float(val["bce"]),
        "val_accuracy": float(val["accuracy"]),
        "test_bce": float(test["bce"]),
        "test_accuracy": float(test["accuracy"]),
        "test_margin": float(test["mean_margin"]),
        "params": float(model.trainable_params),
        "stored_numbers": float(model.stored_numbers),
    }


def run_single(
    seed: int,
    observed_dim: int,
    *,
    projection_method: str,
    source_hidden: int,
    source_depth: int,
    source_epochs: int,
    source_lr: float,
    small_hidden: int,
    small_depth: int,
    small_epochs: int,
    small_lr: float,
    target_budget: int,
    atom_quantile: float,
    global_hidden: int,
    global_depth: int,
    fine_tune_epochs: int,
    fine_tune_lr: float,
) -> dict[str, object]:
    cfg = HighDReluCompressionConfig(
        dataset="rough_spirals",
        observed_dims=(observed_dim,),
        n_samples=2400,
        source_hidden=source_hidden,
        source_depth=source_depth,
        source_epochs=source_epochs,
        baseline_epochs=small_epochs,
        cleanup_epochs=fine_tune_epochs,
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
    X_train_high = _tensor(benchmark.train.X_high, device)
    y_train = _targets(benchmark.train.y, device)
    X_val_high = _tensor(benchmark.val.X_high, device)
    y_val = _targets(benchmark.val.y, device)
    X_test_high = _tensor(benchmark.test.X_high, device)
    y_test = _targets(benchmark.test.y, device)

    source = _make_classifier(observed_dim, source_hidden, source_depth).to(device)
    source_metrics = _fit_classifier_best(
        source,
        X_train_high,
        y_train,
        X_val=X_val_high,
        y_val=y_val,
        X_test=X_test_high,
        y_test=y_test,
        epochs=source_epochs,
        lr=source_lr,
    )

    ambient_small = _make_classifier(observed_dim, small_hidden, small_depth).to(device)
    ambient_small_metrics = _fit_classifier_best(
        ambient_small,
        X_train_high,
        y_train,
        X_val=X_val_high,
        y_val=y_val,
        X_test=X_test_high,
        y_test=y_test,
        epochs=small_epochs,
        lr=small_lr,
    )

    projection_P, projection_mean = _projection_from_teacher(
        source,
        benchmark.train.X_high,
        method=projection_method,
        device=device,
    )
    projection_alignment = _subspace_alignment_score(
        np.asarray(benchmark.embedding.W, dtype=np.float32),
        projection_P,
    )
    z_train_np, complexity = _teacher_complexity(
        source,
        benchmark.train.X_high,
        projection_P=projection_P,
        projection_mean=projection_mean,
        device=device,
    )
    z_test_np, test_complexity = _teacher_complexity(
        source,
        benchmark.test.X_high,
        projection_P=projection_P,
        projection_mean=projection_mean,
        device=device,
    )
    z_val_np = ((benchmark.val.X_high - projection_mean) @ projection_P).astype(np.float32)

    z_train = _tensor(z_train_np, device)
    z_val = _tensor(z_val_np, device)
    z_test = _tensor(z_test_np, device)
    teacher_logits = source(X_train_high).detach()
    weights = torch.tensor(1.0 + 0.6 * _normalize(complexity), dtype=torch.float32, device=device).unsqueeze(1)
    smooth_weights = torch.tensor(1.0 - 0.35 * _normalize(complexity), dtype=torch.float32, device=device).clamp_min_(0.3).unsqueeze(1)

    global_budget = sum(p.numel() for p in MLP(input_size=2, hidden_size=global_hidden, output_size=1, depth=global_depth).parameters())
    remaining = max(target_budget - global_budget, 3)
    n_atoms = max(1, min(160, remaining // 3))
    centers, scales = _build_atom_specs(
        z_train_np,
        complexity,
        n_atoms=n_atoms,
        quantile=atom_quantile,
        seed=seed,
    )
    model = LocalRBFTinyMLPHybrid(
        centers=centers,
        scales=scales,
        global_hidden=global_hidden,
        global_depth=global_depth,
    ).to(device)

    global_pretrain = _fit_classifier_best(
        model.global_model,
        z_train,
        y_train,
        X_val=z_val,
        y_val=y_val,
        X_test=z_test,
        y_test=y_test,
        epochs=max(250, fine_tune_epochs // 2),
        lr=2e-3,
        sample_weights=smooth_weights,
    )
    _fit_local_residual_ridge(model, z_train, teacher_logits)
    hybrid_metrics = _train_hybrid(
        model,
        z_train,
        y_train,
        z_val=z_val,
        y_val=y_val,
        z_test=z_test,
        y_test=y_test,
        teacher_logits=teacher_logits,
        epochs=fine_tune_epochs,
        lr=fine_tune_lr,
        sample_weights=weights,
        distill_weight=0.20,
        weight_decay=5e-4,
    )

    projected_tiny = _fit_tiny_relu_under_budget(2, int(model.trainable_params)).to(device)
    projected_tiny_metrics = _fit_classifier_best(
        projected_tiny,
        z_train,
        y_train,
        X_val=z_val,
        y_val=y_val,
        X_test=z_test,
        y_test=y_test,
        epochs=fine_tune_epochs,
        lr=min(fine_tune_lr, 3e-3),
        teacher_logits=teacher_logits,
        distill_weight=0.15,
        sample_weights=weights,
    )

    with torch.no_grad():
        source_logits_test = source(X_test_high).squeeze(1).detach().cpu().numpy().astype(np.float32)
        ambient_logits_test = ambient_small(X_test_high).squeeze(1).detach().cpu().numpy().astype(np.float32)
        hybrid_logits_test = model(z_test).squeeze(1).detach().cpu().numpy().astype(np.float32)
        tiny_logits_test = projected_tiny(z_test).squeeze(1).detach().cpu().numpy().astype(np.float32)
    bucket_y = benchmark.test.y.astype(np.float32)

    return {
        "source": source_metrics,
        "ambient_small": ambient_small_metrics,
        "projected_tiny_relu": projected_tiny_metrics,
        "projected_global_pretrain": global_pretrain,
        "hybrid_local_rbf_mlp": {
            **hybrid_metrics,
            "n_atoms": float(len(centers)),
            "target_budget": float(target_budget),
        },
        "projection_method": projection_method,
        "projection_alignment": float(projection_alignment),
        "complexity_summary": {
            "mean": float(np.mean(complexity)),
            "max": float(np.max(complexity)),
            "q80": float(np.quantile(complexity, 0.8)),
            "q95": float(np.quantile(complexity, 0.95)),
        },
        "teacher_complexity_buckets": {
            "source": _bucket_metrics(source_logits_test, test_complexity, bucket_y),
            "ambient_small": _bucket_metrics(ambient_logits_test, test_complexity, bucket_y),
            "hybrid_local_rbf_mlp": _bucket_metrics(hybrid_logits_test, test_complexity, bucket_y),
            "projected_tiny_relu": _bucket_metrics(tiny_logits_test, test_complexity, bucket_y),
        },
    }


def _summarize(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]["test_accuracy"]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Projected tiny-MLP + local RBF hybrid on rough spirals")
    parser.add_argument("--observed-dims", type=str, default="100,1000")
    parser.add_argument("--projection-methods", type=str, default="manifold_pca")
    parser.add_argument("--source-hidden", type=int, default=96)
    parser.add_argument("--source-depth", type=int, default=6)
    parser.add_argument("--source-epochs", type=int, default=1000)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--small-hidden", type=int, default=32)
    parser.add_argument("--small-depth", type=int, default=4)
    parser.add_argument("--small-epochs", type=int, default=1000)
    parser.add_argument("--small-lr", type=float, default=5e-4)
    parser.add_argument("--target-budget", type=int, default=384)
    parser.add_argument("--atom-quantile", type=float, default=0.75)
    parser.add_argument("--global-hidden", type=int, default=8)
    parser.add_argument("--global-depth", type=int, default=1)
    parser.add_argument("--fine-tune-epochs", type=int, default=1000)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dims = [int(part) for part in args.observed_dims.split(",") if part]
    methods = [part.strip() for part in args.projection_methods.split(",") if part.strip()]
    seeds = [int(part) for part in args.seeds.split(",") if part]
    started = time.perf_counter()
    out: dict[str, object] = {
        "experiment": "rough_spirals_local_rbf_mlp_hybrid",
        "methods": methods,
        "results": {},
    }

    for method in methods:
        method_rows: dict[str, list[dict[str, object]]] = {str(dim): [] for dim in dims}
        for dim in dims:
            for seed in seeds:
                print(f"=== method {method} dim {dim} seed {seed} start ===", flush=True)
                row = run_single(
                    seed,
                    dim,
                    projection_method=method,
                    source_hidden=args.source_hidden,
                    source_depth=args.source_depth,
                    source_epochs=args.source_epochs,
                    source_lr=args.source_lr,
                    small_hidden=args.small_hidden,
                    small_depth=args.small_depth,
                    small_epochs=args.small_epochs,
                    small_lr=args.small_lr,
                    target_budget=args.target_budget,
                    atom_quantile=args.atom_quantile,
                    global_hidden=args.global_hidden,
                    global_depth=args.global_depth,
                    fine_tune_epochs=args.fine_tune_epochs,
                    fine_tune_lr=args.fine_tune_lr,
                )
                method_rows[str(dim)].append(row)
                print(f"=== method {method} dim {dim} seed {seed} done ===", flush=True)
        out["results"][method] = {}
        for dim, rows in method_rows.items():
            out["results"][method][dim] = {
                "rows": rows,
                "mean_source_acc": _summarize(rows, "source"),
                "mean_ambient_small_acc": _summarize(rows, "ambient_small"),
                "mean_projected_tiny_acc": _summarize(rows, "projected_tiny_relu"),
                "mean_hybrid_acc": _summarize(rows, "hybrid_local_rbf_mlp"),
                "mean_hybrid_vs_small": float(
                    np.mean(
                        [
                            float(row["hybrid_local_rbf_mlp"]["test_accuracy"])
                            - float(row["ambient_small"]["test_accuracy"])
                            for row in rows
                        ]
                    )
                ),
                "mean_hybrid_vs_tiny": float(
                    np.mean(
                        [
                            float(row["hybrid_local_rbf_mlp"]["test_accuracy"])
                            - float(row["projected_tiny_relu"]["test_accuracy"])
                            for row in rows
                        ]
                    )
                ),
                "mean_projection_alignment": float(
                    np.mean([float(row["projection_alignment"]) for row in rows])
                ),
                "mean_hybrid_params": float(
                    np.mean([float(row["hybrid_local_rbf_mlp"]["params"]) for row in rows])
                ),
                "mean_hybrid_stored_numbers": float(
                    np.mean([float(row["hybrid_local_rbf_mlp"]["stored_numbers"]) for row in rows])
                ),
                "mean_n_atoms": float(
                    np.mean([float(row["hybrid_local_rbf_mlp"]["n_atoms"]) for row in rows])
                ),
            }

    out["wall_time_seconds"] = float(time.perf_counter() - started)
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
