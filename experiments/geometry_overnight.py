"""Overnight geometry-guided quantization experiments (Transformer 4-bit anchor).

This runner executes two experiment families:

Q1) Linear recoverability before nonlinear collapse
Q2) Generic correction from quantization-time statistics

It writes machine-readable outputs plus a human-readable assessment memo with
hypotheses, thresholds, and verdicts.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from aleph.datasets import load_shakespeare
from aleph.models import TransformerWithCorrection
from aleph.qgeom import (
    collect_transformer_layer_reports,
    collect_transformer_traces,
    quantize_decompose_tensor,
)
from aleph.quantization import calibrate_model

LOG = logging.getLogger("geometry_overnight")


@dataclass
class BackboneArtifacts:
    model: TransformerWithCorrection
    train_X: torch.Tensor
    train_Y: torch.Tensor
    test_X: torch.Tensor
    test_Y: torch.Tensor
    vocab_size: int
    scales: list[float]
    zero_points: list[float]
    loss_float: float
    loss_quant: float


class ElementCorrector(nn.Module):
    """Generic per-element correction model over quantization-time features."""

    def __init__(
        self,
        n_layers: int,
        *,
        use_features: bool,
        use_embedding: bool,
        feature_dim: int = 5,
        emb_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.use_features = use_features
        self.use_embedding = use_embedding

        in_dim = 0
        if use_features:
            in_dim += feature_dim
        if use_embedding:
            self.layer_emb = nn.Embedding(n_layers, emb_dim)
            in_dim += emb_dim

        if in_dim == 0:
            raise ValueError("At least one of use_features/use_embedding must be True")

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, layer_idx: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.use_features:
            parts.append(features)
        if self.use_embedding:
            parts.append(self.layer_emb(layer_idx))
        x = torch.cat(parts, dim=-1)
        return self.net(x).squeeze(-1)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def quantize_tensor_np(x: np.ndarray, bits: int) -> np.ndarray:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    abs_max = float(np.max(np.abs(x)))
    if abs_max <= 1e-12:
        return x.copy()
    scale = abs_max / qmax
    q = np.round(x / scale)
    q = np.clip(q, qmin, qmax)
    return q * scale


def fit_ridge_linear_map(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Fit Y ~ XW + b in closed form (ridge, no bias regularization)."""
    n, d = X.shape
    X_aug = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
    XtX = X_aug.T @ X_aug
    reg = np.eye(d + 1, dtype=X.dtype) * ridge
    reg[-1, -1] = 0.0
    XtY = X_aug.T @ Y
    W_aug = np.linalg.solve(XtX + reg, XtY)
    W = W_aug[:-1]
    b = W_aug[-1]
    return W, b


def layer_idx_to_key(layer_idx: int) -> str:
    block = layer_idx // 2
    sub = "attn" if layer_idx % 2 == 0 else "ffn"
    return f"{block}_{sub}"


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"

    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    LOG.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


def write_hypothesis_card(output_dir: Path, args: argparse.Namespace) -> None:
    card = f"""# Overnight Experiment Card

## Q1 — Linear Recoverability Before Collapse

**Hypothesis Q1**
A substantial fraction of pre-activation quantization error is linearly recoverable
(at 4-bit correction precision) before nonlinear collapse dominates.

**Assessment policy**
- Compute per-layer `R_lin_q{args.bits}` and `R_lin_float`.
- Compute Spearman correlation between layer `correctability_score` and `R_lin_q{args.bits}`.
- Verdict thresholds:
  - **Supported**: median(`R_lin_q{args.bits}`) >= {args.q1_recovery_threshold:.2f} AND Spearman >= {args.q1_spearman_threshold:.2f}
  - **Refuted**: median(`R_lin_q{args.bits}`) < {args.q1_recovery_threshold * 0.7:.2f} AND Spearman < 0
  - **Unclear**: otherwise

## Q2 — Generic Corrector From Quantization-Time Statistics

**Hypothesis Q2**
Quantization-time features carry transferable correction signal beyond layer identity,
and features+layer context is better than embedding-only.

**Assessment policy**
Compare three variants:
1. `embedding_only`
2. `features_only`
3. `features_plus_embedding`

Metrics:
- held-out elementwise correction MSE
- corrected downstream test loss

Verdict checks:
- **Supported** if
  - `features_only` MSE improves over `embedding_only` by >= {args.q2_feature_mse_gain*100:.1f}% AND
  - best(features models) improves quantized downstream loss by >= {args.q2_loss_gain:.4f}
- **Refuted** if `features_only` is worse than `embedding_only` and no variant improves downstream loss.
- **Unclear** otherwise.

## Where to look tomorrow
- `q1_layer_metrics.csv`
- `q2_variant_metrics.csv`
- `assessment_summary.json`
- `assessment_summary.md`
"""
    (output_dir / "hypothesis_card.md").write_text(card)


def train_backbone(
    *,
    seed: int,
    d_model: int,
    n_layers: int,
    d_ff: int,
    seq_len: int,
    batch_size: int,
    backbone_epochs: int,
    lr: float,
    bits: int,
    device: torch.device,
    train_limit: int | None,
    test_limit: int | None,
) -> BackboneArtifacts:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_X, train_Y, test_X, test_Y, vocab_size, _, _ = load_shakespeare(seq_len=seq_len)

    if train_limit is not None:
        train_X = train_X[:train_limit]
        train_Y = train_Y[:train_limit]
    if test_limit is not None:
        test_X = test_X[:test_limit]
        test_Y = test_Y[:test_limit]

    model = TransformerWithCorrection(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_len,
        dropout=0.1,
        correction_every_n=1,
        correction_hidden=0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_batches = max(len(train_X) // batch_size, 1)

    model.train()
    for _ in range(backbone_epochs):
        perm = torch.randperm(len(train_X))
        tX = train_X[perm]
        tY = train_Y[perm]

        for i in range(n_batches):
            x = tX[i * batch_size:(i + 1) * batch_size].to(device)
            y = tY[i * batch_size:(i + 1) * batch_size].to(device)
            if x.numel() == 0:
                continue
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)

    with torch.no_grad():
        loss_float = F.cross_entropy(model(test_X).reshape(-1, vocab_size), test_Y.reshape(-1)).item()

    calib_x = train_X[:batch_size].to(device)
    scales, zps = calibrate_model(model, calib_x, num_bits=bits)

    with torch.no_grad():
        quant_logits = model.forward_quantized(test_X, scales, zps, num_bits=bits)
        loss_quant = F.cross_entropy(quant_logits.reshape(-1, vocab_size), test_Y.reshape(-1)).item()

    return BackboneArtifacts(
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        test_Y=test_Y,
        vocab_size=vocab_size,
        scales=scales,
        zero_points=zps,
        loss_float=loss_float,
        loss_quant=loss_quant,
    )


def run_q1_linear_recoverability(
    *,
    traces_train: list[dict[str, np.ndarray]],
    traces_test: list[dict[str, np.ndarray]],
    run_report,
    bits: int,
    ridge: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    report_map = {int(r.layer_index): r for r in run_report.layer_reports}

    rows = []
    for tr_train, tr_test in zip(traces_train, traces_test):
        layer_idx = int(tr_train["layer_index"])

        X_train = tr_train["z_quant"]
        Y_train = tr_train["z_float"]
        X_test = tr_test["z_quant"]
        Y_test = tr_test["z_float"]

        W, b = fit_ridge_linear_map(X_train, Y_train, ridge=ridge)

        base_mse = float(np.mean((X_test - Y_test) ** 2))

        pred_float = X_test @ W + b
        mse_float = float(np.mean((pred_float - Y_test) ** 2))

        W_q = quantize_tensor_np(W, bits)
        b_q = quantize_tensor_np(b, bits)
        pred_q = X_test @ W_q + b_q
        mse_q = float(np.mean((pred_q - Y_test) ** 2))

        r_float = 1.0 - (mse_float / max(base_mse, 1e-12))
        r_q = 1.0 - (mse_q / max(base_mse, 1e-12))

        layer_report = report_map[layer_idx]

        rows.append(
            {
                "layer_index": layer_idx,
                "sub_layer": layer_report.metadata.get("sub_layer", "?"),
                "base_mse": base_mse,
                "lin_float_mse": mse_float,
                f"lin_q{bits}_mse": mse_q,
                "R_lin_float": r_float,
                f"R_lin_q{bits}": r_q,
                "correctability_score": layer_report.correctability_score,
                "relu_flip_rate": layer_report.relu_flip_rate,
                "dead_rate": layer_report.dead_rate,
                "anisotropy_ratio": layer_report.anisotropy_ratio,
                "entropy_proxy": layer_report.entropy_proxy,
            }
        )

    df = pd.DataFrame(rows).sort_values("layer_index").reset_index(drop=True)

    spearman = spearmanr(df["correctability_score"], df[f"R_lin_q{bits}"]).correlation
    summary = {
        "median_R_lin_float": float(df["R_lin_float"].median()),
        f"median_R_lin_q{bits}": float(df[f"R_lin_q{bits}"].median()),
        f"spearman_correctability_vs_R_lin_q{bits}": float(0.0 if np.isnan(spearman) else spearman),
    }
    return df, summary


def build_element_dataset(
    traces: list[dict[str, np.ndarray]],
    *,
    max_elements_total: int,
    rng_seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    n_layers = len(traces)
    per_layer = max(max_elements_total // max(n_layers, 1), 1)

    feats_all = []
    targets_all = []
    layer_ids_all = []

    for tr in traces:
        zq = tr["z_quant"].reshape(-1)
        residual = tr["quant_residual"].reshape(-1)
        sat = tr["sat_mask"].reshape(-1)
        target = tr["true_error"].reshape(-1)
        scale = float(tr["scale"])
        layer_idx = int(tr["layer_index"])

        n = zq.shape[0]
        take = min(per_layer, n)
        idx = rng.choice(n, size=take, replace=False)

        zq_s = zq[idx]
        residual_s = residual[idx]
        sat_s = sat[idx]
        target_s = target[idx]

        features = np.column_stack(
            [
                zq_s,
                residual_s,
                sat_s,
                np.full_like(zq_s, scale),
                np.abs(zq_s),
            ]
        ).astype(np.float32)

        feats_all.append(features)
        targets_all.append(target_s.astype(np.float32))
        layer_ids_all.append(np.full(take, layer_idx, dtype=np.int64))

    X = np.concatenate(feats_all, axis=0)
    y = np.concatenate(targets_all, axis=0)
    layer_ids = np.concatenate(layer_ids_all, axis=0)

    return {"features": X, "targets": y, "layer_ids": layer_ids}


def train_element_corrector(
    *,
    dataset: dict[str, np.ndarray],
    n_layers: int,
    variant: str,
    use_features: bool,
    use_embedding: bool,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    rng_seed: int,
) -> tuple[ElementCorrector, dict[str, float]]:
    X = dataset["features"]
    y = dataset["targets"]
    layer_ids = dataset["layer_ids"]

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=rng_seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=rng_seed)

    model = ElementCorrector(
        n_layers=n_layers,
        use_features=use_features,
        use_embedding=use_embedding,
        feature_dim=X.shape[1],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def to_tensors(indices: np.ndarray):
        feats_t = torch.tensor(X[indices], dtype=torch.float32, device=device)
        layer_t = torch.tensor(layer_ids[indices], dtype=torch.long, device=device)
        target_t = torch.tensor(y[indices], dtype=torch.float32, device=device)
        return feats_t, layer_t, target_t

    tr_feats, tr_layer, tr_target = to_tensors(train_idx)
    va_feats, va_layer, va_target = to_tensors(val_idx)
    te_feats, te_layer, te_target = to_tensors(test_idx)

    best_val = math.inf
    best_state = None

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(tr_target.shape[0], device=device)

        for i in range(0, tr_target.shape[0], batch_size):
            bi = perm[i : i + batch_size]
            pred = model(tr_feats[bi], tr_layer[bi])
            loss = F.mse_loss(pred, tr_target[bi])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(va_feats, va_layer)
            val_mse = float(F.mse_loss(val_pred, va_target).item())
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        tr_mse = float(F.mse_loss(model(tr_feats, tr_layer), tr_target).item())
        va_mse = float(F.mse_loss(model(va_feats, va_layer), va_target).item())
        te_mse = float(F.mse_loss(model(te_feats, te_layer), te_target).item())

    return model, {
        "variant": variant,
        "train_mse": tr_mse,
        "val_mse": va_mse,
        "test_mse": te_mse,
        "n_train": int(train_idx.shape[0]),
        "n_val": int(val_idx.shape[0]),
        "n_test": int(test_idx.shape[0]),
    }


def forward_with_generic_corrector(
    model: TransformerWithCorrection,
    x: torch.Tensor,
    scales: list[float],
    zps: list[float],
    *,
    corrector: ElementCorrector,
    bits: int,
) -> torch.Tensor:
    """Quantized forward with generic elementwise correction model."""
    B, T = x.shape
    pos = torch.arange(T, device=x.device).unsqueeze(0)

    xq = model.dropout(model.embedding(x) + model.pos_embedding(pos))
    sf_idx = 0

    for layer_idx, layer in enumerate(model.layers):
        # attn point
        x_ln = layer.ln1(xq)
        attn_out = layer.attn(x_ln)
        pre_q = xq + attn_out
        post_q, _, _, sat = quantize_decompose_tensor(pre_q, scales[sf_idx], zps[sf_idx], bits)
        resid = post_q - pre_q
        scale_t = torch.full_like(post_q, float(scales[sf_idx]))
        features = torch.stack(
            [post_q, resid, torch.tensor(sat, dtype=post_q.dtype, device=post_q.device), scale_t, torch.abs(post_q)],
            dim=-1,
        )
        flat_f = features.reshape(-1, features.shape[-1])
        flat_layer = torch.full((flat_f.shape[0],), 2 * layer_idx, dtype=torch.long, device=post_q.device)
        corr = corrector(flat_f, flat_layer).reshape_as(post_q)
        xq = post_q + corr
        sf_idx += 1

        # ffn point
        x_ln = layer.ln2(xq)
        ffn_out = layer.ffn(x_ln)
        pre_q = xq + ffn_out
        post_q, _, _, sat = quantize_decompose_tensor(pre_q, scales[sf_idx], zps[sf_idx], bits)
        resid = post_q - pre_q
        scale_t = torch.full_like(post_q, float(scales[sf_idx]))
        features = torch.stack(
            [post_q, resid, torch.tensor(sat, dtype=post_q.dtype, device=post_q.device), scale_t, torch.abs(post_q)],
            dim=-1,
        )
        flat_f = features.reshape(-1, features.shape[-1])
        flat_layer = torch.full((flat_f.shape[0],), 2 * layer_idx + 1, dtype=torch.long, device=post_q.device)
        corr = corrector(flat_f, flat_layer).reshape_as(post_q)
        xq = post_q + corr
        sf_idx += 1

    xq = model.ln_f(xq)
    return model.head(xq)


def run_q2_generic_corrector(
    *,
    backbone: BackboneArtifacts,
    traces_train: list[dict[str, np.ndarray]],
    bits: int,
    device: torch.device,
    max_elements_total: int,
    corr_epochs: int,
    corr_batch_size: int,
    corr_lr: float,
    rng_seed: int,
) -> pd.DataFrame:
    dataset = build_element_dataset(
        traces_train,
        max_elements_total=max_elements_total,
        rng_seed=rng_seed,
    )
    n_layers = len(traces_train)

    variants = [
        ("embedding_only", False, True),
        ("features_only", True, False),
        ("features_plus_embedding", True, True),
    ]

    rows: list[dict[str, Any]] = []
    for name, use_features, use_embedding in variants:
        LOG.info("Training generic corrector variant=%s", name)
        corr_model, fit_metrics = train_element_corrector(
            dataset=dataset,
            n_layers=n_layers,
            variant=name,
            use_features=use_features,
            use_embedding=use_embedding,
            device=device,
            epochs=corr_epochs,
            batch_size=corr_batch_size,
            lr=corr_lr,
            rng_seed=rng_seed,
        )

        corr_model.eval()
        with torch.no_grad():
            logits_corr = forward_with_generic_corrector(
                backbone.model,
                backbone.test_X,
                backbone.scales,
                backbone.zero_points,
                corrector=corr_model,
                bits=bits,
            )
            loss_corr = float(
                F.cross_entropy(logits_corr.reshape(-1, backbone.vocab_size), backbone.test_Y.reshape(-1)).item()
            )

        rows.append(
            {
                **fit_metrics,
                "loss_quant": backbone.loss_quant,
                "loss_corrected": loss_corr,
                "loss_gain": backbone.loss_quant - loss_corr,
                "ppl_quant": float(np.exp(backbone.loss_quant)),
                "ppl_corrected": float(np.exp(loss_corr)),
            }
        )

    return pd.DataFrame(rows)


def verdict_q1(summary: dict[str, float], args: argparse.Namespace) -> str:
    median_r = summary[f"median_R_lin_q{args.bits}"]
    spearman = summary[f"spearman_correctability_vs_R_lin_q{args.bits}"]

    if median_r >= args.q1_recovery_threshold and spearman >= args.q1_spearman_threshold:
        return "supported"
    if median_r < args.q1_recovery_threshold * 0.7 and spearman < 0:
        return "refuted"
    return "unclear"


def verdict_q2(df_q2: pd.DataFrame, args: argparse.Namespace) -> str:
    by_name = {row["variant"]: row for _, row in df_q2.iterrows()}
    emb = by_name["embedding_only"]
    feat = by_name["features_only"]
    both = by_name["features_plus_embedding"]

    mse_gain = (emb["test_mse"] - feat["test_mse"]) / max(emb["test_mse"], 1e-12)
    best_loss_gain = max(feat["loss_gain"], both["loss_gain"])

    if mse_gain >= args.q2_feature_mse_gain and best_loss_gain >= args.q2_loss_gain:
        return "supported"
    if feat["test_mse"] > emb["test_mse"] and best_loss_gain <= 0:
        return "refuted"
    return "unclear"


def write_assessment(
    *,
    output_dir: Path,
    q1_summary: dict[str, float],
    q2_df: pd.DataFrame,
    q1_verdict: str,
    q2_verdict: str,
    args: argparse.Namespace,
) -> None:
    summary_json = {
        "q1": {"summary": q1_summary, "verdict": q1_verdict},
        "q2": {
            "table": q2_df.to_dict(orient="records"),
            "verdict": q2_verdict,
        },
        "what_to_check": [
            "q1_layer_metrics.csv",
            "q2_variant_metrics.csv",
            "assessment_summary.md",
        ],
    }
    (output_dir / "assessment_summary.json").write_text(json.dumps(summary_json, indent=2))

    lines = [
        "# Assessment Summary",
        "",
        "## Q1 — Linear Recoverability Before Collapse",
        f"- Verdict: **{q1_verdict}**",
        f"- median(R_lin_q{args.bits}) = {q1_summary[f'median_R_lin_q{args.bits}']:.4f}",
        f"- median(R_lin_float) = {q1_summary['median_R_lin_float']:.4f}",
        f"- Spearman(correctability, R_lin_q{args.bits}) = {q1_summary[f'spearman_correctability_vs_R_lin_q{args.bits}']:.4f}",
        "",
        "## Q2 — Generic Corrector From Quantization-Time Statistics",
        f"- Verdict: **{q2_verdict}**",
        "- Table: `q2_variant_metrics.csv`",
        "",
        "## Morning Checklist",
        "1. Open `q1_layer_metrics.csv`: confirm where linear recoverability is highest and whether it aligns with correctability score.",
        "2. Open `q2_variant_metrics.csv`: compare `features_only` and `features_plus_embedding` against `embedding_only` on both `test_mse` and `loss_gain`.",
        "3. Check verdicts in `assessment_summary.json` and decide next sweep settings.",
    ]
    (output_dir / "assessment_summary.md").write_text("\n".join(lines))


def run_one_setting(args: argparse.Namespace, output_dir: Path, *, seed: int, d_model: int, n_layers: int) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setting_dir = output_dir / f"seed{seed}_d{d_model}_L{n_layers}"
    setting_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Running setting seed=%s d_model=%s n_layers=%s", seed, d_model, n_layers)

    backbone = train_backbone(
        seed=seed,
        d_model=d_model,
        n_layers=n_layers,
        d_ff=4 * d_model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        backbone_epochs=args.backbone_epochs,
        lr=args.lr,
        bits=args.bits,
        device=device,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )

    LOG.info("Backbone losses: float=%.4f quant=%.4f", backbone.loss_float, backbone.loss_quant)

    report_batch = backbone.test_X[: min(backbone.test_X.shape[0], args.report_batch_size)]
    run_report = collect_transformer_layer_reports(
        backbone.model,
        report_batch,
        backbone.scales,
        backbone.zero_points,
        num_bits=args.bits,
        task_name="transformer_shakespeare_geometry_overnight",
    )

    run_report_json = setting_dir / "run_geometry_report.json"
    run_report_json.write_text(json.dumps(run_report.to_dict(), indent=2))

    train_trace_batch = backbone.train_X[: min(backbone.train_X.shape[0], args.trace_train_batch)].to(device)
    test_trace_batch = backbone.test_X[: min(backbone.test_X.shape[0], args.trace_test_batch)]

    traces_train = collect_transformer_traces(
        backbone.model,
        train_trace_batch,
        backbone.scales,
        backbone.zero_points,
        num_bits=args.bits,
        max_points_per_layer=args.max_trace_points,
        rng_seed=seed,
    )
    traces_test = collect_transformer_traces(
        backbone.model,
        test_trace_batch,
        backbone.scales,
        backbone.zero_points,
        num_bits=args.bits,
        max_points_per_layer=args.max_trace_points,
        rng_seed=seed + 1,
    )

    df_q1, q1_summary = run_q1_linear_recoverability(
        traces_train=traces_train,
        traces_test=traces_test,
        run_report=run_report,
        bits=args.bits,
        ridge=args.q1_ridge,
    )
    df_q1.to_csv(setting_dir / "q1_layer_metrics.csv", index=False)

    df_q2 = run_q2_generic_corrector(
        backbone=backbone,
        traces_train=traces_train,
        bits=args.bits,
        device=device,
        max_elements_total=args.q2_max_elements,
        corr_epochs=args.corr_epochs,
        corr_batch_size=args.corr_batch_size,
        corr_lr=args.corr_lr,
        rng_seed=seed,
    )
    df_q2.to_csv(setting_dir / "q2_variant_metrics.csv", index=False)

    q1_v = verdict_q1(q1_summary, args)
    q2_v = verdict_q2(df_q2, args)

    write_assessment(
        output_dir=setting_dir,
        q1_summary=q1_summary,
        q2_df=df_q2,
        q1_verdict=q1_v,
        q2_verdict=q2_v,
        args=args,
    )

    return {
        "seed": seed,
        "d_model": d_model,
        "n_layers": n_layers,
        "loss_float": backbone.loss_float,
        "loss_quant": backbone.loss_quant,
        **q1_summary,
        "q1_verdict": q1_v,
        "q2_verdict": q2_v,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run overnight geometry-guided experiments")

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--d-models", type=str, default="64,128")
    parser.add_argument("--n-layers", type=str, default="2,4")

    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--backbone-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--test-limit", type=int, default=4000)
    parser.add_argument("--report-batch-size", type=int, default=64)
    parser.add_argument("--trace-train-batch", type=int, default=512)
    parser.add_argument("--trace-test-batch", type=int, default=256)
    parser.add_argument("--max-trace-points", type=int, default=20000)

    parser.add_argument("--q1-ridge", type=float, default=1e-4)
    parser.add_argument("--q1-recovery-threshold", type=float, default=0.35)
    parser.add_argument("--q1-spearman-threshold", type=float, default=0.35)

    parser.add_argument("--corr-epochs", type=int, default=20)
    parser.add_argument("--corr-batch-size", type=int, default=4096)
    parser.add_argument("--corr-lr", type=float, default=1e-3)
    parser.add_argument("--q2-max-elements", type=int, default=600000)
    parser.add_argument("--q2-feature-mse-gain", type=float, default=0.15)
    parser.add_argument("--q2-loss-gain", type=float, default=0.01)

    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    args.seeds = "42"
    args.d_models = "64"
    args.n_layers = "2"
    args.backbone_epochs = 1
    args.train_limit = 4096
    args.test_limit = 1024
    args.trace_train_batch = 128
    args.trace_test_batch = 64
    args.max_trace_points = 4096
    args.corr_epochs = 2
    args.q2_max_elements = 50000


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke_defaults(args)

    seeds = parse_int_list(args.seeds)
    d_models = parse_int_list(args.d_models)
    n_layers_list = parse_int_list(args.n_layers)

    if len(d_models) != len(n_layers_list):
        raise ValueError("--d-models and --n-layers must have the same number of entries")

    if args.output_dir is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/geometry_overnight_{ts}")
    else:
        output_dir = Path(args.output_dir)

    setup_logging(output_dir)
    write_hypothesis_card(output_dir, args)

    all_rows = []
    for seed in seeds:
        for d_model, n_layers in zip(d_models, n_layers_list):
            row = run_one_setting(args, output_dir, seed=seed, d_model=d_model, n_layers=n_layers)
            all_rows.append(row)

    master = pd.DataFrame(all_rows)
    master.to_csv(output_dir / "master_summary.csv", index=False)

    LOG.info("Completed all settings. Summary saved to %s", output_dir / "master_summary.csv")


if __name__ == "__main__":
    main()
