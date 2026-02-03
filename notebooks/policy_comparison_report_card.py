# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: quant-aware-training
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Policy Comparison Report Card
#
# Compare two correction placement policies under equal budget:
# - Policy A: geometry-guided ranking
# - Policy B: evenly-spaced baseline

# %%
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from aleph.datasets import load_shakespeare
from aleph.models import TransformerWithCorrection
from aleph.quantization import calibrate_model
from aleph.qgeom import (
    collect_transformer_layer_reports,
    score_correction_points,
    build_baseline_policy,
    simulate_policy,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BITS = 4

CFG = {
    "seq_len": 128,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 256,
    "batch_size": 64,
    "epochs": 2,
    "lr": 3e-4,
    "corr_epochs": 2,
    "corr_lr": 1e-3,
    "corr_train_limit": 2048,
}


def quant_point_idx_to_key(idx):
    layer = idx // 2
    sub = "attn" if idx % 2 == 0 else "ffn"
    return f"{layer}_{sub}"


def train_backbone(cfg):
    torch.manual_seed(SEED)

    train_X, train_Y, test_X, test_Y, vocab_size, _, _ = load_shakespeare(seq_len=cfg["seq_len"])
    model = TransformerWithCorrection(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["seq_len"],
        correction_every_n=1,
        correction_hidden=0,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    n_batches = len(train_X) // cfg["batch_size"]

    model.train()
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(train_X))
        tX = train_X[perm]
        tY = train_Y[perm]
        for i in range(n_batches):
            x = tX[i * cfg["batch_size"]:(i + 1) * cfg["batch_size"]].to(DEVICE)
            y = tY[i * cfg["batch_size"]:(i + 1) * cfg["batch_size"]].to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    test_X = test_X.to(DEVICE)
    test_Y = test_Y.to(DEVICE)
    model.eval()

    with torch.no_grad():
        loss_float = F.cross_entropy(model(test_X).reshape(-1, vocab_size), test_Y.reshape(-1)).item()

    calib_x = train_X[: cfg["batch_size"]].to(DEVICE)
    scales, zps = calibrate_model(model, calib_x, num_bits=BITS)

    with torch.no_grad():
        loss_quant = F.cross_entropy(
            model.forward_quantized(test_X, scales, zps, num_bits=BITS).reshape(-1, vocab_size),
            test_Y.reshape(-1),
        ).item()

    return model, train_X, train_Y, test_X, test_Y, vocab_size, scales, zps, loss_float, loss_quant


def apply_policy_keys(model, selected_points):
    selected_keys = {quant_point_idx_to_key(i) for i in selected_points}
    kept = {k: v for k, v in model.correction_layers.items() if k in selected_keys}
    model.correction_layers = torch.nn.ModuleDict(kept)


def evaluate_policy_actual_gain(
    *,
    base_model,
    train_X,
    test_X,
    test_Y,
    scales,
    zps,
    vocab_size,
    loss_quant,
    selected_points,
):
    if len(selected_points) == 0:
        return {
            "loss_corrected": loss_quant,
            "actual_gain": 0.0,
            "recovery_pct": 0.0,
        }

    corr_model = copy.deepcopy(base_model)
    apply_policy_keys(corr_model, selected_points)

    # Freeze backbone.
    for name, p in corr_model.named_parameters():
        p.requires_grad = "correction_layers" in name

    params = [p for p in corr_model.parameters() if p.requires_grad]
    if not params:
        return {
            "loss_corrected": loss_quant,
            "actual_gain": 0.0,
            "recovery_pct": 0.0,
        }

    optimizer = torch.optim.Adam(params, lr=CFG["corr_lr"])
    n_train = min(len(train_X), CFG["corr_train_limit"])
    n_batches = n_train // CFG["batch_size"]

    for _ in range(CFG["corr_epochs"]):
        perm = torch.randperm(n_train)
        for i in range(n_batches):
            idx = perm[i * CFG["batch_size"]:(i + 1) * CFG["batch_size"]]
            x = train_X[idx].to(DEVICE)

            with torch.no_grad():
                teacher = base_model(x)

            student = corr_model.forward_quantized_with_correction(x, scales, zps, num_bits=BITS)
            loss = F.mse_loss(student, teacher)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    corr_model.eval()
    with torch.no_grad():
        loss_corrected = F.cross_entropy(
            corr_model.forward_quantized_with_correction(test_X, scales, zps, num_bits=BITS).reshape(-1, vocab_size),
            test_Y.reshape(-1),
        ).item()

    quant_gap = max(loss_quant - loss_corrected, 0.0)
    actual_gain = loss_quant - loss_corrected

    return {
        "loss_corrected": loss_corrected,
        "actual_gain": actual_gain,
        "recovery_pct": 100.0 * quant_gap / max(loss_quant, 1e-12),
    }


(
    base_model,
    train_X,
    train_Y,
    test_X,
    test_Y,
    vocab_size,
    scales,
    zps,
    loss_float,
    loss_quant,
) = train_backbone(CFG)

print(f"Float loss: {loss_float:.4f}")
print(f"Quant loss ({BITS}-bit): {loss_quant:.4f}")

run_report = collect_transformer_layer_reports(
    base_model,
    test_X[: CFG["batch_size"]],
    scales,
    zps,
    num_bits=BITS,
    task_name="policy_compare",
)

budget = max(1, len(run_report.layer_reports) // 2)
policy_a = score_correction_points(run_report, budget=budget)
policy_a["name"] = "geometry_guided"
policy_b = build_baseline_policy(run_report, budget=budget)
policy_b["name"] = "even_baseline"


def make_evaluator(base_model, train_X, test_X, test_Y, scales, zps, vocab_size, loss_quant):
    def _eval(*, model, data, quant_cfg, selected_points):
        return evaluate_policy_actual_gain(
            base_model=base_model,
            train_X=train_X,
            test_X=test_X,
            test_Y=test_Y,
            scales=scales,
            zps=zps,
            vocab_size=vocab_size,
            loss_quant=loss_quant,
            selected_points=selected_points,
        )

    return _eval


quant_cfg = {
    "policy_evaluator": make_evaluator(base_model, train_X, test_X, test_Y, scales, zps, vocab_size, loss_quant),
}

res_a = simulate_policy(policy_a, base_model, None, quant_cfg)
res_b = simulate_policy(policy_b, base_model, None, quant_cfg)

summary = pd.DataFrame(
    [
        {
            "policy": res_a.policy_name,
            "selected": res_a.selected_points,
            "predicted_gain": res_a.predicted_gain,
            "actual_gain": res_a.actual_gain,
            "loss_corrected": res_a.details.get("loss_corrected"),
        },
        {
            "policy": res_b.policy_name,
            "selected": res_b.selected_points,
            "predicted_gain": res_b.predicted_gain,
            "actual_gain": res_b.actual_gain,
            "loss_corrected": res_b.details.get("loss_corrected"),
        },
    ]
)

print("\nPolicy comparison")
print(summary)

# %%
rank_df = pd.DataFrame(policy_a["ranking"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(rank_df["layer_index"], rank_df["score"])
axes[0].set_title("Geometry-guided ranking scores")
axes[0].set_xlabel("quant point")
axes[0].set_ylabel("score")
axes[0].grid(True, alpha=0.3)

x = ["geometry_guided", "even_baseline"]
y = [res_a.actual_gain or 0.0, res_b.actual_gain or 0.0]
axes[1].bar(x, y, color=["#1f77b4", "#ff7f0e"])
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_title("Actual gain (loss_quant - loss_corrected)")
axes[1].set_ylabel("gain")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/policy_comparison_report_card.png", dpi=150, bbox_inches="tight")
plt.show()
