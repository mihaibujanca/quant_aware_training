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
# # Transformer Geometry Report Card (4-bit)
#
# High-dimensional anchor report using the same metric schema as low-dim diagnostics.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from aleph.datasets import load_shakespeare
from aleph.models import TransformerWithCorrection
from aleph.quantization import calibrate_model
from aleph.qgeom import collect_transformer_layer_reports, score_correction_points


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BITS = 4
SEED = 42

CFG = {
    "seq_len": 128,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 256,
    "batch_size": 64,
    "epochs": 2,
    "lr": 3e-4,
}


def train_small_transformer(cfg):
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

    model.eval()
    test_X = test_X.to(DEVICE)
    test_Y = test_Y.to(DEVICE)
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


(
    model,
    train_X,
    train_Y,
    test_X,
    test_Y,
    vocab_size,
    scales,
    zps,
    loss_float,
    loss_quant,
) = train_small_transformer(CFG)

print(f"Device: {DEVICE}")
print(f"Float loss: {loss_float:.4f}")
print(f"Quant loss ({BITS}-bit): {loss_quant:.4f}")

report_batch = test_X[: CFG["batch_size"]]
run_report = collect_transformer_layer_reports(
    model,
    report_batch,
    scales,
    zps,
    num_bits=BITS,
    task_name="transformer_shakespeare_report_card",
)

rows = []
for r in run_report.layer_reports:
    rows.append(
        {
            "layer": r.layer_index,
            "sub": r.metadata.get("sub_layer", "?"),
            "linear": r.linear_error_norm,
            "nonlinear": r.nonlinear_error_norm,
            "flip": r.relu_flip_rate,
            "dead": r.dead_rate,
            "survive": r.survive_rate,
            "sat": r.saturation_count,
            "anisotropy": r.anisotropy_ratio,
            "entropy": r.entropy_proxy,
            "correctability": r.correctability_score,
        }
    )

df = pd.DataFrame(rows)
print("\nAggregate:")
print(run_report.aggregate_metrics())
print("\nPer-layer metrics:")
print(df.round(4))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(df["layer"], df["correctability"], "o-", label="correctability")
axes[0].plot(df["layer"], df["flip"], "s--", label="flip")
axes[0].set_title("Correctability vs flip")
axes[0].set_xlabel("quant point")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].bar(df["layer"], df["linear"], label="linear")
axes[1].bar(df["layer"], df["nonlinear"], bottom=df["linear"], label="nonlinear")
axes[1].set_title("Linear/nonlinear error")
axes[1].set_xlabel("quant point")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(df["layer"], np.log1p(df["anisotropy"]), "o-")
axes[2].set_title("log(1+anisotropy)")
axes[2].set_xlabel("quant point")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/transformer_geometry_report_card.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
budget = max(1, len(run_report.layer_reports) // 2)
policy = score_correction_points(run_report, budget=budget)
print("Recommended correction points:", policy["selected_points"])
print(pd.DataFrame(policy["ranking"]).head(10).round(4))
