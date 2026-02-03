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
# # Low-Dim Geometry Report Card
#
# Notebook-first report card for geometry-guided correction diagnostics.
#
# Scenarios:
# 1. Linear-only stack (high correctability)
# 2. ReLU boundary stack (flip-heavy)
# 3. Saturation-heavy stack (collapse-heavy)
#
# Each scenario emits a `RunGeometryReport` with shared per-layer metrics.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from aleph.qgeom import (
    RunGeometryReport,
    build_layer_geometry_report,
    score_correction_points,
    build_baseline_policy,
)


BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))
RNG = np.random.default_rng(42)


def fake_quantize_decompose(x, delta=DELTA, qmin=-8, qmax=7):
    x_scaled = x / delta
    rounded = np.round(x_scaled)
    clipped = np.clip(rounded, qmin, qmax)
    dequant = clipped * delta

    round_err = (rounded - x_scaled) * delta
    sat_err = (clipped - rounded) * delta
    sat_mask = rounded != clipped

    return dequant, round_err, sat_err, sat_mask


def simulate_stack(name, X0, weights, biases, *, use_relu=True, force_clip=0.0):
    """Run float/quantized parallel stack and build RunGeometryReport."""
    x_float = X0.copy()
    x_quant = X0.copy()

    reports = []
    prev_total_error = None

    for layer_idx, (W, b) in enumerate(zip(weights, biases)):
        W_q = np.round(W / DELTA) * DELTA
        b_q = np.round(b / DELTA) * DELTA

        pre_float = x_float @ W.T + b
        pre_quant = x_quant @ W_q.T + b_q

        if force_clip > 0:
            pre_quant = pre_quant + force_clip * np.sign(pre_quant)

        quantized, round_err, sat_err, sat_mask = fake_quantize_decompose(pre_quant)

        # Local linear channel: weight mismatch + rounding.
        weight_err = x_quant @ (W_q - W).T + (b_q - b)
        linear_err = weight_err + round_err
        nonlinear_err = sat_err

        report = build_layer_geometry_report(
            layer_index=layer_idx,
            linear_error=linear_err,
            nonlinear_error=nonlinear_err,
            float_pre_activation=pre_float,
            quant_pre_activation=pre_quant,
            saturation_mask=sat_mask,
            operator_matrix=W,
            prev_total_error=prev_total_error,
            metadata={"scenario": name},
        )
        reports.append(report)

        if use_relu and layer_idx < len(weights) - 1:
            x_float = np.maximum(pre_float, 0.0)
            x_quant = np.maximum(quantized, 0.0)
        else:
            x_float = pre_float
            x_quant = quantized

        prev_total_error = x_quant - x_float

    return RunGeometryReport(
        model_name="LowDimMLP",
        task_name=name,
        bit_width=BITS,
        layer_reports=reports,
        metadata={"n_samples": int(X0.shape[0]), "dims": int(X0.shape[1])},
    )


def report_to_frame(run_report):
    rows = []
    for r in run_report.layer_reports:
        rows.append(
            {
                "layer": r.layer_index,
                "linear": r.linear_error_norm,
                "nonlinear": r.nonlinear_error_norm,
                "linear_fraction": r.linear_fraction,
                "flip_rate": r.relu_flip_rate,
                "sat_count": r.saturation_count,
                "correctability": r.correctability_score,
                "anisotropy": r.anisotropy_ratio,
                "entropy": r.entropy_proxy,
            }
        )
    return pd.DataFrame(rows)


X = RNG.normal(size=(1024, 2))

weights_linear = [
    np.array([[1.2, 0.0], [0.0, 0.9]]),
    np.array([[0.8, 0.0], [0.0, 1.1]]),
    np.array([[1.0, 0.0], [0.0, 1.0]]),
]
biases_linear = [np.zeros(2), np.zeros(2), np.zeros(2)]

weights_relu = [
    np.array([[1.3, -0.8], [0.9, -1.1]]),
    np.array([[0.7, 0.6], [-0.5, 0.8]]),
    np.array([[1.0, -0.4], [0.3, 1.0]]),
]
biases_relu = [np.array([-0.1, 0.05]), np.array([0.0, -0.05]), np.zeros(2)]

weights_sat = [
    np.array([[2.2, 0.7], [0.4, 2.0]]),
    np.array([[1.8, -0.9], [0.6, 1.7]]),
    np.array([[1.5, 0.5], [0.7, 1.6]]),
]
biases_sat = [np.zeros(2), np.zeros(2), np.zeros(2)]

reports = {
    "linear_only": simulate_stack("linear_only", X, weights_linear, biases_linear, use_relu=False),
    "relu_boundary": simulate_stack("relu_boundary", X, weights_relu, biases_relu, use_relu=True),
    "saturation_heavy": simulate_stack("saturation_heavy", X, weights_sat, biases_sat, use_relu=True, force_clip=0.9),
}

for name, report in reports.items():
    df = report_to_frame(report)
    print("\n" + "=" * 70)
    print(name)
    print(report.aggregate_metrics())
    print(df.round(4))

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, report) in zip(axes, reports.items()):
    df = report_to_frame(report)
    ax.plot(df["layer"], df["correctability"], "o-", label="correctability")
    ax.plot(df["layer"], df["linear_fraction"], "s--", label="linear_fraction")
    ax.set_title(name)
    ax.set_xlabel("layer")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
axes[0].legend()
plt.tight_layout()
plt.savefig("plots/lowdim_report_card_scores.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
BUDGET = 2

chosen_report = reports["relu_boundary"]
policy_a = score_correction_points(chosen_report, budget=BUDGET)
policy_b = build_baseline_policy(chosen_report, budget=BUDGET)

print("\nPolicy A (geometry-guided):", policy_a["selected_points"])
print("Policy B (baseline evenly-spaced):", policy_b["selected_points"])

rank_df = pd.DataFrame(policy_a["ranking"])
print("\nRanking")
print(rank_df.head(10).round(4))
