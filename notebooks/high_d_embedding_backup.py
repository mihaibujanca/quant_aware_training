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
#     display_name: quant-aware-training (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # High-D Embedding Experiments (Phase 2)
#
# Embed 3D SDF data into R^d via random orthogonal projection and compare
# source/compressed model degradation as d increases.
#
# Core diagnostic: SVD of the gradient ensemble `G = [∇f(x_i)]_i` to test whether
# differential behavior remains effectively 3-dimensional.

# %% Imports
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from aleph.geometric_compression import (
    GeometricCompressionConfig,
    _make_kan,
    jacobian_rank_profile,
    run_pipeline,
    save_results,
)
from aleph.mesh_datasets import load_mesh, make_embedding_matrix, sample_sdf_points
from aleph.models import MLP

# %% Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

EMBED_DIMS = [3, 10, 50, 100]
PRIMARY_SHAPE = "torus"

PHASE1_SHAPES = ["torus", "sphere", "capsule", "box", "cylinder"]
CROSS_SHAPE_DIMS = [3, 50]

# Heavy by default for research-quality curves; switch on for a smoke run.
FAST_MODE = False

SAVE_RUNS = True
SAVE_TABLES = True
RESULT_ROOT = Path("runs/geometric_compression/high_d")
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

N_JACOBIAN_EVAL = 4096 if FAST_MODE else 10_000
JACOBIAN_CHUNK = 512

BASE_CFG = GeometricCompressionConfig(
    mesh_name=PRIMARY_SHAPE,
    n_points=30_000 if FAST_MODE else 100_000,
    surface_ratio=0.5,
    surface_noise=0.01,
    hidden_dim=256,
    depth=4,
    lr=1e-3,
    epochs=200 if FAST_MODE else 500,
    eikonal_lambda=0.1,
    n_clusters=8,
    kan_grid=5,
    kan_k=3,
    jacobian_lambda=1.0,
    kan_fit_steps=200 if FAST_MODE else 500,
    embed_dim=None,
    embed_seed=SEED,
    device=DEVICE,
    seed=SEED,
    batch_size=4096,
    n_profile=4_000 if FAST_MODE else 10_000,
    n_fit=3_000 if FAST_MODE else 5_000,
    grid_res=48 if FAST_MODE else 64,
)

print(f"Device: {DEVICE}")
print(f"FAST_MODE={FAST_MODE}, SAVE_RUNS={SAVE_RUNS}")


# %% Helpers
def build_cfg(mesh_name: str, embed_dim: int | None) -> GeometricCompressionConfig:
    return replace(BASE_CFG, mesh_name=mesh_name, embed_dim=embed_dim)


def instantiate_models(result: dict, device: str) -> dict[str, torch.nn.Module]:
    cfg = result["config"]
    profile = result["geometric_profile"]
    state_dicts = result["all_state_dicts"]

    input_dim = int(profile["input_dim"])
    hidden_dim = int(cfg["hidden_dim"])
    depth = int(cfg["depth"])
    kan_width = profile["kan_width"]
    kan_grid = int(profile["kan_grid_used"])
    kan_k = int(cfg.get("kan_k", 3))
    seed = int(cfg["seed"])
    rbf_width_scale = float(cfg.get("rbf_width_scale", 1.0))
    small_hidden = int(profile["small_mlp_hidden"])

    source_mlp = MLP(
        input_size=input_dim,
        hidden_size=hidden_dim,
        output_size=1,
        depth=depth,
    ).to(device)
    source_mlp.load_state_dict(state_dicts["source_mlp"])
    source_mlp.eval()

    geometric_kan = _make_kan(
        kan_width, kan_grid, seed, device, kan_k=kan_k, rbf_width_scale=rbf_width_scale
    )
    geometric_kan.load_state_dict(state_dicts["geometric_kan"])
    geometric_kan.eval()

    models = {"source_mlp": source_mlp, "geometric_kan": geometric_kan}

    if "distilled_kan" in state_dicts:
        distilled_kan = _make_kan(
            kan_width, kan_grid, seed, device, kan_k=kan_k, rbf_width_scale=rbf_width_scale
        )
        distilled_kan.load_state_dict(state_dicts["distilled_kan"])
        distilled_kan.eval()
        models["distilled_kan"] = distilled_kan

    if "small_mlp" in state_dicts and small_hidden > 0:
        small_mlp = MLP(
            input_size=input_dim,
            hidden_size=small_hidden,
            output_size=1,
            depth=2,
        ).to(device)
        small_mlp.load_state_dict(state_dicts["small_mlp"])
        small_mlp.eval()
        models["small_mlp"] = small_mlp

    if "kan_scratch" in state_dicts:
        kan_scratch = _make_kan(
            kan_width, kan_grid, seed, device, kan_k=kan_k, rbf_width_scale=rbf_width_scale
        )
        kan_scratch.load_state_dict(state_dicts["kan_scratch"])
        kan_scratch.eval()
        models["kan_scratch"] = kan_scratch

    return models


def make_eval_inputs(mesh_name: str, cfg_dict: dict, n_points: int, device: str) -> torch.Tensor:
    mesh = load_mesh(mesh_name)
    X_3d, _ = sample_sdf_points(
        mesh,
        n_points=n_points,
        surface_ratio=float(cfg_dict["surface_ratio"]),
        surface_noise=float(cfg_dict["surface_noise"]),
    )
    X_3d = X_3d.to(device)

    embed_dim = cfg_dict.get("embed_dim")
    if embed_dim is None:
        return X_3d

    A = make_embedding_matrix(3, int(embed_dim), seed=int(cfg_dict["embed_seed"])).to(device)
    return X_3d @ A.T


def run_one(mesh_name: str, embed_dim: int | None, run_group: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg = build_cfg(mesh_name, embed_dim)
    result = run_pipeline(cfg)

    dim_label = "native3d" if embed_dim is None else f"d{embed_dim}"
    out_dir = RESULT_ROOT / run_group / mesh_name / dim_label
    if SAVE_RUNS:
        save_results(result, out_dir)

    metrics_rows = []
    for method, vals in result["metrics"].items():
        row = {
            "shape": mesh_name,
            "embed_dim": 3 if embed_dim is None else int(embed_dim),
            "embed_mode": "native" if embed_dim is None else "embedded",
            "method": method,
            "test_mse": float(vals["test_mse"]),
            "jacobian_mse": float(vals["jacobian_mse"]),
            "chamfer": float(vals.get("chamfer", np.nan)),
            "hausdorff": float(vals.get("hausdorff", np.nan)),
            "params": int(vals["params"]),
            "compression_ratio": float(vals["compression_ratio"]),
        }
        metrics_rows.append(row)

    X_eval = make_eval_inputs(mesh_name, result["config"], N_JACOBIAN_EVAL, DEVICE)
    models = instantiate_models(result, DEVICE)

    jac_rows = []
    for method, model in models.items():
        prof = jacobian_rank_profile(model, X_eval, chunk_size=JACOBIAN_CHUNK)
        svals = prof["singular_values"].numpy()
        energy = prof["energy_ratio"].numpy()

        row = {
            "shape": mesh_name,
            "embed_dim": 3 if embed_dim is None else int(embed_dim),
            "embed_mode": "native" if embed_dim is None else "embedded",
            "method": method,
            "effective_rank": float(prof["effective_rank"]),
            "rank_90": int(prof["rank_90"]),
            "rank_95": int(prof["rank_95"]),
            "rank_99": int(prof["rank_99"]),
            "top3_energy": float(energy[:3].sum()),
            "top5_energy": float(energy[:5].sum()),
            "s1": float(svals[0]) if len(svals) > 0 else np.nan,
            "s2": float(svals[1]) if len(svals) > 1 else np.nan,
            "s3": float(svals[2]) if len(svals) > 2 else np.nan,
            "s4": float(svals[3]) if len(svals) > 3 else np.nan,
            "singular_values_top10": svals[:10].tolist(),
        }
        jac_rows.append(row)

    return result, pd.DataFrame(metrics_rows), pd.DataFrame(jac_rows)


# %% [markdown]
# ## 1) Single-shape embedding sweep (torus)

# %% Run torus sweep across EMBED_DIMS
sweep_results = {}
sweep_metrics_frames = []
sweep_jac_frames = []

for d in EMBED_DIMS:
    print(f"Running shape={PRIMARY_SHAPE}, embed_dim={d}")
    result, metrics_df, jac_df = run_one(PRIMARY_SHAPE, d, run_group="torus_sweep")
    sweep_results[d] = result
    sweep_metrics_frames.append(metrics_df)
    sweep_jac_frames.append(jac_df)

sweep_metrics = pd.concat(sweep_metrics_frames, ignore_index=True)
sweep_jac = pd.concat(sweep_jac_frames, ignore_index=True)

display(sweep_metrics.sort_values(["embed_dim", "method"]).head(20))
display(sweep_jac.sort_values(["embed_dim", "method"]).head(20))


# %% [markdown]
# ## 2) Jacobian rank diagnostics
#
# Expectation under the hypothesis:
# - dominant gradient subspace remains ~3D even as ambient `d` grows
# - top-3 energy remains high

# %% Rank summary for source and geometric KAN
rank_summary = sweep_jac[sweep_jac["method"].isin(["source_mlp", "geometric_kan"])].copy()
rank_summary = rank_summary[
    ["embed_dim", "method", "effective_rank", "rank_90", "rank_95", "rank_99", "top3_energy", "top5_energy"]
].sort_values(["method", "embed_dim"])
display(rank_summary)


# %% Plot effective rank and top-3 energy vs embed dim
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for method in ["source_mlp", "geometric_kan", "distilled_kan", "small_mlp", "kan_scratch"]:
    sub = sweep_jac[sweep_jac["method"] == method].sort_values("embed_dim")
    axes[0].plot(sub["embed_dim"], sub["effective_rank"], marker="o", label=method)
    axes[1].plot(sub["embed_dim"], sub["top3_energy"], marker="o", label=method)

axes[0].set_title("Gradient-ensemble effective rank")
axes[0].set_xlabel("Embedding dimension d")
axes[0].set_ylabel("Effective rank")
axes[0].grid(alpha=0.3)

axes[1].set_title("Top-3 singular-value energy")
axes[1].set_xlabel("Embedding dimension d")
axes[1].set_ylabel("Energy captured by top 3")
axes[1].set_ylim(0.0, 1.02)
axes[1].grid(alpha=0.3)
axes[1].legend(loc="best", fontsize=8)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 3) Degradation curves (MSE vs d)

# %% Plot test MSE / Jacobian MSE vs d
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for method in ["source_mlp", "geometric_kan", "distilled_kan", "small_mlp", "kan_scratch"]:
    sub = sweep_metrics[sweep_metrics["method"] == method].sort_values("embed_dim")
    axes[0].plot(sub["embed_dim"], sub["test_mse"], marker="o", label=method)
    axes[1].plot(sub["embed_dim"], sub["jacobian_mse"], marker="o", label=method)

axes[0].set_title(f"{PRIMARY_SHAPE}: Test MSE vs embedding dimension")
axes[0].set_xlabel("Embedding dimension d")
axes[0].set_ylabel("Test MSE")
axes[0].set_yscale("log")
axes[0].grid(alpha=0.3)

axes[1].set_title(f"{PRIMARY_SHAPE}: Jacobian MSE vs embedding dimension")
axes[1].set_xlabel("Embedding dimension d")
axes[1].set_ylabel("Jacobian MSE")
axes[1].set_yscale("log")
axes[1].grid(alpha=0.3)
axes[1].legend(loc="best", fontsize=8)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4) Verification: native 3D vs embedded d=3
#
# `embed_dim=3` should behave close to native 3D (rotation in input space).

# %% Native baseline vs embedded d=3
native_result, native_metrics_df, native_jac_df = run_one(PRIMARY_SHAPE, None, run_group="native_vs_d3")
embedded_d3_metrics = sweep_metrics[sweep_metrics["embed_dim"] == 3].copy()
embedded_d3_jac = sweep_jac[sweep_jac["embed_dim"] == 3].copy()

m_native = native_metrics_df.set_index("method")
m_d3 = embedded_d3_metrics.set_index("method")
m_compare = pd.DataFrame(
    {
        "native_test_mse": m_native["test_mse"],
        "embedded_d3_test_mse": m_d3["test_mse"],
        "native_jacobian_mse": m_native["jacobian_mse"],
        "embedded_d3_jacobian_mse": m_d3["jacobian_mse"],
    }
)
m_compare["test_mse_ratio_d3_over_native"] = m_compare["embedded_d3_test_mse"] / m_compare["native_test_mse"]
m_compare["jacobian_mse_ratio_d3_over_native"] = (
    m_compare["embedded_d3_jacobian_mse"] / m_compare["native_jacobian_mse"]
)
display(m_compare.sort_index())

j_native = native_jac_df.set_index("method")
j_d3 = embedded_d3_jac.set_index("method")
j_compare = pd.DataFrame(
    {
        "native_effective_rank": j_native["effective_rank"],
        "embedded_d3_effective_rank": j_d3["effective_rank"],
        "native_top3_energy": j_native["top3_energy"],
        "embedded_d3_top3_energy": j_d3["top3_energy"],
    }
)
display(j_compare.sort_index())


# %% [markdown]
# ## 5) Per-shape comparison at d=3 vs d=50
#
# One network per `(shape, d)` combination.

# %% Cross-shape run (Phase 1 shapes)
cross_metrics_frames = []
cross_jac_frames = []

for shape in PHASE1_SHAPES:
    for d in CROSS_SHAPE_DIMS:
        print(f"Running shape={shape}, embed_dim={d}")
        _, m_df, j_df = run_one(shape, d, run_group="cross_shape")
        cross_metrics_frames.append(m_df)
        cross_jac_frames.append(j_df)

cross_metrics = pd.concat(cross_metrics_frames, ignore_index=True)
cross_jac = pd.concat(cross_jac_frames, ignore_index=True)

display(cross_metrics.sort_values(["shape", "embed_dim", "method"]).head(30))
display(cross_jac.sort_values(["shape", "embed_dim", "method"]).head(30))


# %% Build table: shape x method x d
d_small, d_large = CROSS_SHAPE_DIMS
table = (
    cross_metrics.pivot_table(
        index=["shape", "method"],
        columns="embed_dim",
        values="test_mse",
        aggfunc="first",
    )
    .reset_index()
    .rename(columns={d_small: f"mse_d{d_small}", d_large: f"mse_d{d_large}"})
)
table[f"mse_ratio_d{d_large}_over_d{d_small}"] = table[f"mse_d{d_large}"] / table[f"mse_d{d_small}"]

display(table.sort_values(["shape", "method"]))


# %% [markdown]
# ## 6) Save summary tables

# %% Save CSV summaries
if SAVE_TABLES:
    summary_dir = RESULT_ROOT / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    sweep_metrics.to_csv(summary_dir / "torus_sweep_metrics.csv", index=False)
    sweep_jac.to_csv(summary_dir / "torus_sweep_jacobian_rank.csv", index=False)
    m_compare.to_csv(summary_dir / "native_vs_embedded_d3_metrics.csv")
    j_compare.to_csv(summary_dir / "native_vs_embedded_d3_rank.csv")
    cross_metrics.to_csv(summary_dir / "cross_shape_metrics.csv", index=False)
    cross_jac.to_csv(summary_dir / "cross_shape_jacobian_rank.csv", index=False)
    table.to_csv(summary_dir / "cross_shape_mse_table.csv", index=False)

    print(f"Saved tables under: {summary_dir}")
