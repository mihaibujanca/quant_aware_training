"""
Cycle 312 / item 322 — Higher-width (hidden=128) MLP on 50D input eff_rank census.

Direct follow-up to cycle 307 (hidden=32) to test whether width unblocks the
layer-0 ReLU activation-pattern bottleneck. hidden=128 = 2.56x input_dim (vs
cycle 307 0.64x). If the Hanin-Rolnick N^d bottleneck story from cycle 307 is
correct, init eff_rank should stay elevated past layer 0. If the GPT-2 cycle
278 bathtub (b1=1.94, b2=1.05) is a generic ReLU phenomenon, training should
still collapse eff_rank.

Item 322 uses stable rank (sum sigma)^2 / sum sigma^2 of the activation matrix
directly (NOT spectral entropy of the Gram eigenvalues like cycle 307). The
difference: stable rank is based on singular values of Y (2000, 128); cycle 307
used the exp(entropy) of the Gram spectrum. Both are effective-rank proxies but
numerically different — the plot/comparison to cycle 307 is rough.

Verdict (item 322 verbatim):
  CONFIRMED iff post mean eff_rank <= 5 at >=8/12 layers across all 3 seeds
               AND trajectory monotone-decreasing at >=6/12 layers.
               ("training-induced eff_rank collapse is reproducible in MLPs
                once width >> input_dim; cycle 278 bathtub IS a generic ReLU
                phenomenon and cycle 307 collapse-failure was a width artifact")
  REFUTED   iff init mean eff_rank >= 60 at all 12 layers
               AND post mean eff_rank >= 30 at all 12 layers.
               ("ReLU MLPs at width 128 never collapse eff_rank below ~30;
                cycle 278 bathtub is genuinely transformer-specific")
  INCONCLUSIVE otherwise.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification

from aleph.models import MLP

# ---------- config ----------
SEEDS = [42, 137, 2024]
HIDDEN = 128
DEPTH = 12
N_SAMPLES = 2000
N_FEATURES = 50
N_INFORMATIVE = 10
N_CLASSES = 2
LR = 1e-3
MAX_EPOCHS = 20000
TRAIN_ACC_TARGET = 0.90
TRAJECTORY_EPOCHS = [0, 50, 200, 500, 2000]

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "autoresearch" / "results"
LOGS_DIR = REPO / "autoresearch" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS_DIR / "cycle_312_eff_rank_50d_w128.json"
SUMMARY_PATH = RESULTS_DIR / "cycle_312_summary.md"
LOG_PATH = LOGS_DIR / "cycle_312_run.log"


# ---------- utilities ----------
def make_dataset(seed):
    """50D make_classification with per-feature z-score standardisation."""
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=0,
        n_repeated=0,
        n_classes=N_CLASSES,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=1.0,
        random_state=seed,
    )
    X = X.astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sd
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(np.asarray(y)).float()
    return Xt, yt


def build_mlp(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(
        input_size=N_FEATURES,
        hidden_size=HIDDEN,
        output_size=1,
        depth=DEPTH,
        activation_fn=nn.ReLU,
    )
    return model


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_post_relu(model, X):
    """Walk model.layers and capture post-ReLU activations per hidden layer."""
    activations = []
    h = X
    seq = model.layers
    expected_len = DEPTH * 2 + 1  # Lin, ReLU, ..., Lin, ReLU, Lin_out
    assert len(seq) == expected_len, f"unexpected layers length {len(seq)} != {expected_len}"
    with torch.no_grad():
        for m in seq:
            h = m(h)
            if isinstance(m, nn.ReLU):
                activations.append(h.detach().clone())
    assert len(activations) == DEPTH, f"captured {len(activations)} activations, expected {DEPTH}"
    return activations


def eff_rank_from_Y(Y):
    """Item 322 verbatim: eff_rank = (sum sigma)^2 / sum sigma^2 where sigma
    are singular values of Y (2000, HIDDEN). This is the 'stable rank' of Y.

    Note: this differs from cycle 307's entropy-based eff_rank. Direct numeric
    comparison to cycle 307 numbers is rough — they are different functionals
    of the singular spectrum.
    """
    sv = torch.linalg.svdvals(Y)  # (min(N, HIDDEN),)
    sv = torch.clamp(sv, min=0.0)
    num = sv.sum().item()
    den = (sv * sv).sum().item()
    if den <= 0.0:
        return 1.0
    return (num * num) / den


def train_with_trajectory(model, X, y, lr, max_epochs, target_acc, traj_epochs):
    """Full-batch Adam. Capture eff_rank per layer at each traj_epoch (including 0)
    and train until target_acc reached or max_epochs exhausted."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    traj = {}

    if 0 in traj_epochs:
        acts = collect_post_relu(model, X)
        traj[0] = [eff_rank_from_Y(Y) for Y in acts]

    final_loss = float("nan")
    final_acc = 0.0
    target_reached_epoch = None
    traj_targets = set(e for e in traj_epochs if e > 0)

    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        logits = model(X).squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if epoch in traj_targets:
            acts = collect_post_relu(model, X)
            traj[epoch] = [eff_rank_from_Y(Y) for Y in acts]
        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                pred = (model(X).squeeze(-1) > 0.0).float()
                acc = (pred == y).float().mean().item()
            final_acc = acc
            final_loss = loss.item()
            if acc >= target_acc and target_reached_epoch is None:
                target_reached_epoch = epoch
                if all(t <= epoch for t in traj_targets):
                    break

    with torch.no_grad():
        pred = (model(X).squeeze(-1) > 0.0).float()
        acc = (pred == y).float().mean().item()
    final_acc = acc
    epochs_used = target_reached_epoch if target_reached_epoch is not None else max_epochs
    return final_acc, epochs_used, final_loss, traj


def census_for(seed, log_lines):
    t0 = time.time()
    X, y = make_dataset(seed)
    model = build_mlp(seed)
    n_params = trainable_param_count(model)

    init_acts = collect_post_relu(model, X)
    init_eff = [eff_rank_from_Y(Y) for Y in init_acts]

    final_acc, epochs, final_loss, traj = train_with_trajectory(
        model, X, y, LR, MAX_EPOCHS, TRAIN_ACC_TARGET, TRAJECTORY_EPOCHS
    )
    converged = final_acc >= TRAIN_ACC_TARGET

    post_acts = collect_post_relu(model, X)
    post_eff = [eff_rank_from_Y(Y) for Y in post_acts]
    wall = time.time() - t0
    line = (
        f"  seed={seed}: n_params={n_params} epochs={epochs} "
        f"train_acc={final_acc:.4f} loss={final_loss:.4f} converged={converged} wall={wall:.1f}s"
    )
    print(line, flush=True)
    log_lines.append(line)
    log_lines.append(f"    init_eff_rank={[round(x, 3) for x in init_eff]}")
    log_lines.append(f"    post_eff_rank={[round(x, 3) for x in post_eff]}")
    for e in sorted(traj.keys()):
        log_lines.append(
            f"    traj_e={e}: eff_rank={[round(x, 3) for x in traj[e]]}"
        )
    return {
        "seed": seed,
        "n_params": n_params,
        "epochs": epochs,
        "final_train_acc": final_acc,
        "final_train_loss": final_loss,
        "converged": converged,
        "init_eff_rank": init_eff,
        "post_eff_rank": post_eff,
        "trajectory": {str(k): v for k, v in sorted(traj.items())},
        "wall_sec": wall,
    }


def aggregate_per_layer(records):
    init = np.array([r["init_eff_rank"] for r in records])  # (S, DEPTH)
    post = np.array([r["post_eff_rank"] for r in records])
    agg = {
        "n_seeds": len(records),
        "init_mean": init.mean(axis=0).tolist(),
        "init_std": init.std(axis=0, ddof=0).tolist(),
        "post_mean": post.mean(axis=0).tolist(),
        "post_std": post.std(axis=0, ddof=0).tolist(),
    }
    epoch_keys = sorted(set.intersection(*[set(r["trajectory"].keys()) for r in records]), key=int)
    traj_mean = {}
    for ek in epoch_keys:
        arr = np.array([r["trajectory"][ek] for r in records])
        traj_mean[ek] = arr.mean(axis=0).tolist()
    agg["trajectory_mean"] = traj_mean
    agg["trajectory_epochs"] = epoch_keys
    return agg


def verdict_item_322(records, agg):
    """Apply item 322 thresholds verbatim.

    CONFIRMED iff:
      (a) post mean eff_rank <= 5 at >= 8/12 layers across all 3 seeds, i.e.
          count layers L with all 3 seeds post_eff_rank[L] <= 5; need >= 8.
      (b) trajectory monotone-decreasing at >= 6/12 layers (per-layer mean
          across seeds non-increasing over TRAJECTORY_EPOCHS).
    REFUTED iff:
      (c) init mean eff_rank >= 60 at all 12 layers, AND
      (d) post mean eff_rank >= 30 at all 12 layers.
    INCONCLUSIVE otherwise.
    """
    # (a)
    layers_post_le5 = 0
    per_layer_post_le5_flags = []
    for L in range(DEPTH):
        if all(r["post_eff_rank"][L] <= 5.0 for r in records):
            layers_post_le5 += 1
            per_layer_post_le5_flags.append(True)
        else:
            per_layer_post_le5_flags.append(False)
    cond_a = layers_post_le5 >= 8

    # (b)
    traj_mean = agg["trajectory_mean"]
    traj_epoch_keys = sorted(traj_mean.keys(), key=int)
    layers_monotone = 0
    per_layer_monotone_flags = []
    for L in range(DEPTH):
        series = [traj_mean[ek][L] for ek in traj_epoch_keys]
        is_mono = all(series[i + 1] <= series[i] + 1e-9 for i in range(len(series) - 1))
        per_layer_monotone_flags.append(is_mono)
        if is_mono:
            layers_monotone += 1
    cond_b = layers_monotone >= 6

    confirmed = cond_a and cond_b

    # (c) + (d) for REFUTED
    init_mean = agg["init_mean"]
    post_mean = agg["post_mean"]
    cond_c = all(im >= 60.0 for im in init_mean)
    cond_d = all(pm >= 30.0 for pm in post_mean)
    refuted = cond_c and cond_d

    if confirmed:
        verdict = "CONFIRMED"
    elif refuted:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "cond_a_layers_post_le5_allseeds": layers_post_le5,
        "cond_a_pass": cond_a,
        "cond_b_layers_trajectory_monotone": layers_monotone,
        "cond_b_pass": cond_b,
        "cond_c_init_mean_ge60_allL_pass": cond_c,
        "cond_d_post_mean_ge30_allL_pass": cond_d,
        "per_layer_post_le5": per_layer_post_le5_flags,
        "per_layer_monotone": per_layer_monotone_flags,
    }


def main():
    log_lines = []
    log_lines.append("Cycle 312 / item 322 — 50D eff_rank census at HIDDEN=128")
    log_lines.append(
        f"config: HIDDEN={HIDDEN} DEPTH={DEPTH} N={N_SAMPLES} N_FEATURES={N_FEATURES} "
        f"N_INFORMATIVE={N_INFORMATIVE} LR={LR} max_epochs={MAX_EPOCHS} "
        f"target_acc={TRAIN_ACC_TARGET} seeds={SEEDS} traj_epochs={TRAJECTORY_EPOCHS}"
    )
    print(log_lines[-2], flush=True)
    print(log_lines[-1], flush=True)

    t_start = time.time()
    records = []
    for seed in SEEDS:
        records.append(census_for(seed, log_lines))
    aggregate = aggregate_per_layer(records)
    verdict_info = verdict_item_322(records, aggregate)
    total_wall = time.time() - t_start

    summary = {
        "cycle": 312,
        "item": 322,
        "config": {
            "hidden": HIDDEN,
            "depth": DEPTH,
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_informative": N_INFORMATIVE,
            "n_classes": N_CLASSES,
            "lr": LR,
            "max_epochs": MAX_EPOCHS,
            "target_acc": TRAIN_ACC_TARGET,
            "seeds": SEEDS,
            "trajectory_epochs": TRAJECTORY_EPOCHS,
            "eff_rank_def": "stable_rank = (sum sigma)^2 / sum sigma^2 of SVD(Y)",
        },
        "records": records,
        "aggregate": aggregate,
        "verdict_info": verdict_info,
        "verdict": verdict_info["verdict"],
        "total_wall_sec": total_wall,
        "baselines": {
            "cycle_307_hidden32_post_mean_entropy": [
                13.475, 2.370, 1.745, 1.741, 1.767, 1.767,
                1.712, 1.677, 1.551, 1.524, 1.547, 1.743
            ],
            "cycle_307_hidden32_init_mean_entropy": [
                15.698, 5.926, 2.963, 1.349, 1.051, 1.009,
                1.002, 1.000, 1.000, 1.000, 1.000, 1.000
            ],
            "cycle_278_gpt2_c_proj_output_eff_rank": {
                "0": 71.93, "1": 1.94, "2": 1.05, "3": 24.29, "4": 49.04,
                "5": 179.19, "9": 195.5, "11": 22.49
            },
            "random_init_expected_hidden128": 128,
            "note": "cycle 307 used entropy-based eff_rank; cycle 312 uses stable rank per item 322 verbatim. Direct numeric comparison is rough.",
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    # Summary markdown
    lines = []
    lines.append("# Cycle 312 — 50D input eff_rank census at HIDDEN=128 (item 322)")
    lines.append("")
    lines.append(f"**Verdict:** {verdict_info['verdict']}")
    lines.append("")
    lines.append("## Config")
    lines.append(
        f"- MLP({N_FEATURES}, {HIDDEN}, 1, depth={DEPTH}, ReLU); Adam lr={LR}; full-batch BCEWithLogits"
    )
    lines.append(
        f"- make_classification(n_samples={N_SAMPLES}, n_features={N_FEATURES}, "
        f"n_informative={N_INFORMATIVE}, n_classes={N_CLASSES}) + per-feature z-score"
    )
    lines.append(f"- Stop when train_acc >= {TRAIN_ACC_TARGET} (cap {MAX_EPOCHS} epochs)")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- Trajectory epochs: {TRAJECTORY_EPOCHS}")
    lines.append(f"- Trainable params/net = {records[0]['n_params']}")
    lines.append("- eff_rank = (sum sigma)^2 / sum sigma^2 of SVD(Y) (stable rank, item 322 verbatim)")
    lines.append("")
    lines.append("## Verdict logic (item 322 verbatim)")
    lines.append(
        "- CONFIRMED iff (a) post mean eff_rank <= 5 at >= 8/12 layers across all 3 seeds "
        "AND (b) trajectory monotone-decreasing at >= 6/12 layers."
    )
    lines.append(
        "- REFUTED iff (c) init mean eff_rank >= 60 at all 12 layers AND (d) post mean >= 30 at all 12 layers."
    )
    lines.append("- INCONCLUSIVE otherwise.")
    lines.append("")
    lines.append("## Per-layer aggregates (mean ± std across 3 seeds)")
    lines.append("")
    lines.append("| layer | init_mean ± std | post_mean ± std |")
    lines.append("|------:|----------------:|----------------:|")
    for L in range(DEPTH):
        lines.append(
            f"| {L} | {aggregate['init_mean'][L]:.3f} ± {aggregate['init_std'][L]:.3f} "
            f"| {aggregate['post_mean'][L]:.3f} ± {aggregate['post_std'][L]:.3f} |"
        )
    lines.append("")
    lines.append("## Trajectory (mean across 3 seeds)")
    lines.append("")
    header = "| layer |" + "".join(f" e={ek} |" for ek in aggregate["trajectory_epochs"])
    sep = "|------:|" + "".join("------:|" for _ in aggregate["trajectory_epochs"])
    lines.append(header)
    lines.append(sep)
    for L in range(DEPTH):
        row = f"| {L} |"
        for ek in aggregate["trajectory_epochs"]:
            row += f" {aggregate['trajectory_mean'][ek][L]:.3f} |"
        lines.append(row)
    lines.append("")
    lines.append("## Verdict clause breakdown")
    lines.append(
        f"- (a) post eff_rank <= 5 layers (all seeds): **{verdict_info['cond_a_layers_post_le5_allseeds']}/12** "
        f"(need >= 8 -> {verdict_info['cond_a_pass']})"
    )
    lines.append(
        f"- (b) trajectory monotone decreasing layers: **{verdict_info['cond_b_layers_trajectory_monotone']}/12** "
        f"(need >= 6 -> {verdict_info['cond_b_pass']})"
    )
    lines.append(
        f"- (c) init mean >= 60 at all 12 layers: **{verdict_info['cond_c_init_mean_ge60_allL_pass']}**"
    )
    lines.append(
        f"- (d) post mean >= 30 at all 12 layers: **{verdict_info['cond_d_post_mean_ge30_allL_pass']}**"
    )
    lines.append("")
    lines.append("## Per-seed convergence")
    for r in records:
        lines.append(
            f"- seed={r['seed']}: epochs={r['epochs']}, train_acc={r['final_train_acc']:.4f}, "
            f"converged={r['converged']}"
        )
    lines.append("")
    lines.append(f"Total wall: {total_wall:.1f}s")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")

    log_lines.append("")
    log_lines.append("=== Aggregate per-layer means ===")
    log_lines.append(f"    init_mean={[round(x,3) for x in aggregate['init_mean']]}")
    log_lines.append(f"    post_mean={[round(x,3) for x in aggregate['post_mean']]}")
    for ek in aggregate["trajectory_epochs"]:
        log_lines.append(
            f"    traj_e={ek} mean={[round(x,3) for x in aggregate['trajectory_mean'][ek]]}"
        )
    log_lines.append(
        f"cond_a_layers_post_le5_allseeds={verdict_info['cond_a_layers_post_le5_allseeds']}/12 pass={verdict_info['cond_a_pass']}"
    )
    log_lines.append(
        f"cond_b_layers_trajectory_monotone={verdict_info['cond_b_layers_trajectory_monotone']}/12 pass={verdict_info['cond_b_pass']}"
    )
    log_lines.append(
        f"cond_c_init_mean_ge60_allL={verdict_info['cond_c_init_mean_ge60_allL_pass']}"
    )
    log_lines.append(
        f"cond_d_post_mean_ge30_allL={verdict_info['cond_d_post_mean_ge30_allL_pass']}"
    )
    log_lines.append(f"OVERALL VERDICT: {verdict_info['verdict']}")
    log_lines.append(f"total wall: {total_wall:.1f}s")
    LOG_PATH.write_text("\n".join(log_lines) + "\n")

    print()
    print(f"OVERALL VERDICT: {verdict_info['verdict']}")
    print(
        f"cond_a={verdict_info['cond_a_pass']} cond_b={verdict_info['cond_b_pass']} "
        f"cond_c={verdict_info['cond_c_init_mean_ge60_allL_pass']} cond_d={verdict_info['cond_d_post_mean_ge30_allL_pass']}"
    )
    print(f"total wall: {total_wall:.1f}s")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
