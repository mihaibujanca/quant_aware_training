"""
Cycle 307 / item 317 — Higher-dim (50D) input eff_rank census on trained MLP.

Follow-up to cycle 295 (2D spirals/moons): the 2D-input testbed cannot answer
"does training collapse eff_rank" because random Gaussian + ReLU on a 2D input
already collapses init eff_rank to ~1 by layer 7 (Hanin & Rolnick N^d bound).
This cycle raises input dimension to 50 and asks whether training actually
collapses eff_rank when there is high-rank slack at init.

For each seed {42, 137, 2024}:
  - Draw X, y from sklearn.datasets.make_classification(n_samples=2000,
        n_features=50, n_informative=10, n_classes=2, random_state=seed),
    then per-feature z-score standardise X.
  - Build MLP(50 -> 32 -> 32 -> ... -> 32 -> 1, depth=12 hidden, ReLU).
  - Capture post-ReLU activations Y_l at INIT (no training).
  - Train Adam lr=1e-3 BCEWithLogitsLoss full-batch until train_acc >= 0.90
    (cap MAX_EPOCHS=20000). Capture activations again at target epochs for
    the trajectory subrecord at {0, 50, 200, 500, 2000} steps.
  - Capture post-ReLU activations after training.
  - Per layer compute eff_rank = exp(-sum p_i log p_i) where p_i are
    normalised eigenvalues of C_l = Y_l^T Y_l / N (spectral entropy).

Verdict (item 317 verbatim):
  CONFIRMED iff (a) init mean eff_rank >= 15 at layers 1-6 across all 3
               seeds (6 * 3 = 18 per-seed/per-layer cells all pass) AND
               (b) post-training mean eff_rank <= 5 at >= 8 of 12 layers
               across all 3 seeds AND
               (c) trajectory is MONOTONE DECREASING at >= 6 of 12 layers
               (layer monotone iff across 5 trajectory epochs per-epoch
               mean eff_rank across seeds is non-increasing).
  REFUTED   iff init mean eff_rank >= 15 holds at layers 1-6 (precondition
               present) but post-training eff_rank <= 5 at < 4 of 12 layers.
  INCONCLUSIVE otherwise.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification

from aleph.models import MLP

# ---------- config ----------
SEEDS = [42, 137, 2024]
HIDDEN = 32
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

JSON_PATH = RESULTS_DIR / "cycle_307_eff_rank_50d.json"
SUMMARY_PATH = RESULTS_DIR / "cycle_307_summary.md"
LOG_PATH = LOGS_DIR / "cycle_307_run.log"


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
    """eff_rank = exp(-sum p_i log p_i) where p_i = lambda_i / sum(lambda_i),
    lambda_i = eigenvalues of Y^T Y / N (clipped to >= 0)."""
    N = Y.shape[0]
    C = (Y.t() @ Y) / float(N)
    eig = torch.linalg.eigvalsh(C)
    eig = torch.clamp(eig, min=0.0)
    s = eig.sum()
    if s.item() <= 0.0:
        return 1.0
    p = eig / s
    nz = p > 0
    H = -(p[nz] * torch.log(p[nz])).sum().item()
    return math.exp(H)


def train_with_trajectory(model, X, y, lr, max_epochs, target_acc, traj_epochs):
    """Full-batch Adam. Capture eff_rank per layer at each traj_epoch (including 0)
    and train until target_acc reached or max_epochs exhausted."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    traj = {}

    # Capture at epoch 0 BEFORE any step
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
                # Don't break — keep going only if we need to hit more traj targets
                if all(t <= epoch for t in traj_targets):
                    break

    # Make sure any remaining traj targets that are beyond where we stopped get
    # captured as end-of-training values (pad with post-training eff_rank). But
    # the safer behaviour is: if we stopped before a traj target, skip it.
    # Still, we ensure we at least have the end captures via the caller.
    with torch.no_grad():
        pred = (model(X).squeeze(-1) > 0.0).float()
        acc = (pred == y).float().mean().item()
    final_acc = acc
    if not math.isnan(final_loss):
        pass
    else:
        final_loss = loss.item() if "loss" in locals() else float("nan")

    epochs_used = target_reached_epoch if target_reached_epoch is not None else max_epochs
    return final_acc, epochs_used, final_loss, traj


def census_for(seed, log_lines):
    t0 = time.time()
    X, y = make_dataset(seed)
    model = build_mlp(seed)
    n_params = trainable_param_count(model)

    # init activations (pre-training, pre-epoch 0 trajectory capture done inside)
    init_acts = collect_post_relu(model, X)
    init_eff = [eff_rank_from_Y(Y) for Y in init_acts]

    # train and record trajectory (trajectory captures including epoch 0 are
    # internal; we also keep the external init_eff for the verdict clauses)
    final_acc, epochs, final_loss, traj = train_with_trajectory(
        model, X, y, LR, MAX_EPOCHS, TRAIN_ACC_TARGET, TRAJECTORY_EPOCHS
    )
    converged = final_acc >= TRAIN_ACC_TARGET

    # post-training activations
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
    """Mean per-layer eff_rank across seeds (init and post), plus trajectory
    mean per epoch."""
    init = np.array([r["init_eff_rank"] for r in records])  # (S, DEPTH)
    post = np.array([r["post_eff_rank"] for r in records])
    agg = {
        "n_seeds": len(records),
        "init_mean": init.mean(axis=0).tolist(),
        "init_std": init.std(axis=0, ddof=0).tolist(),
        "post_mean": post.mean(axis=0).tolist(),
        "post_std": post.std(axis=0, ddof=0).tolist(),
    }
    # trajectory: only include epochs captured by all seeds
    epoch_keys = sorted(set.intersection(*[set(r["trajectory"].keys()) for r in records]), key=int)
    traj_mean = {}
    for ek in epoch_keys:
        arr = np.array([r["trajectory"][ek] for r in records])
        traj_mean[ek] = arr.mean(axis=0).tolist()
    agg["trajectory_mean"] = traj_mean
    agg["trajectory_epochs"] = epoch_keys
    return agg


def verdict_item_317(records, agg):
    """Apply item 317 thresholds.

    (a) init mean eff_rank >= 15 at layers 1-6 across all 3 seeds.
        Interpreted as: for every seed AND every layer in [1..6] (6 layers,
        1-indexed => zero-indexed slice 1:7), per-seed-per-layer init eff_rank
        >= 15. Reported as fraction 18 cells (= 3 seeds * 6 layers).
    (b) post-training mean eff_rank <= 5 at >= 8 of 12 layers across all 3
        seeds. Interpreted as: count layers L such that all 3 seeds have
        post_eff_rank[L] <= 5; need that count >= 8.
    (c) trajectory monotone decreasing at >= 6 of 12 layers. For each layer,
        check that the per-epoch MEAN eff_rank across seeds is non-increasing
        over TRAJECTORY_EPOCHS; count such layers; need >= 6.

    REFUTED iff (a) passes but count-layers-with-post<=5 across-seeds < 4.
    """
    layers_1_6 = list(range(1, 7))  # zero-indexed layers 1..6 = hidden layers 2..7

    # (a): per-seed per-layer init eff_rank >= 15 for all cells in layers_1_6
    cells_total = len(records) * len(layers_1_6)
    cells_pass = 0
    for r in records:
        for L in layers_1_6:
            if r["init_eff_rank"][L] >= 15.0:
                cells_pass += 1
    cond_a = cells_pass == cells_total

    # (b): layers with all-seeds post eff_rank <= 5
    layers_post_le5 = 0
    per_layer_post_le5_flags = []
    for L in range(DEPTH):
        if all(r["post_eff_rank"][L] <= 5.0 for r in records):
            layers_post_le5 += 1
            per_layer_post_le5_flags.append(True)
        else:
            per_layer_post_le5_flags.append(False)
    cond_b = layers_post_le5 >= 8

    # (c): trajectory monotone non-increasing (mean across seeds) at layer L
    # over TRAJECTORY_EPOCHS (only epochs present in agg.trajectory_mean)
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
    cond_c = layers_monotone >= 6

    confirmed = cond_a and cond_b and cond_c

    # REFUTED: (a) holds but post <= 5 layers < 4
    refuted = cond_a and (layers_post_le5 < 4)

    if confirmed:
        verdict = "CONFIRMED"
    elif refuted:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "cond_a_init_ge15_cells_1_6": f"{cells_pass}/{cells_total}",
        "cond_a_pass": cond_a,
        "cond_b_layers_post_le5_allseeds": layers_post_le5,
        "cond_b_pass": cond_b,
        "cond_c_layers_trajectory_monotone": layers_monotone,
        "cond_c_pass": cond_c,
        "per_layer_post_le5": per_layer_post_le5_flags,
        "per_layer_monotone": per_layer_monotone_flags,
    }


def main():
    log_lines = []
    log_lines.append("Cycle 307 / item 317 — 50D eff_rank census")
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
    verdict_info = verdict_item_317(records, aggregate)
    total_wall = time.time() - t_start

    summary = {
        "cycle": 307,
        "item": 317,
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
        },
        "records": records,
        "aggregate": aggregate,
        "verdict_info": verdict_info,
        "verdict": verdict_info["verdict"],
        "total_wall_sec": total_wall,
        "baselines": {
            "cycle_295_spirals_post_mean": [
                3.14, 2.86, 2.11, 2.09, 2.15, 1.94, 1.99, 2.07, 2.05, 2.03, 1.76, 1.74
            ],
            "cycle_295_spirals_init_mean": [
                2.99, 2.71, 1.87, 1.19, 1.06, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00
            ],
            "cycle_278_gpt2_c_proj_output_eff_rank": {
                "0": 71.93, "1": 1.94, "2": 1.05, "3": 24.29, "4": 49.04,
                "5": 179.19, "9": 195.5, "11": 22.49
            },
            "random_init_expected_32_wide": 32,
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    # Summary markdown
    lines = []
    lines.append("# Cycle 307 — 50D input eff_rank census (item 317)")
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
    lines.append("")
    lines.append("## Verdict logic (item 317 verbatim)")
    lines.append(
        "- CONFIRMED iff (a) init mean eff_rank >= 15 at layers 1-6 across all 3 seeds "
        "AND (b) post-training mean eff_rank <= 5 at >= 8 of 12 layers across all 3 seeds "
        "AND (c) trajectory monotone decreasing at >= 6 of 12 layers."
    )
    lines.append("- REFUTED iff (a) passes but post-training eff_rank <= 5 at < 4 of 12 layers.")
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
        f"- (a) init eff_rank >= 15 at layers 1-6 per-seed: **{verdict_info['cond_a_init_ge15_cells_1_6']}** "
        f"(need 18/18 → {verdict_info['cond_a_pass']})"
    )
    lines.append(
        f"- (b) layers with post eff_rank <= 5 across all seeds: **{verdict_info['cond_b_layers_post_le5_allseeds']}/12** "
        f"(need >= 8 → {verdict_info['cond_b_pass']})"
    )
    lines.append(
        f"- (c) trajectory monotone decreasing layers: **{verdict_info['cond_c_layers_trajectory_monotone']}/12** "
        f"(need >= 6 → {verdict_info['cond_c_pass']})"
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
        f"cond_a_init_ge15_cells_1_6={verdict_info['cond_a_init_ge15_cells_1_6']} pass={verdict_info['cond_a_pass']}"
    )
    log_lines.append(
        f"cond_b_layers_post_le5_allseeds={verdict_info['cond_b_layers_post_le5_allseeds']}/12 pass={verdict_info['cond_b_pass']}"
    )
    log_lines.append(
        f"cond_c_layers_trajectory_monotone={verdict_info['cond_c_layers_trajectory_monotone']}/12 pass={verdict_info['cond_c_pass']}"
    )
    log_lines.append(f"OVERALL VERDICT: {verdict_info['verdict']}")
    log_lines.append(f"total wall: {total_wall:.1f}s")
    LOG_PATH.write_text("\n".join(log_lines) + "\n")

    print()
    print(f"OVERALL VERDICT: {verdict_info['verdict']}")
    print(f"cond_a={verdict_info['cond_a_pass']} cond_b={verdict_info['cond_b_pass']} cond_c={verdict_info['cond_c_pass']}")
    print(f"total wall: {total_wall:.1f}s")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
