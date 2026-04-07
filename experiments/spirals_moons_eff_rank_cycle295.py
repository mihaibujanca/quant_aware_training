"""
Cycle 295 / item 305 — Effective rank census on trained spirals/moons MLPs.

For each dataset {spirals, moons} and seed {42, 137, 2024}:
  - Build MLP(2 -> 32 -> 32 -> ... -> 32 -> 1, depth=12 hidden, ReLU)
  - Capture post-ReLU activations Y_l ∈ R^(2000, 32) at INIT
  - Train Adam lr=1e-3 BCEWithLogitsLoss full-batch until train_acc >= 0.95
  - Capture post-ReLU activations Y_l after training
  - Per layer compute eff_rank = exp(- sum p_i log p_i)
    where p_i are normalized eigenvalues of C_l = Y_l^T Y_l / N

Verdict (item 305 verbatim):
  CONFIRMED iff post-training eff_rank <= 3 in >= 8 of 12 layers
             AND init eff_rank >= 20 in >= 8 of 12 layers
  REFUTED   iff post-training eff_rank >= 10 in >= 6 layers
  INCONCLUSIVE otherwise
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

from aleph.datasets import make_spirals
from aleph.models import MLP

# ---------- config ----------
SEEDS = [42, 137, 2024]
HIDDEN = 32
DEPTH = 12  # number of hidden Linear+ReLU layers; output Linear is +1 on top
N_SAMPLES = 2000
SPIRALS_NOISE = 0.5
SPIRALS_TURNS = 3
SPIRALS_INPUT_SCALE = 2.0  # MEMORY.md spirals recipe for depth-12 32-wide
MOONS_NOISE = 0.1
LR = 1e-3
MAX_EPOCHS = 20000
TRAIN_ACC_TARGET = 0.95

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "autoresearch" / "results"
LOGS_DIR = REPO / "autoresearch" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS_DIR / "cycle_295_spirals_moons_eff_rank.json"
SUMMARY_PATH = RESULTS_DIR / "cycle_295_summary.md"
LOG_PATH = LOGS_DIR / "cycle_295_run.log"


# ---------- utilities ----------
def make_dataset(name, seed):
    if name == "spirals":
        X, y = make_spirals(
            n_samples=N_SAMPLES, noise=SPIRALS_NOISE, n_turns=SPIRALS_TURNS, random_state=seed
        )
        X = X * SPIRALS_INPUT_SCALE  # MEMORY.md recipe: 2x scale for depth-12 trainability
    elif name == "moons":
        X, y = make_moons(n_samples=N_SAMPLES, noise=MOONS_NOISE, random_state=seed)
    else:
        raise ValueError(name)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(np.asarray(y)).float()
    return Xt, yt


def build_mlp(seed):
    """Build MLP(2, 32, 1, depth=12 hidden) with ReLU. Identity quant stubs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(input_size=2, hidden_size=HIDDEN, output_size=1, depth=DEPTH, activation_fn=nn.ReLU)
    return model


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_post_relu(model, X):
    """
    Walk model.layers (Sequential of Linear, ReLU, Linear, ReLU, ..., Linear)
    and capture the activation AFTER each ReLU. Return list of length DEPTH,
    each tensor of shape (N, HIDDEN).
    """
    activations = []
    h = X
    seq = model.layers
    # Sequence pattern: [Lin, ReLU, Lin, ReLU, ..., Lin, ReLU, Lin_out]
    # That's DEPTH * 2 + 1 modules.
    expected_len = DEPTH * 2 + 1
    assert len(seq) == expected_len, f"unexpected layers length {len(seq)} != {expected_len}"
    with torch.no_grad():
        for i, m in enumerate(seq):
            h = m(h)
            if isinstance(m, nn.ReLU):
                activations.append(h.detach().clone())
    assert len(activations) == DEPTH, f"captured {len(activations)} activations, expected {DEPTH}"
    return activations


def eff_rank_from_Y(Y):
    """eff_rank = exp(-sum p_i log p_i) where p_i = lambda_i / sum(lambda_i),
    lambda_i = eigenvalues of Y^T Y / N (clipped to >=0)."""
    N = Y.shape[0]
    C = (Y.t() @ Y) / float(N)
    # eigvalsh on a symmetric PSD matrix
    eig = torch.linalg.eigvalsh(C)
    eig = torch.clamp(eig, min=0.0)
    s = eig.sum()
    if s.item() <= 0.0:
        return 1.0
    p = eig / s
    # nan-safe entropy: 0*log(0) = 0
    nz = p > 0
    H = -(p[nz] * torch.log(p[nz])).sum().item()
    return math.exp(H)


def train_until_acc(model, X, y, lr, max_epochs, target_acc):
    """Full-batch Adam BCEWithLogits. Return (final_train_acc, epochs_used, final_loss)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    final_loss = float("nan")
    final_acc = 0.0
    for epoch in range(1, max_epochs + 1):
        opt.zero_grad()
        logits = model(X).squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                pred = (model(X).squeeze(-1) > 0.0).float()
                acc = (pred == y).float().mean().item()
            final_acc = acc
            final_loss = loss.item()
            if acc >= target_acc:
                return acc, epoch, final_loss
    # final acc
    with torch.no_grad():
        pred = (model(X).squeeze(-1) > 0.0).float()
        acc = (pred == y).float().mean().item()
    final_loss = loss.item()
    return acc, max_epochs, final_loss


def census_for(dataset_name, seed, log_lines):
    t0 = time.time()
    X, y = make_dataset(dataset_name, seed)
    model = build_mlp(seed)
    n_params = trainable_param_count(model)
    # init activations
    init_acts = collect_post_relu(model, X)
    init_eff = [eff_rank_from_Y(Y) for Y in init_acts]
    # train
    final_acc, epochs, final_loss = train_until_acc(model, X, y, LR, MAX_EPOCHS, TRAIN_ACC_TARGET)
    converged = final_acc >= TRAIN_ACC_TARGET
    # post activations
    post_acts = collect_post_relu(model, X)
    post_eff = [eff_rank_from_Y(Y) for Y in post_acts]
    wall = time.time() - t0
    line = (
        f"  {dataset_name} seed={seed}: n_params={n_params} epochs={epochs} "
        f"train_acc={final_acc:.4f} loss={final_loss:.4f} converged={converged} wall={wall:.1f}s"
    )
    print(line, flush=True)
    log_lines.append(line)
    log_lines.append(
        f"    init_eff_rank={[round(x, 3) for x in init_eff]}"
    )
    log_lines.append(
        f"    post_eff_rank={[round(x, 3) for x in post_eff]}"
    )
    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_params": n_params,
        "epochs": epochs,
        "final_train_acc": final_acc,
        "final_train_loss": final_loss,
        "converged": converged,
        "init_eff_rank": init_eff,
        "post_eff_rank": post_eff,
        "wall_sec": wall,
    }


def aggregate_per_layer(records, dataset):
    """Mean per-layer eff_rank across seeds for a dataset (init and post)."""
    rs = [r for r in records if r["dataset"] == dataset]
    init = np.array([r["init_eff_rank"] for r in rs])  # (S, DEPTH)
    post = np.array([r["post_eff_rank"] for r in rs])
    init_mean = init.mean(axis=0).tolist()
    init_std = init.std(axis=0, ddof=0).tolist()
    post_mean = post.mean(axis=0).tolist()
    post_std = post.std(axis=0, ddof=0).tolist()
    return {
        "n_seeds": len(rs),
        "init_mean": init_mean,
        "init_std": init_std,
        "post_mean": post_mean,
        "post_std": post_std,
    }


def verdict_for_dataset(agg):
    """Apply item 305 thresholds to per-dataset post and init means."""
    post = agg["post_mean"]
    init = agg["init_mean"]
    n_post_le3 = sum(1 for x in post if x <= 3.0)
    n_init_ge20 = sum(1 for x in init if x >= 20.0)
    n_post_ge10 = sum(1 for x in post if x >= 10.0)
    confirmed = (n_post_le3 >= 8) and (n_init_ge20 >= 8)
    refuted = n_post_ge10 >= 6
    if confirmed:
        verdict = "CONFIRMED"
    elif refuted:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "n_post_le3": n_post_le3,
        "n_init_ge20": n_init_ge20,
        "n_post_ge10": n_post_ge10,
    }


def main():
    log_lines = []
    log_lines.append(f"Cycle 295 / item 305 — spirals/moons eff_rank census")
    log_lines.append(
        f"config: HIDDEN={HIDDEN} DEPTH={DEPTH} N={N_SAMPLES} LR={LR} max_epochs={MAX_EPOCHS} "
        f"target_acc={TRAIN_ACC_TARGET} seeds={SEEDS}"
    )
    print(log_lines[-2], flush=True)
    print(log_lines[-1], flush=True)
    t_start = time.time()
    records = []
    for dataset in ("spirals", "moons"):
        for seed in SEEDS:
            r = census_for(dataset, seed, log_lines)
            records.append(r)
    aggregates = {ds: aggregate_per_layer(records, ds) for ds in ("spirals", "moons")}
    verdicts = {ds: verdict_for_dataset(aggregates[ds]) for ds in ("spirals", "moons")}

    # Overall: CONFIRMED iff both CONFIRMED; REFUTED iff either REFUTED; else INCONCLUSIVE
    vs = [verdicts["spirals"]["verdict"], verdicts["moons"]["verdict"]]
    if all(v == "CONFIRMED" for v in vs):
        overall = "CONFIRMED"
    elif any(v == "REFUTED" for v in vs):
        overall = "REFUTED"
    else:
        overall = "INCONCLUSIVE"

    total_wall = time.time() - t_start
    summary = {
        "cycle": 295,
        "item": 305,
        "config": {
            "hidden": HIDDEN,
            "depth": DEPTH,
            "n_samples": N_SAMPLES,
            "lr": LR,
            "max_epochs": MAX_EPOCHS,
            "target_acc": TRAIN_ACC_TARGET,
            "seeds": SEEDS,
            "spirals_noise": SPIRALS_NOISE,
            "spirals_turns": SPIRALS_TURNS,
            "moons_noise": MOONS_NOISE,
        },
        "records": records,
        "aggregates": aggregates,
        "verdicts": verdicts,
        "verdict": overall,
        "total_wall_sec": total_wall,
        "baselines": {
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
    lines.append("# Cycle 295 — Spirals/Moons eff_rank census (item 305)")
    lines.append("")
    lines.append(f"**Verdict:** {overall}")
    lines.append("")
    lines.append("## Config")
    lines.append(
        f"- MLP(2, {HIDDEN}, 1, depth={DEPTH}, ReLU); Adam lr={LR}; full-batch BCEWithLogits"
    )
    lines.append(f"- N={N_SAMPLES}; spirals(noise={SPIRALS_NOISE}, n_turns={SPIRALS_TURNS}); moons(noise={MOONS_NOISE})")
    lines.append(f"- Stop when train_acc ≥ {TRAIN_ACC_TARGET} (cap {MAX_EPOCHS} epochs)")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- Trainable params/net = {records[0]['n_params']}")
    lines.append("")
    lines.append("## Verdict logic (item 305 verbatim)")
    lines.append("- CONFIRMED iff post eff_rank ≤ 3 in ≥8 of 12 layers AND init eff_rank ≥ 20 in ≥8 of 12 layers")
    lines.append("- REFUTED iff post eff_rank ≥ 10 in ≥6 layers")
    lines.append("- INCONCLUSIVE otherwise")
    lines.append("")
    for ds in ("spirals", "moons"):
        agg = aggregates[ds]
        v = verdicts[ds]
        lines.append(f"## {ds} (per-dataset verdict: {v['verdict']})")
        lines.append("")
        lines.append("| layer | init_mean ± std | post_mean ± std |")
        lines.append("|------:|----------------:|----------------:|")
        for l in range(DEPTH):
            lines.append(
                f"| {l} | {agg['init_mean'][l]:.3f} ± {agg['init_std'][l]:.3f} "
                f"| {agg['post_mean'][l]:.3f} ± {agg['post_std'][l]:.3f} |"
            )
        lines.append("")
        lines.append(
            f"- post layers with eff_rank ≤ 3: **{v['n_post_le3']}/12** "
            f"(need ≥8 for CONFIRMED)"
        )
        lines.append(
            f"- init layers with eff_rank ≥ 20: **{v['n_init_ge20']}/12** "
            f"(need ≥8 for CONFIRMED)"
        )
        lines.append(
            f"- post layers with eff_rank ≥ 10: **{v['n_post_ge10']}/12** "
            f"(REFUTED iff ≥6)"
        )
        lines.append("")
    lines.append("## Per-seed convergence")
    for r in records:
        lines.append(
            f"- {r['dataset']} seed={r['seed']}: epochs={r['epochs']}, "
            f"train_acc={r['final_train_acc']:.4f}, converged={r['converged']}"
        )
    lines.append("")
    lines.append(f"Total wall: {total_wall:.1f}s")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")

    log_lines.append("")
    log_lines.append(f"=== Aggregate per-dataset means ===")
    for ds in ("spirals", "moons"):
        agg = aggregates[ds]
        v = verdicts[ds]
        log_lines.append(f"  {ds}: verdict={v['verdict']}")
        log_lines.append(
            f"    init_mean={[round(x,3) for x in agg['init_mean']]}"
        )
        log_lines.append(
            f"    post_mean={[round(x,3) for x in agg['post_mean']]}"
        )
        log_lines.append(
            f"    n_post_le3={v['n_post_le3']}/12  n_init_ge20={v['n_init_ge20']}/12  "
            f"n_post_ge10={v['n_post_ge10']}/12"
        )
    log_lines.append(f"OVERALL VERDICT: {overall}")
    log_lines.append(f"total wall: {total_wall:.1f}s")
    LOG_PATH.write_text("\n".join(log_lines) + "\n")

    print()
    print(f"OVERALL VERDICT: {overall}")
    print(f"spirals: {verdicts['spirals']}")
    print(f"moons:   {verdicts['moons']}")
    print(f"total wall: {total_wall:.1f}s")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
