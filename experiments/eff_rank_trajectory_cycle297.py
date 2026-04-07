"""
Cycle 297 / item 307 — Effective rank trajectory through training on spirals.

For each seed in {42, 137, 2024}:
  - Build MLP(2 -> 32 -> 32 -> ... -> 32 -> 1, depth=12 hidden, ReLU)
  - Train Adam lr=1e-3 BCEWithLogitsLoss on spirals 1000 train / 1000 held
    (input_scale=2.0 per MEMORY.md depth-12 recipe)
  - Snapshot every 100 epochs (and at epoch 0): per-layer post-ReLU eff_rank,
    train loss, held-out accuracy
  - Report (i) init eff_rank per layer, (ii) final eff_rank per layer,
    (iii) max single-window collapse rate per layer (max drop within any
    100-epoch window), (iv) step at which each layer first reaches 1.5×
    its final value (left-undefined if monotone-non-decreasing)

Verdict (item 307 verbatim):
  CONFIRMED gradual         iff per-layer eff_rank curves are monotone-decreasing
                                across the entire training run AND no layer drops
                                by >= 50% within any 100-step window
  CONFIRMED phase-transition iff any layer drops by >= 50% within a 100-step window
  CONFIRMED init-rank        iff initial eff_rank is already <= 5 for >= 6 layers
  REFUTED                    iff eff_rank doesn't collapse at all
  Multiple CONFIRMED-* clauses can apply simultaneously (e.g. init-rank AND gradual).
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aleph.datasets import make_spirals
from aleph.models import MLP

# ---------- config ----------
SEEDS = [42, 137, 2024]
HIDDEN = 32
DEPTH = 12
N_SAMPLES = 2000  # 1000 train / 1000 held per seed
SPIRALS_NOISE = 0.5
SPIRALS_TURNS = 3
INPUT_SCALE = 2.0  # MEMORY.md depth-12 trainability recipe
LR = 1e-3
MAX_EPOCHS = 12000
SNAPSHOT_EVERY = 100  # epochs
TRAIN_ACC_TARGET = 0.95
EARLY_STOP_ACC = 0.99  # avoid runaway training past convergence

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "autoresearch" / "results"
LOGS_DIR = REPO / "autoresearch" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS_DIR / "cycle_297_eff_rank_trajectory.json"
SUMMARY_PATH = RESULTS_DIR / "cycle_297_summary.md"
PLOT_PATH = RESULTS_DIR / "cycle_297_trajectory.png"
LOG_PATH = LOGS_DIR / "cycle_297_run.log"


def make_dataset(seed):
    X, y = make_spirals(
        n_samples=N_SAMPLES, noise=SPIRALS_NOISE, n_turns=SPIRALS_TURNS, random_state=seed
    )
    X = X * INPUT_SCALE
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(np.asarray(y)).float()
    # 1000 train / 1000 held split, deterministic per seed
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N_SAMPLES, generator=g)
    train_idx = perm[: N_SAMPLES // 2]
    held_idx = perm[N_SAMPLES // 2 :]
    return Xt[train_idx], yt[train_idx], Xt[held_idx], yt[held_idx]


def build_mlp(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MLP(input_size=2, hidden_size=HIDDEN, output_size=1, depth=DEPTH, activation_fn=nn.ReLU)


def collect_post_relu(model, X):
    """Walk model.layers and capture activation AFTER each ReLU (DEPTH activations)."""
    seq = model.layers
    expected_len = DEPTH * 2 + 1
    assert len(seq) == expected_len, f"unexpected layers length {len(seq)} != {expected_len}"
    activations = []
    h = X
    with torch.no_grad():
        for m in seq:
            h = m(h)
            if isinstance(m, nn.ReLU):
                activations.append(h.detach().clone())
    assert len(activations) == DEPTH
    return activations


def eff_rank_from_Y(Y):
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


def snapshot(model, X_train, X_held, y_held, train_loss):
    """Compute per-layer eff_rank on training set + held-out accuracy."""
    acts = collect_post_relu(model, X_train)
    er = [eff_rank_from_Y(Y) for Y in acts]
    for v in er:
        assert 1.0 - 1e-6 <= v <= float(HIDDEN) + 1e-6, f"eff_rank out of range: {v}"
    with torch.no_grad():
        pred = (model(X_held).squeeze(-1) > 0.0).float()
        held_acc = (pred == y_held).float().mean().item()
        pred_t = (model(X_train).squeeze(-1) > 0.0).float()
    return er, held_acc, train_loss


def train_with_snapshots(model, X_train, y_train, X_held, y_held, log_lines):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    snaps = []
    # Snapshot at epoch 0 (init), with a dummy loss (eval before training)
    with torch.no_grad():
        init_logits = model(X_train).squeeze(-1)
        init_loss = loss_fn(init_logits, y_train).item()
        init_train_acc = ((init_logits > 0.0).float() == y_train).float().mean().item()
    er0, held0, _ = snapshot(model, X_train, X_held, y_held, init_loss)
    snaps.append({
        "epoch": 0,
        "train_loss": init_loss,
        "train_acc": init_train_acc,
        "held_acc": held0,
        "eff_rank": er0,
    })

    final_train_acc = init_train_acc
    final_epoch = 0
    converged = False
    for epoch in range(1, MAX_EPOCHS + 1):
        opt.zero_grad()
        logits = model(X_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
        if epoch % SNAPSHOT_EVERY == 0:
            with torch.no_grad():
                tr_logits = model(X_train).squeeze(-1)
                tr_acc = ((tr_logits > 0.0).float() == y_train).float().mean().item()
            er, held, _ = snapshot(model, X_train, X_held, y_held, loss.item())
            snaps.append({
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_acc": tr_acc,
                "held_acc": held,
                "eff_rank": er,
            })
            final_train_acc = tr_acc
            final_epoch = epoch
            if tr_acc >= TRAIN_ACC_TARGET and not converged:
                converged = True
                log_lines.append(f"    converged@epoch={epoch} train_acc={tr_acc:.4f}")
            if tr_acc >= EARLY_STOP_ACC:
                # ample post-convergence trajectory captured
                break
    return snaps, final_train_acc, final_epoch, converged


def trajectory_stats(snaps):
    """Per-layer init/final eff_rank, max 100-step drop, step at 1.5x final."""
    epochs = [s["epoch"] for s in snaps]
    er = np.array([s["eff_rank"] for s in snaps])  # (T, DEPTH)
    init = er[0].tolist()
    final = er[-1].tolist()
    # max single-window drop per layer (snap to next snap is one window of SNAPSHOT_EVERY epochs)
    drops = np.diff(er, axis=0)  # (T-1, DEPTH); negative = decrease
    max_drop = (-drops.min(axis=0)).tolist()  # most-negative diff per layer, sign flipped
    # max single-window relative drop (to detect 50%): max(drop / value_before)
    rel_drops = []
    for li in range(er.shape[1]):
        col = er[:, li]
        rd = 0.0
        for ti in range(1, col.shape[0]):
            before = col[ti - 1]
            if before > 1e-9:
                d = (before - col[ti]) / before
                if d > rd:
                    rd = float(d)
        rel_drops.append(rd)
    # epoch at which each layer first reaches 1.5x its final value
    # (only meaningful when init > final; otherwise None)
    epoch_at_1p5_final = []
    for li in range(er.shape[1]):
        col = er[:, li]
        target = 1.5 * final[li]
        if init[li] <= target:
            epoch_at_1p5_final.append(None)
            continue
        hit = None
        for ti in range(col.shape[0]):
            if col[ti] <= target:
                hit = epochs[ti]
                break
        epoch_at_1p5_final.append(hit)
    # monotone-decreasing across the run? (allow tiny float noise)
    monotone = bool(np.all(drops <= 1e-6))
    return {
        "init": init,
        "final": final,
        "max_window_drop_abs": max_drop,
        "max_window_rel_drop": rel_drops,
        "epoch_at_1p5_final": epoch_at_1p5_final,
        "monotone_nonincreasing": monotone,
    }


def per_seed_verdict(stats):
    """Apply item 307 verbatim verdict logic to a single seed's trajectory."""
    init = stats["init"]
    rel_drops = stats["max_window_rel_drop"]
    monotone = stats["monotone_nonincreasing"]
    n_init_le5 = sum(1 for x in init if x <= 5.0)
    any_50pct_drop = any(d >= 0.5 for d in rel_drops)
    confirmed_init_rank = n_init_le5 >= 6
    confirmed_phase = any_50pct_drop
    confirmed_gradual = monotone and not any_50pct_drop
    # REFUTED if eff_rank doesn't collapse at all (init essentially equals final)
    deltas = [init[i] - stats["final"][i] for i in range(len(init))]
    refuted = all(abs(d) <= 0.1 for d in deltas)
    flags = []
    if confirmed_init_rank:
        flags.append("CONFIRMED-init-rank")
    if confirmed_phase:
        flags.append("CONFIRMED-phase")
    if confirmed_gradual:
        flags.append("CONFIRMED-gradual")
    if refuted:
        flags.append("REFUTED")
    if not flags:
        flags.append("INCONCLUSIVE")
    return {
        "n_init_le5": n_init_le5,
        "any_50pct_drop": any_50pct_drop,
        "monotone_nonincreasing": monotone,
        "max_rel_drop": float(max(rel_drops)),
        "refuted_no_collapse": refuted,
        "flags": flags,
    }


def overall_verdict(per_seed):
    """Combine per-seed verdicts. Honor multi-clause: report flags that hold for >=2 of 3 seeds."""
    from collections import Counter
    flag_counts = Counter()
    for v in per_seed.values():
        for f in v["flags"]:
            flag_counts[f] += 1
    final_flags = [f for f, c in flag_counts.items() if c >= 2]
    if not final_flags:
        return "INCONCLUSIVE", flag_counts
    # Drop INCONCLUSIVE if real CONFIRMED/REFUTED clauses also hold
    real = [f for f in final_flags if f != "INCONCLUSIVE"]
    if real:
        return "+".join(sorted(real)), flag_counts
    return "INCONCLUSIVE", flag_counts


def maybe_plot(records):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return False, f"matplotlib unavailable: {e}"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # Top-left: eff_rank trajectories per layer (mean across seeds), one line per layer
    ax = axes[0, 0]
    # gather aligned-epoch arrays
    common_epochs = sorted(set.intersection(*[set(s["epoch"] for s in r["snapshots"]) for r in records]))
    arr_per_seed = []
    for r in records:
        ep_to_er = {s["epoch"]: s["eff_rank"] for s in r["snapshots"]}
        arr_per_seed.append(np.array([ep_to_er[e] for e in common_epochs]))  # (T, DEPTH)
    arr = np.stack(arr_per_seed, axis=0)  # (S, T, DEPTH)
    mean = arr.mean(axis=0)  # (T, DEPTH)
    cmap = plt.get_cmap("viridis")
    for li in range(DEPTH):
        ax.plot(common_epochs, mean[:, li], color=cmap(li / max(1, DEPTH - 1)), label=f"L{li}", linewidth=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("eff_rank")
    ax.set_title("Per-layer eff_rank (mean across 3 seeds)")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    # Top-right: train loss per seed
    ax = axes[0, 1]
    for r in records:
        ax.plot([s["epoch"] for s in r["snapshots"]], [s["train_loss"] for s in r["snapshots"]], label=f"seed={r['seed']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train BCE loss")
    ax.set_title("Train loss per seed")
    ax.legend()
    ax.grid(alpha=0.3)
    # Bottom-left: held-out acc per seed
    ax = axes[1, 0]
    for r in records:
        ax.plot([s["epoch"] for s in r["snapshots"]], [s["held_acc"] for s in r["snapshots"]], label=f"seed={r['seed']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out accuracy")
    ax.set_title("Held-out accuracy per seed")
    ax.legend()
    ax.grid(alpha=0.3)
    # Bottom-right: per-layer eff_rank for seed 42 only (full curves)
    ax = axes[1, 1]
    r0 = records[0]
    ep0 = [s["epoch"] for s in r0["snapshots"]]
    er0 = np.array([s["eff_rank"] for s in r0["snapshots"]])
    for li in range(DEPTH):
        ax.plot(ep0, er0[:, li], color=cmap(li / max(1, DEPTH - 1)), linewidth=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("eff_rank")
    ax.set_title(f"Per-layer eff_rank (seed={r0['seed']})")
    ax.grid(alpha=0.3)
    fig.suptitle("Cycle 297 / item 307 — eff_rank trajectory through spirals training")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)
    return True, str(PLOT_PATH)


def main():
    log_lines = []
    log_lines.append("Cycle 297 / item 307 — eff_rank trajectory through spirals training")
    log_lines.append(
        f"config: HIDDEN={HIDDEN} DEPTH={DEPTH} N={N_SAMPLES} LR={LR} MAX_EPOCHS={MAX_EPOCHS} "
        f"SNAPSHOT_EVERY={SNAPSHOT_EVERY} INPUT_SCALE={INPUT_SCALE} EARLY_STOP_ACC={EARLY_STOP_ACC} seeds={SEEDS}"
    )
    print(log_lines[-2], flush=True)
    print(log_lines[-1], flush=True)

    t_start = time.time()
    records = []
    per_seed_stats = {}
    per_seed_v = {}
    for seed in SEEDS:
        t0 = time.time()
        X_tr, y_tr, X_he, y_he = make_dataset(seed)
        model = build_mlp(seed)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params == 11745, f"unexpected n_params {n_params}"
        log_lines.append(f"  seed={seed} n_params={n_params}")
        snaps, final_acc, final_epoch, converged = train_with_snapshots(
            model, X_tr, y_tr, X_he, y_he, log_lines
        )
        wall = time.time() - t0
        log_lines.append(
            f"    seed={seed} final_epoch={final_epoch} train_acc={final_acc:.4f} "
            f"converged={converged} n_snapshots={len(snaps)} wall={wall:.1f}s"
        )
        print(log_lines[-1], flush=True)
        stats = trajectory_stats(snaps)
        verdict = per_seed_verdict(stats)
        per_seed_stats[seed] = stats
        per_seed_v[seed] = verdict
        records.append({
            "seed": seed,
            "n_params": n_params,
            "final_epoch": final_epoch,
            "final_train_acc": final_acc,
            "converged": converged,
            "wall_sec": wall,
            "snapshots": snaps,
            "stats": stats,
            "verdict": verdict,
        })
        log_lines.append(
            f"    init eff_rank: {[round(x,3) for x in stats['init']]}"
        )
        log_lines.append(
            f"    final eff_rank: {[round(x,3) for x in stats['final']]}"
        )
        log_lines.append(
            f"    max_window_rel_drop: {[round(x,3) for x in stats['max_window_rel_drop']]}"
        )
        log_lines.append(
            f"    epoch_at_1.5x_final: {stats['epoch_at_1p5_final']}"
        )
        log_lines.append(
            f"    flags: {verdict['flags']}"
        )

    overall, flag_counts = overall_verdict(per_seed_v)
    total_wall = time.time() - t_start

    plot_ok, plot_msg = maybe_plot(records)

    summary = {
        "cycle": 297,
        "item": 307,
        "config": {
            "hidden": HIDDEN,
            "depth": DEPTH,
            "n_samples": N_SAMPLES,
            "lr": LR,
            "max_epochs": MAX_EPOCHS,
            "snapshot_every": SNAPSHOT_EVERY,
            "train_acc_target": TRAIN_ACC_TARGET,
            "early_stop_acc": EARLY_STOP_ACC,
            "input_scale": INPUT_SCALE,
            "spirals_noise": SPIRALS_NOISE,
            "spirals_turns": SPIRALS_TURNS,
            "seeds": SEEDS,
        },
        "records": records,
        "per_seed_verdicts": per_seed_v,
        "overall_verdict": overall,
        "flag_counts": dict(flag_counts),
        "total_wall_sec": total_wall,
        "plot": plot_msg,
        "baselines": {
            "cycle_295_spirals_init_mean": [2.994, 2.705, 1.874, 1.187, 1.058, 1.010, 1.002, 1.001, 1.000, 1.000, 1.000, 1.000],
            "cycle_295_spirals_post_mean": [3.143, 2.859, 2.106, 2.088, 2.151, 1.937, 1.986, 2.072, 2.049, 2.032, 1.756, 1.736],
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    # Summary markdown
    lines = []
    lines.append("# Cycle 297 — eff_rank trajectory on spirals (item 307)")
    lines.append("")
    lines.append(f"**Verdict:** {overall}")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- MLP(2, {HIDDEN}, 1, depth={DEPTH}, ReLU); Adam lr={LR}; full-batch BCEWithLogits")
    lines.append(f"- Spirals N={N_SAMPLES} (1000 train / 1000 held), noise={SPIRALS_NOISE}, n_turns={SPIRALS_TURNS}, input_scale={INPUT_SCALE}")
    lines.append(f"- Snapshot every {SNAPSHOT_EVERY} epochs; max {MAX_EPOCHS}; early stop at train_acc >= {EARLY_STOP_ACC}")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- Trainable params/net = {records[0]['n_params']}")
    lines.append("")
    lines.append("## Verdict logic (item 307 verbatim)")
    lines.append("- CONFIRMED gradual: per-layer eff_rank monotone-decreasing AND no >=50% drop in any 100-step window")
    lines.append("- CONFIRMED phase-transition: any layer drops by >=50% within a 100-step window")
    lines.append("- CONFIRMED init-rank: initial eff_rank <= 5 for >= 6 layers")
    lines.append("- REFUTED: eff_rank doesn't collapse at all")
    lines.append("- Multiple CONFIRMED-* clauses can apply simultaneously.")
    lines.append("")
    lines.append("## Per-seed results")
    lines.append("")
    for seed in SEEDS:
        r = next(x for x in records if x["seed"] == seed)
        v = per_seed_v[seed]
        s = per_seed_stats[seed]
        lines.append(f"### seed={seed}")
        lines.append(f"- final_epoch={r['final_epoch']}  train_acc={r['final_train_acc']:.4f}  converged={r['converged']}  snapshots={len(r['snapshots'])}")
        lines.append(f"- init eff_rank: {[round(x,3) for x in s['init']]}")
        lines.append(f"- final eff_rank: {[round(x,3) for x in s['final']]}")
        lines.append(f"- max single-window relative drop per layer: {[round(x,3) for x in s['max_window_rel_drop']]}")
        lines.append(f"- monotone non-increasing: {s['monotone_nonincreasing']}")
        lines.append(f"- n_init_le5: {v['n_init_le5']}/12  (CONFIRMED-init-rank iff >=6)")
        lines.append(f"- any 50% drop: {v['any_50pct_drop']}  (CONFIRMED-phase iff true)")
        lines.append(f"- max relative drop: {v['max_rel_drop']:.4f}")
        lines.append(f"- epoch at 1.5x final per layer: {s['epoch_at_1p5_final']}")
        lines.append(f"- flags: {v['flags']}")
        lines.append("")
    lines.append("## Overall verdict")
    lines.append(f"- {overall} (flag counts across 3 seeds: {dict(flag_counts)})")
    lines.append("")
    lines.append(f"Total wall: {total_wall:.1f}s")
    if plot_ok:
        lines.append(f"Plot: {plot_msg}")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")

    log_lines.append("")
    log_lines.append(f"=== Per-seed verdicts ===")
    for seed in SEEDS:
        log_lines.append(f"  seed={seed}: {per_seed_v[seed]['flags']}")
    log_lines.append(f"OVERALL: {overall}  flag_counts={dict(flag_counts)}")
    log_lines.append(f"total wall: {total_wall:.1f}s")
    LOG_PATH.write_text("\n".join(log_lines) + "\n")

    print()
    print(f"OVERALL VERDICT: {overall}")
    for seed in SEEDS:
        print(f"  seed={seed}: {per_seed_v[seed]['flags']}")
    print(f"flag_counts: {dict(flag_counts)}")
    print(f"total wall: {total_wall:.1f}s")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {LOG_PATH}")
    if plot_ok:
        print(f"wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
