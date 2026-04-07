"""
Cycle 296 / item 306 — LASER recipe on spirals/moons (cycle 280 replication on
a provably-2D task).

For each dataset {spirals, moons} × seed {42, 137, 2024}:
  1. Build MLP(2 -> 32 -> ... -> 32 -> 1, depth=12 hidden, ReLU).
  2. Random 50/50 split (1000 train / 1000 held-out) on N=2000.
  3. Train Adam lr=1e-3 BCEWithLogitsLoss full-batch on TRAIN until train_acc >= 0.95
     (cycle 295 recipe; spirals input ×2.0 per MEMORY.md depth-12 trainability).
  4. Identify l* = argmin_{l in 2..11} eff_rank(post-ReLU activations on TRAIN).
     (Skip l=0 because W is (32, 2) — first hidden Linear isn't a 32x32 SVD case.
      Skip l=1 conservatively; layers 2-11 all have hidden_size=32.)
  5. For r in {1, 2, 3, 5}:
     a) NO-RETRAIN: write W_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :] into the layer
        weight; eval HELD-OUT acc; restore W.
     b) RETRAIN: replace model.layers[2*l*] with LowRankLinear(A_frozen, B_init,
        bias_init). Freeze every other parameter. Adam lr=1e-3 on (B, bias) for
        500 full-batch BCEWithLogits steps on TRAIN. Eval HELD-OUT every 25 steps;
        track best held-out acc; restore best (B, bias). Restore the original
        Linear afterward.

Verdict (item 306 verbatim):
  CONFIRMED iff rank-2 retrain held-out acc >= float held-out acc - 0.5% on spirals
              AND no-retrain rank-2 >= 80% of float held-out acc.
  REFUTED   iff rank-2 retrain held-out gap > 5% from float on spirals.
  INCONCLUSIVE otherwise.
Apply to spirals; report moons in same format for context.
"""
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

from aleph.datasets import make_spirals
from aleph.models import MLP

# ---------- config ----------
SEEDS = [42, 137, 2024]
HIDDEN = 32
DEPTH = 12  # number of hidden Linear+ReLU layers
N_SAMPLES = 2000
SPIRALS_NOISE = 0.5
SPIRALS_TURNS = 3
SPIRALS_INPUT_SCALE = 2.0
MOONS_NOISE = 0.1
LR_TRAIN = 1e-3
LR_RETRAIN = 1e-3
MAX_EPOCHS = 20000
TRAIN_ACC_TARGET = 0.95
RETRAIN_STEPS = 500
EVAL_EVERY = 25
RANKS = [1, 2, 3, 5]

# Verdict thresholds (item 306 verbatim)
CONFIRMED_RETRAIN_GAP = 0.005   # rank-2 retrain >= float - 0.5%
CONFIRMED_NORETRAIN_FRAC = 0.80 # no-retrain rank-2 >= 80% of float
REFUTED_RETRAIN_GAP = 0.05      # rank-2 retrain gap > 5%

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "autoresearch" / "results"
LOGS_DIR = REPO / "autoresearch" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS_DIR / "cycle_296_spirals_moons_laser.json"
SUMMARY_PATH = RESULTS_DIR / "cycle_296_summary.md"
LOG_PATH = LOGS_DIR / "cycle_296_run.log"


# ---------- utilities ----------
def make_dataset(name, seed):
    if name == "spirals":
        X, y = make_spirals(
            n_samples=N_SAMPLES, noise=SPIRALS_NOISE, n_turns=SPIRALS_TURNS, random_state=seed
        )
        X = X * SPIRALS_INPUT_SCALE
    elif name == "moons":
        X, y = make_moons(n_samples=N_SAMPLES, noise=MOONS_NOISE, random_state=seed)
    else:
        raise ValueError(name)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(np.asarray(y)).float()
    return Xt, yt


def split_train_holdout(X, y, seed):
    """50/50 random split, deterministic per seed."""
    g = torch.Generator().manual_seed(seed + 1)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    half = n // 2
    train_idx = perm[:half]
    held_idx = perm[half:]
    return X[train_idx], y[train_idx], X[held_idx], y[held_idx]


def build_mlp(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MLP(input_size=2, hidden_size=HIDDEN, output_size=1, depth=DEPTH, activation_fn=nn.ReLU)


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_acc(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = (model(X).squeeze(-1) > 0.0).float()
        return (pred == y).float().mean().item()


def train_until_acc(model, X, y, lr, max_epochs, target_acc):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    final_acc = 0.0
    final_loss = float("nan")
    for epoch in range(1, max_epochs + 1):
        model.train()
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
    with torch.no_grad():
        pred = (model(X).squeeze(-1) > 0.0).float()
        final_acc = (pred == y).float().mean().item()
    final_loss = loss.item()
    return final_acc, max_epochs, final_loss


def collect_post_relu_train(model, X):
    """Collect post-ReLU activations on TRAIN inputs for layers l in {0..DEPTH-1}."""
    activations = []
    h = X
    seq = model.layers
    expected = DEPTH * 2 + 1
    assert len(seq) == expected, f"unexpected layers length {len(seq)} != {expected}"
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


class LowRankLinear(nn.Module):
    """Replacement for nn.Linear at rank r.

    Frozen A = U[:, :r] * S[:r] in R^(out, r). Trainable B in R^(r, in)
    initialised from Vh[:r, :]. Trainable bias.
    Forward: x @ (A @ B)^T + bias = x @ B^T @ A^T + bias.
    For nn.Linear semantics, weight has shape (out, in) and forward is
    F.linear(x, W, b) = x @ W.T + b.
    """

    def __init__(self, A_frozen, B_init, b_init):
        super().__init__()
        # A: (out, r); B: (r, in)
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())
        self.bias = nn.Parameter(b_init.clone().contiguous())
        self.out_features = A_frozen.shape[0]
        self.in_features = B_init.shape[1]

    def forward(self, x):
        W = self.A @ self.B  # (out, in)
        return torch.nn.functional.linear(x, W, self.bias)


def laser_for_layer(model, layer_seq_idx, X_tr, y_tr, X_te, y_te, ranks, log_lines, dataset, seed):
    """For each rank r: no-retrain held-out acc, then retrain held-out best acc."""
    orig_lin = model.layers[layer_seq_idx]
    assert isinstance(orig_lin, nn.Linear)
    W_orig = orig_lin.weight.data.detach().clone()
    b_orig = orig_lin.bias.data.detach().clone()
    out_dim, in_dim = W_orig.shape
    assert (out_dim, in_dim) == (HIDDEN, HIDDEN), f"layer weight shape {(out_dim,in_dim)} != ({HIDDEN},{HIDDEN})"

    # Float (uncompressed) held-out acc baseline (with original weights restored)
    with torch.no_grad():
        orig_lin.weight.data.copy_(W_orig)
        orig_lin.bias.data.copy_(b_orig)
    float_held_acc = eval_acc(model, X_te, y_te)

    # SVD on the original weight
    U, S, Vh = torch.linalg.svd(W_orig.float(), full_matrices=False)
    s_top5 = S[:5].tolist()
    log_lines.append(f"    {dataset} seed={seed} layer_seq_idx={layer_seq_idx} W shape={tuple(W_orig.shape)}")
    log_lines.append(f"      SVD top-5: {[round(x,4) for x in s_top5]}")
    log_lines.append(f"      float held-out acc = {float_held_acc:.4f}")

    rank_results = {}
    for r in ranks:
        # NO-RETRAIN
        A_r = (U[:, :r] * S[:r])  # (32, r)
        B_r = Vh[:r, :]            # (r, 32)
        W_r = (A_r @ B_r).to(dtype=W_orig.dtype)
        with torch.no_grad():
            orig_lin.weight.data.copy_(W_r)
            orig_lin.bias.data.copy_(b_orig)
        nor_train_acc = eval_acc(model, X_tr, y_tr)
        nor_held_acc = eval_acc(model, X_te, y_te)

        # restore original to keep state clean for retrain step which inserts a fresh module
        with torch.no_grad():
            orig_lin.weight.data.copy_(W_orig)
            orig_lin.bias.data.copy_(b_orig)

        # RETRAIN: swap in LowRankLinear
        lr_lin = LowRankLinear(
            A_frozen=A_r.detach().clone(),
            B_init=B_r.detach().clone(),
            b_init=b_orig.detach().clone(),
        )
        model.layers[layer_seq_idx] = lr_lin

        # Freeze everything except B and bias
        for p in model.parameters():
            p.requires_grad_(False)
        lr_lin.B.requires_grad_(True)
        lr_lin.bias.requires_grad_(True)
        n_trainable = sum(p.numel() for p in [lr_lin.B, lr_lin.bias])
        expected_trainable = r * HIDDEN + HIDDEN
        assert n_trainable == expected_trainable, f"trainable {n_trainable} != {expected_trainable}"

        opt = torch.optim.Adam([lr_lin.B, lr_lin.bias], lr=LR_RETRAIN)
        loss_fn = nn.BCEWithLogitsLoss()

        # initial held-out (should equal nor_held_acc within float noise)
        init_held = eval_acc(model, X_te, y_te)
        assert abs(init_held - nor_held_acc) < 1e-6, f"init held {init_held} != noretrain {nor_held_acc}"
        best_held = init_held
        best_step = 0
        best_state = {
            "B": lr_lin.B.detach().clone(),
            "bias": lr_lin.bias.detach().clone(),
        }
        history = [{"step": 0, "held_acc": float(init_held)}]

        for step in range(1, RETRAIN_STEPS + 1):
            model.train()
            opt.zero_grad()
            logits = model(X_tr).squeeze(-1)
            loss = loss_fn(logits, y_tr)
            loss.backward()
            opt.step()
            if step % EVAL_EVERY == 0:
                held = eval_acc(model, X_te, y_te)
                history.append({"step": step, "held_acc": float(held), "loss": float(loss.item())})
                if held > best_held:
                    best_held = held
                    best_step = step
                    best_state["B"] = lr_lin.B.detach().clone()
                    best_state["bias"] = lr_lin.bias.detach().clone()

        # restore best
        with torch.no_grad():
            lr_lin.B.data.copy_(best_state["B"])
            lr_lin.bias.data.copy_(best_state["bias"])
        restored = eval_acc(model, X_te, y_te)
        assert abs(restored - best_held) < 1e-9

        # restore original Linear in place
        model.layers[layer_seq_idx] = orig_lin
        with torch.no_grad():
            orig_lin.weight.data.copy_(W_orig)
            orig_lin.bias.data.copy_(b_orig)
        # sanity: float held-out acc reproduced
        post_restore_held = eval_acc(model, X_te, y_te)
        assert abs(post_restore_held - float_held_acc) < 1e-9, (
            f"post-restore held {post_restore_held} != float {float_held_acc}"
        )

        rank_results[r] = {
            "no_retrain_train_acc": float(nor_train_acc),
            "no_retrain_held_acc": float(nor_held_acc),
            "retrain_init_held_acc": float(init_held),
            "retrain_best_held_acc": float(best_held),
            "retrain_best_step": int(best_step),
            "retrain_n_trainable": int(n_trainable),
            "history": history,
        }
        log_lines.append(
            f"      r={r}: noretrain held={nor_held_acc:.4f}  "
            f"retrain best={best_held:.4f}@{best_step}  init_held={init_held:.4f}"
        )

    return float_held_acc, rank_results, [round(x,4) for x in s_top5]


def laser_for(dataset_name, seed, log_lines):
    t0 = time.time()
    X, y = make_dataset(dataset_name, seed)
    X_tr, y_tr, X_te, y_te = split_train_holdout(X, y, seed)
    model = build_mlp(seed)
    n_params = trainable_param_count(model)
    final_acc, epochs, final_loss = train_until_acc(model, X_tr, y_tr, LR_TRAIN, MAX_EPOCHS, TRAIN_ACC_TARGET)
    converged = final_acc >= TRAIN_ACC_TARGET
    if not converged:
        log_lines.append(f"  {dataset_name} seed={seed}: FAILED to converge train_acc={final_acc:.4f}")
        return {
            "dataset": dataset_name, "seed": seed, "n_params": n_params,
            "epochs": epochs, "final_train_acc": final_acc, "converged": False,
        }

    # Effective rank per layer on TRAIN
    train_acts = collect_post_relu_train(model, X_tr)
    train_eff = [eff_rank_from_Y(Y) for Y in train_acts]

    # Pick l* = argmin over l in {2..11} (32x32 layers; skip l=0 (32x2) and l=1 conservatively)
    candidate_layers = list(range(2, DEPTH))
    l_star = min(candidate_layers, key=lambda l: train_eff[l])
    layer_seq_idx = 2 * l_star  # Linear at this index produces post-ReLU at index 2*l_star + 1

    # Float held-out acc baseline (after training, before any LASER)
    float_train_acc = eval_acc(model, X_tr, y_tr)
    float_held_acc = eval_acc(model, X_te, y_te)
    log_lines.append(
        f"  {dataset_name} seed={seed}: epochs={epochs} train_acc={final_acc:.4f} "
        f"float_train={float_train_acc:.4f} float_held={float_held_acc:.4f} "
        f"l*={l_star} eff_rank[l*]={train_eff[l_star]:.3f}"
    )
    log_lines.append(f"    train_eff_rank={[round(x,3) for x in train_eff]}")

    held_baseline, rank_results, svd_top5 = laser_for_layer(
        model, layer_seq_idx, X_tr, y_tr, X_te, y_te, RANKS, log_lines, dataset_name, seed
    )
    assert abs(held_baseline - float_held_acc) < 1e-9, (
        f"held_baseline {held_baseline} != float_held_acc {float_held_acc}"
    )

    wall = time.time() - t0
    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_params": n_params,
        "epochs": epochs,
        "final_train_acc": float(final_acc),
        "converged": True,
        "float_train_acc": float(float_train_acc),
        "float_held_acc": float(float_held_acc),
        "train_eff_rank": train_eff,
        "l_star": int(l_star),
        "layer_seq_idx": int(layer_seq_idx),
        "svd_top5": svd_top5,
        "ranks": {str(r): rank_results[r] for r in RANKS},
        "wall_sec": float(wall),
    }


def aggregate_per_dataset(records, dataset):
    """Mean across seeds: float held, no-retrain held per r, retrain best per r."""
    rs = [r for r in records if r["dataset"] == dataset and r.get("converged", False)]
    if not rs:
        return None
    float_held_mean = float(np.mean([r["float_held_acc"] for r in rs]))
    float_held_std = float(np.std([r["float_held_acc"] for r in rs], ddof=0))
    per_r = {}
    for r in RANKS:
        no_retr = [rec["ranks"][str(r)]["no_retrain_held_acc"] for rec in rs]
        retr = [rec["ranks"][str(r)]["retrain_best_held_acc"] for rec in rs]
        per_r[str(r)] = {
            "no_retrain_mean": float(np.mean(no_retr)),
            "no_retrain_std": float(np.std(no_retr, ddof=0)),
            "retrain_mean": float(np.mean(retr)),
            "retrain_std": float(np.std(retr, ddof=0)),
        }
    return {
        "n_seeds": len(rs),
        "float_held_mean": float_held_mean,
        "float_held_std": float_held_std,
        "per_rank": per_r,
    }


def verdict_for_spirals(agg):
    """Item 306 verbatim verdict applied to spirals aggregate."""
    if agg is None:
        return {"verdict": "INCONCLUSIVE", "reason": "no converged seeds"}
    f = agg["float_held_mean"]
    r2_retrain = agg["per_rank"]["2"]["retrain_mean"]
    r2_noretrain = agg["per_rank"]["2"]["no_retrain_mean"]
    retrain_gap = f - r2_retrain  # positive means retrain below float
    noretrain_frac = r2_noretrain / f if f > 0 else 0.0
    confirmed = (retrain_gap <= CONFIRMED_RETRAIN_GAP) and (noretrain_frac >= CONFIRMED_NORETRAIN_FRAC)
    refuted = retrain_gap > REFUTED_RETRAIN_GAP
    if confirmed:
        v = "CONFIRMED"
    elif refuted:
        v = "REFUTED"
    else:
        v = "INCONCLUSIVE"
    return {
        "verdict": v,
        "float_held_mean": f,
        "rank2_noretrain_mean": r2_noretrain,
        "rank2_retrain_mean": r2_retrain,
        "rank2_retrain_gap": retrain_gap,
        "rank2_noretrain_frac_of_float": noretrain_frac,
        "confirmed_retrain_gap_threshold": CONFIRMED_RETRAIN_GAP,
        "confirmed_noretrain_frac_threshold": CONFIRMED_NORETRAIN_FRAC,
        "refuted_retrain_gap_threshold": REFUTED_RETRAIN_GAP,
    }


def main():
    log_lines = []
    log_lines.append("Cycle 296 / item 306 — LASER recipe on spirals/moons")
    log_lines.append(
        f"config: HIDDEN={HIDDEN} DEPTH={DEPTH} N={N_SAMPLES} LR_TRAIN={LR_TRAIN} "
        f"LR_RETRAIN={LR_RETRAIN} max_epochs={MAX_EPOCHS} target_acc={TRAIN_ACC_TARGET} "
        f"retrain_steps={RETRAIN_STEPS} eval_every={EVAL_EVERY} ranks={RANKS} seeds={SEEDS}"
    )
    print(log_lines[-2], flush=True)
    print(log_lines[-1], flush=True)
    t_start = time.time()
    records = []
    for dataset in ("spirals", "moons"):
        for seed in SEEDS:
            r = laser_for(dataset, seed, log_lines)
            records.append(r)
            print(log_lines[-1], flush=True)
    aggregates = {ds: aggregate_per_dataset(records, ds) for ds in ("spirals", "moons")}
    spirals_v = verdict_for_spirals(aggregates["spirals"])
    moons_v = verdict_for_spirals(aggregates["moons"])  # context
    overall = spirals_v["verdict"]
    total_wall = time.time() - t_start

    summary = {
        "cycle": 296,
        "item": 306,
        "config": {
            "hidden": HIDDEN, "depth": DEPTH, "n_samples": N_SAMPLES,
            "split": "1000 train / 1000 held-out per seed",
            "lr_train": LR_TRAIN, "lr_retrain": LR_RETRAIN,
            "max_epochs": MAX_EPOCHS, "target_acc": TRAIN_ACC_TARGET,
            "retrain_steps": RETRAIN_STEPS, "eval_every": EVAL_EVERY,
            "ranks": RANKS, "seeds": SEEDS,
            "spirals_noise": SPIRALS_NOISE, "spirals_turns": SPIRALS_TURNS,
            "spirals_input_scale": SPIRALS_INPUT_SCALE, "moons_noise": MOONS_NOISE,
            "layer_selection": "argmin eff_rank over l in [2..11]",
        },
        "records": records,
        "aggregates": aggregates,
        "verdict_spirals": spirals_v,
        "verdict_moons_context": moons_v,
        "verdict": overall,
        "total_wall_sec": total_wall,
        "baselines": {
            "cycle_280_block1_rank2_retrain_ppl": 49.03,
            "cycle_280_float_ppl": 53.895,
            "cycle_292_block1_rank1_retrain_ppl": 50.08,
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    # Markdown summary
    lines = []
    lines.append("# Cycle 296 — LASER recipe on spirals/moons (item 306)")
    lines.append("")
    lines.append(f"**Verdict (spirals): {overall}**")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- MLP(2, {HIDDEN}, 1, depth={DEPTH}, ReLU); Adam train lr={LR_TRAIN}; full-batch BCEWithLogits")
    lines.append(f"- N={N_SAMPLES} (1000 train / 1000 held-out per seed)")
    lines.append(f"- spirals(noise={SPIRALS_NOISE}, n_turns={SPIRALS_TURNS}, input_scale={SPIRALS_INPUT_SCALE}); moons(noise={MOONS_NOISE})")
    lines.append(f"- Stop training when train_acc >= {TRAIN_ACC_TARGET} (cap {MAX_EPOCHS} epochs)")
    lines.append(f"- LASER: select l* = argmin eff_rank over l in [2..11]; SVD W (32x32) at l*; ranks {RANKS}")
    lines.append(f"- Retrain: Adam lr={LR_RETRAIN}, {RETRAIN_STEPS} full-batch steps on TRAIN, eval HELD-OUT every {EVAL_EVERY}, best-state restore")
    lines.append(f"- Trainable per fit: 32(r+1); seeds {SEEDS}")
    lines.append("")
    lines.append("## Verdict logic (item 306 verbatim, applied to spirals)")
    lines.append("- CONFIRMED iff rank-2 retrain held-out acc >= float held-out acc - 0.5% on spirals AND no-retrain rank-2 >= 80% of float held-out acc")
    lines.append("- REFUTED iff rank-2 retrain held-out gap > 5% from float on spirals")
    lines.append("- INCONCLUSIVE between")
    lines.append("")
    for ds in ("spirals", "moons"):
        agg = aggregates[ds]
        v = spirals_v if ds == "spirals" else moons_v
        lines.append(f"## {ds} (verdict: {v['verdict']})")
        lines.append("")
        if agg is None:
            lines.append("(no converged seeds)")
            lines.append("")
            continue
        lines.append(f"- float held-out acc (mean ± std): **{agg['float_held_mean']:.4f} ± {agg['float_held_std']:.4f}**")
        lines.append("")
        lines.append("| r | no_retrain_mean | no_retrain_std | retrain_mean | retrain_std | gap_vs_float |")
        lines.append("|--:|----------------:|---------------:|-------------:|------------:|-------------:|")
        for r in RANKS:
            pr = agg["per_rank"][str(r)]
            gap = agg["float_held_mean"] - pr["retrain_mean"]
            lines.append(
                f"| {r} | {pr['no_retrain_mean']:.4f} | {pr['no_retrain_std']:.4f} "
                f"| {pr['retrain_mean']:.4f} | {pr['retrain_std']:.4f} | {gap:+.4f} |"
            )
        lines.append("")
        lines.append(f"- rank-2 retrain gap vs float: **{agg['float_held_mean'] - agg['per_rank']['2']['retrain_mean']:+.4f}**")
        lines.append(f"- rank-2 no-retrain frac of float: **{agg['per_rank']['2']['no_retrain_mean']/agg['float_held_mean']:.4f}**")
        lines.append("")
    lines.append("## Per-seed details")
    for rec in records:
        if not rec.get("converged", False):
            lines.append(f"- {rec['dataset']} seed={rec['seed']}: FAILED to converge")
            continue
        lines.append(
            f"- {rec['dataset']} seed={rec['seed']}: epochs={rec['epochs']} train_acc={rec['final_train_acc']:.4f} "
            f"float_held={rec['float_held_acc']:.4f} l*={rec['l_star']} eff_rank[l*]={rec['train_eff_rank'][rec['l_star']]:.3f}"
        )
        for r in RANKS:
            rr = rec["ranks"][str(r)]
            lines.append(
                f"  - r={r}: no_retrain={rr['no_retrain_held_acc']:.4f}  "
                f"retrain_best={rr['retrain_best_held_acc']:.4f}@{rr['retrain_best_step']}"
            )
    lines.append("")
    lines.append(f"Total wall: {total_wall:.1f}s")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")

    log_lines.append("")
    log_lines.append("=== Aggregate ===")
    for ds in ("spirals", "moons"):
        agg = aggregates[ds]
        if agg is None:
            log_lines.append(f"  {ds}: no converged seeds")
            continue
        log_lines.append(f"  {ds}: float_held_mean={agg['float_held_mean']:.4f} ± {agg['float_held_std']:.4f}")
        for r in RANKS:
            pr = agg["per_rank"][str(r)]
            log_lines.append(
                f"    r={r}: noretrain={pr['no_retrain_mean']:.4f}±{pr['no_retrain_std']:.4f}  "
                f"retrain={pr['retrain_mean']:.4f}±{pr['retrain_std']:.4f}"
            )
    log_lines.append(f"spirals verdict: {spirals_v['verdict']}")
    log_lines.append(f"  rank2_retrain_gap={spirals_v.get('rank2_retrain_gap', float('nan')):+.4f}  "
                     f"rank2_noretrain_frac_of_float={spirals_v.get('rank2_noretrain_frac_of_float', float('nan')):.4f}")
    log_lines.append(f"moons context verdict: {moons_v['verdict']}")
    log_lines.append(f"OVERALL VERDICT: {overall}")
    log_lines.append(f"total wall: {total_wall:.1f}s")
    LOG_PATH.write_text("\n".join(log_lines) + "\n")

    print()
    print(f"OVERALL VERDICT (spirals): {overall}")
    print(f"spirals: {spirals_v}")
    print(f"moons:   {moons_v}")
    print(f"total wall: {total_wall:.1f}s")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
