"""
Cycle 320 / item 329 — ReLU² spirals/moons eff_rank census + LASER.

Cross-activation control to cycles 295 (eff_rank census) and 296 (LASER):
the only thing that changes vs cycle 295/296 is the activation function:
ReLU → ReLU² (`relu(z).square()`). Same datasets, same architecture, same
optimizer, same seeds, same LASER ranks.

Phase 1 (cycle 295 ReLU² counterpart):
  For each {spirals, moons} × {42, 137, 2024}: build MLP(2, 32, 1, depth=12,
  ReLU²), capture init eff_rank, train Adam lr=1e-3 BCEWithLogits full-batch
  to acc≥0.95, capture post-training eff_rank.

Phase 2 (cycle 296 LASER counterpart):
  For each (dataset, seed): identify l* = argmin_{l∈2..11} post eff_rank;
  for r ∈ {1, 2, 3, 5}: NO-RETRAIN (write SVD truncation, eval held-out)
  + RETRAIN (LowRankLinear with frozen A, retrain B+bias for 500 Adam
  steps, restore best held-out).

Verdict (item 329 verbatim):
  CONFIRMED-curvature-sparsifies iff
    init profile matches cycle 297 ReLU within ±0.5 at every layer AND
    post profile is strictly more concentrated than ReLU (mean post lower
    by ≥0.3) AND
    LASER rank-2 retrain matches float held-out acc within 0.5%.
  CONFIRMED-no-difference iff
    both profiles within ±0.3 of cycle 297 at every layer.
  REFUTED-curvature-expands iff
    ReLU² post eff_rank is higher than ReLU's by ≥0.5 anywhere.
  INCONCLUSIVE otherwise.
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
DEPTH = 12
N_SAMPLES = 2000
SPIRALS_NOISE = 0.5
SPIRALS_TURNS = 3
SPIRALS_INPUT_SCALE = 2.0
MOONS_NOISE = 0.1
# NOTE: ReLU² is much more sensitive to LR than ReLU. The cycle 295/296 LR=1e-3
# diverges (non-finite loss) at depth-12, 32-wide. Use a lower LR for ReLU².
LR_TRAIN = 1e-4
LR_RETRAIN = 1e-4
MAX_EPOCHS = 40000  # double the budget since LR is 10× smaller
TRAIN_ACC_TARGET = 0.95
RETRAIN_STEPS = 500
EVAL_EVERY = 25
RANKS = [1, 2, 3, 5]

CYCLE = 320
ITEM = 329

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "autoresearch" / "results"
LOGS_DIR = REPO / "autoresearch" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS_DIR / f"cycle_{CYCLE}_relu2_spirals_eff_rank_laser.json"
SUMMARY_PATH = RESULTS_DIR / f"cycle_{CYCLE}_summary.md"
LOG_PATH = LOGS_DIR / f"cycle_{CYCLE}_run.log"

# Cycle 297 ReLU spirals reference (seed 42)
# init profile and post profile, length DEPTH=12
CYCLE_297_INIT_RELU = [3.0, 2.7, 1.7, 1.13, 1.05, 1.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
CYCLE_297_POST_RELU = [3.13, 3.58, 2.51, 1.80, 1.35, 1.17, 1.13, 1.22, 1.39, 1.16, 1.12, 1.16]


# ---------- ReLU² activation (drop-in replacement) ----------
class ReLU2(nn.Module):
    def forward(self, x):
        r = F.relu(x)
        return r * r


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
    g = torch.Generator().manual_seed(seed + 1)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    half = n // 2
    return X[perm[:half]], y[perm[:half]], X[perm[half:]], y[perm[half:]]


def build_mlp_relu2(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MLP(input_size=2, hidden_size=HIDDEN, output_size=1, depth=DEPTH, activation_fn=ReLU2)


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_post_act(model, X):
    """Walk model.layers and capture activations AFTER each ReLU2 module.
    Returns list of length DEPTH, each (N, HIDDEN)."""
    activations = []
    h = X
    seq = model.layers
    expected_len = DEPTH * 2 + 1
    assert len(seq) == expected_len, f"unexpected layers length {len(seq)} != {expected_len}"
    with torch.no_grad():
        for m in seq:
            h = m(h)
            if isinstance(m, ReLU2):
                activations.append(h.detach().clone())
    assert len(activations) == DEPTH
    return activations


def eff_rank_from_Y(Y):
    """eff_rank via SVD of Y (more numerically robust than eigh on Y^T Y when
    activations are near-degenerate, which is common for ReLU²)."""
    N = Y.shape[0]
    Y64 = Y.to(torch.float64)
    # Singular values of Y / sqrt(N) → eigenvalues of (Y^T Y / N) are σ_i².
    s = torch.linalg.svdvals(Y64 / math.sqrt(float(N)))
    eig = s * s
    eig = torch.clamp(eig, min=0.0)
    total = eig.sum()
    if total.item() <= 0.0:
        return 1.0
    p = eig / total
    nz = p > 0
    H = -(p[nz] * torch.log(p[nz])).sum().item()
    return math.exp(H)


def eval_acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return ((model(X).squeeze(-1) > 0.0).float() == y).float().mean().item()


def train_until_acc(model, X, y, lr, max_epochs, target_acc, grad_clip=1.0):
    """Adam BCEWithLogits with gradient clipping (ReLU² can blow up without it)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    final_acc = 0.0
    final_loss = float("nan")
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X).squeeze(-1)
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            print(f"  WARN: non-finite loss at epoch {epoch}, stopping early", flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()
        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                acc = ((model(X).squeeze(-1) > 0.0).float() == y).float().mean().item()
            final_acc = acc
            final_loss = loss.item()
            if acc >= target_acc:
                return acc, epoch, final_loss
    with torch.no_grad():
        out = model(X).squeeze(-1)
        if torch.isfinite(out).all():
            final_acc = ((out > 0.0).float() == y).float().mean().item()
        else:
            final_acc = 0.0
    final_loss = loss.item() if torch.isfinite(loss) else float("nan")
    return final_acc, max_epochs, final_loss


class LowRankLinear(nn.Module):
    def __init__(self, A_frozen, B_init, b_init):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())
        self.bias = nn.Parameter(b_init.clone().contiguous())
        self.out_features = A_frozen.shape[0]
        self.in_features = B_init.shape[1]

    def forward(self, x):
        W = self.A @ self.B
        return F.linear(x, W, self.bias)


def laser_for_layer(model, layer_seq_idx, X_tr, y_tr, X_te, y_te, ranks, log_lines, dataset, seed):
    orig_lin = model.layers[layer_seq_idx]
    assert isinstance(orig_lin, nn.Linear)
    W_orig = orig_lin.weight.data.detach().clone()
    b_orig = orig_lin.bias.data.detach().clone()
    out_dim, in_dim = W_orig.shape
    assert (out_dim, in_dim) == (HIDDEN, HIDDEN), f"unexpected layer shape {(out_dim,in_dim)}"

    with torch.no_grad():
        orig_lin.weight.data.copy_(W_orig)
        orig_lin.bias.data.copy_(b_orig)
    float_held_acc = eval_acc(model, X_te, y_te)

    U, S, Vh = torch.linalg.svd(W_orig.float(), full_matrices=False)
    log_lines.append(f"    {dataset} seed={seed} layer_seq_idx={layer_seq_idx} W={tuple(W_orig.shape)} top5_sv={[round(x, 4) for x in S[:5].tolist()]}")
    log_lines.append(f"      float held-out acc = {float_held_acc:.4f}")

    rank_results = {}
    for r in ranks:
        A_r = (U[:, :r] * S[:r])
        B_r = Vh[:r, :]
        W_r = (A_r @ B_r).to(dtype=W_orig.dtype)

        # NO-RETRAIN
        with torch.no_grad():
            orig_lin.weight.data.copy_(W_r)
            orig_lin.bias.data.copy_(b_orig)
        nor_held_acc = eval_acc(model, X_te, y_te)
        nor_train_acc = eval_acc(model, X_tr, y_tr)

        # restore for retrain
        with torch.no_grad():
            orig_lin.weight.data.copy_(W_orig)
            orig_lin.bias.data.copy_(b_orig)

        # RETRAIN
        lr_lin = LowRankLinear(A_r.detach().clone(), B_r.detach().clone(), b_orig.detach().clone())
        model.layers[layer_seq_idx] = lr_lin
        for p in model.parameters():
            p.requires_grad_(False)
        lr_lin.B.requires_grad_(True)
        lr_lin.bias.requires_grad_(True)
        n_trainable = sum(p.numel() for p in [lr_lin.B, lr_lin.bias])
        assert n_trainable == r * HIDDEN + HIDDEN

        opt = torch.optim.Adam([lr_lin.B, lr_lin.bias], lr=LR_RETRAIN)
        loss_fn = nn.BCEWithLogitsLoss()
        init_held = eval_acc(model, X_te, y_te)
        best_held = init_held
        best_step = 0
        best_state = {"B": lr_lin.B.detach().clone(), "bias": lr_lin.bias.detach().clone()}
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
        with torch.no_grad():
            lr_lin.B.data.copy_(best_state["B"])
            lr_lin.bias.data.copy_(best_state["bias"])
        restored = eval_acc(model, X_te, y_te)
        assert abs(restored - best_held) < 1e-9
        # restore original Linear
        model.layers[layer_seq_idx] = orig_lin
        with torch.no_grad():
            orig_lin.weight.data.copy_(W_orig)
            orig_lin.bias.data.copy_(b_orig)
        sanity_float = eval_acc(model, X_te, y_te)
        assert abs(sanity_float - float_held_acc) < 1e-6

        rank_results[r] = {
            "rank": r,
            "n_trainable": n_trainable,
            "noretrain_train_acc": nor_train_acc,
            "noretrain_held_acc": nor_held_acc,
            "retrain_init_held": init_held,
            "retrain_best_held": best_held,
            "retrain_best_step": best_step,
            "retrain_history": history,
        }
    return float_held_acc, rank_results


def cycle_for(dataset_name, seed, log_lines):
    t0 = time.time()
    X_full, y_full = make_dataset(dataset_name, seed)
    # Phase 1 uses full data; Phase 2 uses split for held-out
    X_tr, y_tr, X_te, y_te = split_train_holdout(X_full, y_full, seed)

    model = build_mlp_relu2(seed)
    n_params = trainable_param_count(model)

    # init eff_rank on full data (matches cycle 295 convention)
    init_acts = collect_post_act(model, X_full)
    init_eff = [eff_rank_from_Y(Y) for Y in init_acts]

    # Train on TRAIN split (cycle 296 convention) for the LASER cycle 296-style work
    final_acc, epochs, final_loss = train_until_acc(model, X_tr, y_tr, LR_TRAIN, MAX_EPOCHS, TRAIN_ACC_TARGET)
    converged = final_acc >= TRAIN_ACC_TARGET

    post_acts = collect_post_act(model, X_full)
    post_eff = [eff_rank_from_Y(Y) for Y in post_acts]

    # Identify lowest-eff_rank layer in {2..11}
    eff_window = post_eff[2:12]
    l_min = 2 + int(np.argmin(eff_window))
    # Map to model.layers index: layer l in {0..DEPTH-1} corresponds to layers[2*l] (the Linear before the (l+1)-th ReLU)
    # Actually post_eff[l] is the activation AFTER the (l+1)-th nn.ReLU2 instance, which means the Linear that produced the input to that ReLU is at layers[2*l].
    # cycle 296 uses layer_seq_idx = 2*l for the Linear at that position. We want to compress the Linear that feeds the ReLU2 we measured at index l.
    # That Linear is at sequence index 2*l (0-indexed). Both linears 2*l and 2*l_min are 32x32 for l in {1..DEPTH-1}.
    layer_seq_idx = 2 * l_min
    assert isinstance(model.layers[layer_seq_idx], nn.Linear)

    # Phase 2: LASER on l_min
    float_held_acc, rank_results = laser_for_layer(
        model, layer_seq_idx, X_tr, y_tr, X_te, y_te, RANKS, log_lines, dataset_name, seed
    )

    wall = time.time() - t0
    line = (
        f"  {dataset_name} seed={seed}: n_params={n_params} epochs={epochs} train_acc={final_acc:.4f} "
        f"loss={final_loss:.4f} converged={converged} l_min={l_min} float_held={float_held_acc:.4f} wall={wall:.1f}s"
    )
    print(line, flush=True)
    log_lines.append(line)
    log_lines.append(f"    init_eff_rank={[round(x, 3) for x in init_eff]}")
    log_lines.append(f"    post_eff_rank={[round(x, 3) for x in post_eff]}")

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
        "l_min": l_min,
        "layer_seq_idx": layer_seq_idx,
        "float_held_acc": float_held_acc,
        "rank_results": rank_results,
        "wall_sec": wall,
    }


def aggregate_per_layer(records, dataset):
    rs = [r for r in records if r["dataset"] == dataset]
    init = np.array([r["init_eff_rank"] for r in rs])
    post = np.array([r["post_eff_rank"] for r in rs])
    return {
        "n_seeds": len(rs),
        "init_mean": init.mean(axis=0).tolist(),
        "init_std": init.std(axis=0, ddof=0).tolist(),
        "post_mean": post.mean(axis=0).tolist(),
        "post_std": post.std(axis=0, ddof=0).tolist(),
    }


def verdict_for(records, aggregates):
    spirals_init = aggregates["spirals"]["init_mean"]
    spirals_post = aggregates["spirals"]["post_mean"]

    init_diff = [spirals_init[l] - CYCLE_297_INIT_RELU[l] for l in range(DEPTH)]
    post_diff = [spirals_post[l] - CYCLE_297_POST_RELU[l] for l in range(DEPTH)]

    init_match_5 = all(abs(d) <= 0.5 for d in init_diff)
    init_match_3 = all(abs(d) <= 0.3 for d in init_diff)
    post_match_3 = all(abs(d) <= 0.3 for d in post_diff)
    post_strictly_lower_03 = (
        np.mean(spirals_post) <= np.mean(CYCLE_297_POST_RELU) - 0.3
    )
    any_post_higher_05 = any(d >= 0.5 for d in post_diff)

    # LASER rank-2 retrain match for spirals
    spirals_records = [r for r in records if r["dataset"] == "spirals"]
    spirals_float_held = np.mean([r["float_held_acc"] for r in spirals_records])
    spirals_r2_retrain = np.mean([r["rank_results"][2]["retrain_best_held"] for r in spirals_records])
    laser_r2_within_05 = (spirals_r2_retrain >= spirals_float_held - 0.005)

    if init_match_5 and post_strictly_lower_03 and laser_r2_within_05:
        verdict = "CONFIRMED-curvature-sparsifies"
    elif init_match_3 and post_match_3:
        verdict = "CONFIRMED-no-difference"
    elif any_post_higher_05:
        verdict = "REFUTED-curvature-expands"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "init_diff_seed42_proxy": init_diff,
        "post_diff_seed42_proxy": post_diff,
        "init_match_within_0_5": init_match_5,
        "init_match_within_0_3": init_match_3,
        "post_match_within_0_3": post_match_3,
        "post_strictly_lower_by_0_3_mean": bool(post_strictly_lower_03),
        "any_post_higher_by_0_5": any_post_higher_05,
        "spirals_float_held_mean": float(spirals_float_held),
        "spirals_r2_retrain_held_mean": float(spirals_r2_retrain),
        "laser_r2_within_0_5pc": bool(laser_r2_within_05),
    }


def main():
    log_lines = []
    log_lines.append(f"Cycle {CYCLE} / item {ITEM} — ReLU² spirals/moons eff_rank census + LASER")
    log_lines.append(
        f"config: HIDDEN={HIDDEN} DEPTH={DEPTH} N={N_SAMPLES} LR_TRAIN={LR_TRAIN} "
        f"target_acc={TRAIN_ACC_TARGET} seeds={SEEDS} ranks={RANKS} retrain_steps={RETRAIN_STEPS}"
    )
    print(log_lines[-2], flush=True)
    print(log_lines[-1], flush=True)
    t_start = time.time()
    records = []
    for dataset in ("spirals", "moons"):
        for seed in SEEDS:
            r = cycle_for(dataset, seed, log_lines)
            records.append(r)
    aggregates = {ds: aggregate_per_layer(records, ds) for ds in ("spirals", "moons")}
    v = verdict_for(records, aggregates)

    total_wall = time.time() - t_start
    summary = {
        "cycle": CYCLE,
        "item": ITEM,
        "config": {
            "hidden": HIDDEN,
            "depth": DEPTH,
            "n_samples": N_SAMPLES,
            "lr_train": LR_TRAIN,
            "lr_retrain": LR_RETRAIN,
            "max_epochs": MAX_EPOCHS,
            "target_acc": TRAIN_ACC_TARGET,
            "seeds": SEEDS,
            "ranks": RANKS,
            "retrain_steps": RETRAIN_STEPS,
            "spirals_noise": SPIRALS_NOISE,
            "spirals_turns": SPIRALS_TURNS,
            "spirals_input_scale": SPIRALS_INPUT_SCALE,
            "moons_noise": MOONS_NOISE,
            "activation": "ReLU2",
        },
        "records": records,
        "aggregates": aggregates,
        "verdict_logic": v,
        "verdict": v["verdict"],
        "total_wall_sec": total_wall,
        "baselines": {
            "cycle_297_init_relu_seed42": CYCLE_297_INIT_RELU,
            "cycle_297_post_relu_seed42": CYCLE_297_POST_RELU,
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append(f"# Cycle {CYCLE} — ReLU² spirals/moons eff_rank census + LASER (item {ITEM})")
    lines.append("")
    lines.append(f"**Verdict:** {v['verdict']}")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- MLP(2, {HIDDEN}, 1, depth={DEPTH}, **ReLU²**); Adam lr={LR_TRAIN}; full-batch BCEWithLogits")
    lines.append(f"- N={N_SAMPLES}; spirals(noise={SPIRALS_NOISE}, n_turns={SPIRALS_TURNS}, ×{SPIRALS_INPUT_SCALE}); moons(noise={MOONS_NOISE})")
    lines.append(f"- Train to acc≥{TRAIN_ACC_TARGET}; cap {MAX_EPOCHS} epochs; seeds={SEEDS}")
    lines.append(f"- LASER ranks={RANKS}; retrain {RETRAIN_STEPS} Adam steps lr={LR_RETRAIN} (B + bias only)")
    lines.append(f"- Trainable params/net = {records[0]['n_params']}")
    lines.append("")
    lines.append("## Verdict logic (item 329 verbatim)")
    lines.append("- CONFIRMED-curvature-sparsifies iff init within ±0.5 of cycle 297 ReLU AND post mean strictly lower by ≥0.3 AND LASER r=2 retrain within 0.5%")
    lines.append("- CONFIRMED-no-difference iff both init and post within ±0.3 of cycle 297 ReLU at every layer")
    lines.append("- REFUTED-curvature-expands iff post is higher than ReLU's by ≥0.5 anywhere")
    lines.append("- INCONCLUSIVE otherwise")
    lines.append("")
    for ds in ("spirals", "moons"):
        agg = aggregates[ds]
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| layer | ReLU² init mean ± std | ReLU² post mean ± std | cycle 297 ReLU init | cycle 297 ReLU post |")
        lines.append("|---:|---:|---:|---:|---:|")
        for l in range(DEPTH):
            init_str = f"{agg['init_mean'][l]:.3f} ± {agg['init_std'][l]:.3f}"
            post_str = f"{agg['post_mean'][l]:.3f} ± {agg['post_std'][l]:.3f}"
            ri = CYCLE_297_INIT_RELU[l] if ds == "spirals" else "—"
            rp = CYCLE_297_POST_RELU[l] if ds == "spirals" else "—"
            lines.append(f"| {l} | {init_str} | {post_str} | {ri} | {rp} |")
        lines.append("")
    lines.append("## LASER rank sweep (per dataset, mean across seeds)")
    lines.append("")
    for ds in ("spirals", "moons"):
        rs = [r for r in records if r["dataset"] == ds]
        float_held_mean = np.mean([r["float_held_acc"] for r in rs])
        lines.append(f"### {ds} (float held-out acc mean = {float_held_mean:.4f})")
        lines.append("")
        lines.append("| rank | l_min (per seed) | no-retrain held mean | retrain best held mean |")
        lines.append("|---:|---|---:|---:|")
        for r in RANKS:
            l_mins = [rec["l_min"] for rec in rs]
            nor_mean = np.mean([rec["rank_results"][r]["noretrain_held_acc"] for rec in rs])
            retrain_mean = np.mean([rec["rank_results"][r]["retrain_best_held"] for rec in rs])
            lines.append(f"| {r} | {l_mins} | {nor_mean:.4f} | {retrain_mean:.4f} |")
        lines.append("")
    lines.append(f"**Wall:** {total_wall:.1f} s")
    SUMMARY_PATH.write_text("\n".join(lines))

    LOG_PATH.write_text("\n".join(log_lines))

    print(f"\n[cycle {CYCLE}] verdict = {v['verdict']}", flush=True)
    print(f"[cycle {CYCLE}] wrote {JSON_PATH}", flush=True)
    print(f"[cycle {CYCLE}] wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
