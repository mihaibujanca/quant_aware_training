"""
Cycle 298 / item 308 — SDF rank census + LASER replication on 3D mesh fitting.

Train an MLP(3 -> 256 -> 256 -> 256 -> 256 -> 1) on torus SDF. Measure:
  - per-layer post-ReLU eff_rank on TRAIN
  - jacobian_rank_profile (gradient ensemble SVD, scalar-output) on TRAIN
Pick l* = argmin eff_rank over hidden layers 1..3 (skip layer 0 because input
weight (256, 3) is not 256x256). For r in {2, 3, 5, 10}: SVD-truncate W_l* and
(a) eval no-retrain held-out MSE, (b) replace with LowRankLinear, freeze
everything else, retrain B+bias 500 steps Adam, eval best held-out MSE,
restore. Verdict per item 308 verbatim.

Item 308 hypothesis: hidden eff_rank ≈ 3 (input dim), gradient ensemble eff_rank
≈ 2 (surface manifold dim), rank-3 retrain matches float, rank-2 fails.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from aleph.geometric_compression import jacobian_rank_profile
from aleph.mesh_datasets import load_mesh, sample_sdf_points
from aleph.models import MLP

# ---------- config ----------
SEED = 42
HIDDEN = 256
DEPTH = 4  # number of hidden Linear+ReLU pairs
INPUT_DIM = 3
N_POINTS = 30_000
SURFACE_RATIO = 0.5
SURFACE_NOISE = 0.01
MESH_NAME = "torus"
LR_TRAIN = 1e-3
LR_RETRAIN = 1e-3
MAX_EPOCHS = 5000
PLATEAU_PATIENCE = 250  # stop if held-out MSE doesn't improve for this many epochs
PLATEAU_MIN_REL_IMPROVE = 1e-4
RETRAIN_STEPS = 500
EVAL_EVERY = 25
RANKS = [2, 3, 5, 10]
TRAIN_FRAC = 0.8
FLOAT_HELD_MSE_REQUIREMENT = 0.01  # CLAUDE.md verify task performance

# Verdict thresholds (item 308 verbatim)
CONFIRMED_HIDDEN_LO = 2.5
CONFIRMED_HIDDEN_HI = 3.5
CONFIRMED_GRAD_LO = 1.8
CONFIRMED_GRAD_HI = 2.5
CONFIRMED_RANK3_RATIO = 1.1
REFUTED_HIDDEN_HI = 8.0
REFUTED_RANK3_RATIO = 2.0

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "autoresearch" / "results"
LOGS_DIR = REPO / "autoresearch" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS_DIR / "cycle_298_sdf_rank_laser.json"
SUMMARY_PATH = RESULTS_DIR / "cycle_298_summary.md"
LOG_PATH = LOGS_DIR / "cycle_298_run.log"


def make_sdf_data(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mesh = load_mesh(MESH_NAME)
    pts, sdf = sample_sdf_points(
        mesh, n_points=N_POINTS, surface_ratio=SURFACE_RATIO, surface_noise=SURFACE_NOISE
    )
    g = torch.Generator().manual_seed(seed + 1)
    n = pts.shape[0]
    perm = torch.randperm(n, generator=g)
    n_train = int(n * TRAIN_FRAC)
    train_idx = perm[:n_train]
    held_idx = perm[n_train:]
    return pts[train_idx], sdf[train_idx], pts[held_idx], sdf[held_idx]


def build_mlp(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MLP(input_size=INPUT_DIM, hidden_size=HIDDEN, output_size=1, depth=DEPTH, activation_fn=nn.ReLU)


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_mse(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        return float(((pred - y) ** 2).mean().item())


def train_until_plateau(model, X_tr, y_tr, X_te, y_te, lr, max_epochs, log_lines):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_held = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(X_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == 1:
            held = eval_mse(model, X_te, y_te)
            tr = float(loss.item())
            history.append({"epoch": epoch, "train_mse": tr, "held_mse": held})
            improved = (best_held - held) / max(best_held, 1e-12) > PLATEAU_MIN_REL_IMPROVE
            if held < best_held:
                best_held = held
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 25
            if epochs_no_improve >= PLATEAU_PATIENCE:
                log_lines.append(
                    f"  plateau at epoch {epoch}: best_held={best_held:.6e}@{best_epoch} "
                    f"(no improve >= {PLATEAU_MIN_REL_IMPROVE} for {PLATEAU_PATIENCE} epochs)"
                )
                break
    model.load_state_dict(best_state)
    final_train = eval_mse(model, X_tr, y_tr)
    final_held = eval_mse(model, X_te, y_te)
    return final_train, final_held, best_epoch, history


def collect_post_relu(model, X):
    """Collect post-ReLU activations on inputs for layers l in {0..DEPTH-1}."""
    activations = []
    h = X
    seq = model.layers
    expected = DEPTH * 2 + 1
    assert len(seq) == expected, f"unexpected layers length {len(seq)} != {expected}"
    # MLP has quant/dequant stubs; bypass them by using internal sequential
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
    """

    def __init__(self, A_frozen, B_init, b_init):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())
        self.bias = nn.Parameter(b_init.clone().contiguous())
        self.out_features = A_frozen.shape[0]
        self.in_features = B_init.shape[1]

    def forward(self, x):
        W = self.A @ self.B  # (out, in)
        return torch.nn.functional.linear(x, W, self.bias)


def laser_for_layer(model, layer_seq_idx, X_tr, y_tr, X_te, y_te, ranks, log_lines):
    """For each rank r: no-retrain held-out MSE, then retrain held-out best MSE."""
    orig_lin = model.layers[layer_seq_idx]
    assert isinstance(orig_lin, nn.Linear)
    W_orig = orig_lin.weight.data.detach().clone()
    b_orig = orig_lin.bias.data.detach().clone()
    out_dim, in_dim = W_orig.shape
    assert (out_dim, in_dim) == (HIDDEN, HIDDEN), (
        f"layer weight shape {(out_dim,in_dim)} != ({HIDDEN},{HIDDEN})"
    )

    with torch.no_grad():
        orig_lin.weight.data.copy_(W_orig)
        orig_lin.bias.data.copy_(b_orig)
    float_held_mse = eval_mse(model, X_te, y_te)

    U, S, Vh = torch.linalg.svd(W_orig.float(), full_matrices=False)
    s_top10 = S[:10].tolist()
    log_lines.append(f"    layer_seq_idx={layer_seq_idx} W shape={tuple(W_orig.shape)}")
    log_lines.append(f"      SVD top-10: {[round(x,4) for x in s_top10]}")
    log_lines.append(f"      float held-out MSE = {float_held_mse:.6e}")

    rank_results = {}
    for r in ranks:
        # NO-RETRAIN
        A_r = (U[:, :r] * S[:r])
        B_r = Vh[:r, :]
        W_r = (A_r @ B_r).to(dtype=W_orig.dtype)
        with torch.no_grad():
            orig_lin.weight.data.copy_(W_r)
            orig_lin.bias.data.copy_(b_orig)
        nor_train_mse = eval_mse(model, X_tr, y_tr)
        nor_held_mse = eval_mse(model, X_te, y_te)

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

        for p in model.parameters():
            p.requires_grad_(False)
        lr_lin.B.requires_grad_(True)
        lr_lin.bias.requires_grad_(True)
        n_trainable = sum(p.numel() for p in [lr_lin.B, lr_lin.bias])
        expected_trainable = r * HIDDEN + HIDDEN
        assert n_trainable == expected_trainable, f"trainable {n_trainable} != {expected_trainable}"

        opt = torch.optim.Adam([lr_lin.B, lr_lin.bias], lr=LR_RETRAIN)
        loss_fn = nn.MSELoss()

        init_held = eval_mse(model, X_te, y_te)
        assert abs(init_held - nor_held_mse) < 1e-9, f"init held {init_held} != noretrain {nor_held_mse}"
        best_held = init_held
        best_step = 0
        best_state = {
            "B": lr_lin.B.detach().clone(),
            "bias": lr_lin.bias.detach().clone(),
        }
        history = [{"step": 0, "held_mse": float(init_held)}]

        for step in range(1, RETRAIN_STEPS + 1):
            model.train()
            opt.zero_grad()
            pred = model(X_tr)
            loss = loss_fn(pred, y_tr)
            loss.backward()
            opt.step()
            if step % EVAL_EVERY == 0:
                held = eval_mse(model, X_te, y_te)
                history.append({"step": step, "held_mse": float(held), "loss": float(loss.item())})
                if held < best_held:
                    best_held = held
                    best_step = step
                    best_state["B"] = lr_lin.B.detach().clone()
                    best_state["bias"] = lr_lin.bias.detach().clone()

        with torch.no_grad():
            lr_lin.B.data.copy_(best_state["B"])
            lr_lin.bias.data.copy_(best_state["bias"])
        restored = eval_mse(model, X_te, y_te)
        assert abs(restored - best_held) < 1e-9

        # restore original Linear in place
        model.layers[layer_seq_idx] = orig_lin
        # re-enable grads on full model so future passes (or other tests) can train
        for p in model.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            orig_lin.weight.data.copy_(W_orig)
            orig_lin.bias.data.copy_(b_orig)
        post_restore_held = eval_mse(model, X_te, y_te)
        assert abs(post_restore_held - float_held_mse) < 1e-9, (
            f"post-restore held {post_restore_held} != float {float_held_mse}"
        )

        rank_results[r] = {
            "no_retrain_train_mse": float(nor_train_mse),
            "no_retrain_held_mse": float(nor_held_mse),
            "retrain_init_held_mse": float(init_held),
            "retrain_best_held_mse": float(best_held),
            "retrain_best_step": int(best_step),
            "retrain_n_trainable": int(n_trainable),
            "history": history,
        }
        log_lines.append(
            f"      r={r}: noretrain held={nor_held_mse:.6e}  "
            f"retrain best={best_held:.6e}@{best_step}  "
            f"ratio_vs_float={best_held/float_held_mse:.4f}"
        )

    return float_held_mse, rank_results, [round(x, 4) for x in s_top10]


def verdict_for(hidden_layer_means, grad_eff_rank, rank3_retrain_mse, float_mse):
    """Item 308 verbatim verdict."""
    hidden_mean = float(np.mean(hidden_layer_means))
    in_hidden = (hidden_mean >= CONFIRMED_HIDDEN_LO) and (hidden_mean <= CONFIRMED_HIDDEN_HI)
    in_grad = (grad_eff_rank >= CONFIRMED_GRAD_LO) and (grad_eff_rank <= CONFIRMED_GRAD_HI)
    rank3_ratio = rank3_retrain_mse / float_mse if float_mse > 0 else float("inf")
    rank3_ok = rank3_ratio <= CONFIRMED_RANK3_RATIO

    confirmed = in_hidden and in_grad and rank3_ok
    refuted = (hidden_mean >= REFUTED_HIDDEN_HI) or (rank3_ratio >= REFUTED_RANK3_RATIO)

    if confirmed:
        v = "CONFIRMED"
    elif refuted:
        v = "REFUTED"
    else:
        v = "INCONCLUSIVE"
    return {
        "verdict": v,
        "hidden_mean_eff_rank": hidden_mean,
        "hidden_in_range": in_hidden,
        "grad_eff_rank": float(grad_eff_rank),
        "grad_in_range": in_grad,
        "rank3_retrain_mse": float(rank3_retrain_mse),
        "float_held_mse": float(float_mse),
        "rank3_retrain_ratio": float(rank3_ratio),
        "rank3_ratio_ok": rank3_ok,
        "thresholds": {
            "hidden_lo": CONFIRMED_HIDDEN_LO,
            "hidden_hi": CONFIRMED_HIDDEN_HI,
            "grad_lo": CONFIRMED_GRAD_LO,
            "grad_hi": CONFIRMED_GRAD_HI,
            "confirmed_rank3_ratio": CONFIRMED_RANK3_RATIO,
            "refuted_hidden_hi": REFUTED_HIDDEN_HI,
            "refuted_rank3_ratio": REFUTED_RANK3_RATIO,
        },
    }


def main():
    log_lines = []
    log_lines.append("Cycle 298 / item 308 — SDF rank census + LASER replication on 3D mesh fitting")
    log_lines.append(
        f"config: HIDDEN={HIDDEN} DEPTH={DEPTH} INPUT_DIM={INPUT_DIM} N={N_POINTS} "
        f"surface_ratio={SURFACE_RATIO} mesh={MESH_NAME} LR_TRAIN={LR_TRAIN} "
        f"LR_RETRAIN={LR_RETRAIN} max_epochs={MAX_EPOCHS} retrain_steps={RETRAIN_STEPS} "
        f"eval_every={EVAL_EVERY} ranks={RANKS} seed={SEED}"
    )
    print(log_lines[-2], flush=True)
    print(log_lines[-1], flush=True)
    t_start = time.time()

    # 1. Data
    X_tr, y_tr, X_te, y_te = make_sdf_data(SEED)
    log_lines.append(
        f"  data: train {tuple(X_tr.shape)} y {tuple(y_tr.shape)} | "
        f"held {tuple(X_te.shape)} y {tuple(y_te.shape)}"
    )
    log_lines.append(
        f"  sdf train abs mean={y_tr.abs().mean().item():.4f} held abs mean={y_te.abs().mean().item():.4f}"
    )

    # 2. Model
    model = build_mlp(SEED)
    n_params = trainable_param_count(model)
    log_lines.append(f"  MLP params: {n_params}")

    # 3. Train
    t_train = time.time()
    final_train_mse, float_held_mse, best_epoch, train_history = train_until_plateau(
        model, X_tr, y_tr, X_te, y_te, LR_TRAIN, MAX_EPOCHS, log_lines
    )
    train_wall = time.time() - t_train
    log_lines.append(
        f"  trained: train_mse={final_train_mse:.6e} held_mse={float_held_mse:.6e} "
        f"best_epoch={best_epoch} wall={train_wall:.1f}s"
    )

    if float_held_mse > FLOAT_HELD_MSE_REQUIREMENT:
        log_lines.append(
            f"  WARNING: float held MSE {float_held_mse:.6e} > requirement "
            f"{FLOAT_HELD_MSE_REQUIREMENT:.6e}; CLAUDE.md verify-task-perf rule"
        )
        print("  WARNING: float held MSE above requirement", flush=True)

    # 4. Hidden eff_rank census
    train_acts = collect_post_relu(model, X_tr)
    train_eff = [eff_rank_from_Y(Y) for Y in train_acts]
    log_lines.append(f"  hidden eff_rank per layer: {[round(x,3) for x in train_eff]}")

    # 5. Gradient ensemble eff_rank
    grad_profile = jacobian_rank_profile(model, X_tr, chunk_size=512)
    grad_eff_rank = grad_profile["effective_rank"]
    grad_sv = grad_profile["singular_values"].tolist()
    log_lines.append(
        f"  jacobian_rank_profile: eff_rank={grad_eff_rank:.4f} "
        f"rank_90={grad_profile['rank_90']} rank_95={grad_profile['rank_95']} "
        f"rank_99={grad_profile['rank_99']} sv={[round(x,4) for x in grad_sv]}"
    )

    # 6. Pick l*: argmin eff_rank over hidden layers 1..3 (skip layer 0; (256, 3))
    candidate_layers = list(range(1, DEPTH))
    l_star = min(candidate_layers, key=lambda l: train_eff[l])
    layer_seq_idx = 2 * l_star  # Linear at this index produces post-ReLU at index 2*l_star + 1
    log_lines.append(
        f"  l* = {l_star} (eff_rank[l*] = {train_eff[l_star]:.4f}); layer_seq_idx={layer_seq_idx}"
    )

    # 7. LASER sweep
    held_baseline, rank_results, svd_top10 = laser_for_layer(
        model, layer_seq_idx, X_tr, y_tr, X_te, y_te, RANKS, log_lines
    )
    assert abs(held_baseline - float_held_mse) < 1e-9, (
        f"held_baseline {held_baseline} != float_held_mse {float_held_mse}"
    )

    # 8. Verdict
    rank3_retrain = rank_results[3]["retrain_best_held_mse"]
    v = verdict_for(train_eff[1:DEPTH], grad_eff_rank, rank3_retrain, float_held_mse)
    log_lines.append(f"verdict: {v['verdict']}")
    log_lines.append(
        f"  hidden_mean_eff_rank={v['hidden_mean_eff_rank']:.4f} (target [2.5,3.5]) "
        f"in_range={v['hidden_in_range']}"
    )
    log_lines.append(
        f"  grad_eff_rank={v['grad_eff_rank']:.4f} (target [1.8,2.5]) in_range={v['grad_in_range']}"
    )
    log_lines.append(
        f"  rank3_retrain_ratio_vs_float={v['rank3_retrain_ratio']:.4f} (CONFIRMED <=1.1, REFUTED >=2.0)"
    )

    total_wall = time.time() - t_start

    summary = {
        "cycle": 298,
        "item": 308,
        "config": {
            "hidden": HIDDEN, "depth": DEPTH, "input_dim": INPUT_DIM,
            "n_points": N_POINTS, "train_frac": TRAIN_FRAC,
            "surface_ratio": SURFACE_RATIO, "surface_noise": SURFACE_NOISE,
            "mesh": MESH_NAME, "lr_train": LR_TRAIN, "lr_retrain": LR_RETRAIN,
            "max_epochs": MAX_EPOCHS, "plateau_patience": PLATEAU_PATIENCE,
            "retrain_steps": RETRAIN_STEPS, "eval_every": EVAL_EVERY,
            "ranks": RANKS, "seed": SEED,
            "layer_selection": "argmin eff_rank over l in [1..3]",
        },
        "n_params": n_params,
        "train_history": train_history,
        "final_train_mse": float(final_train_mse),
        "float_held_mse": float(float_held_mse),
        "best_train_epoch": int(best_epoch),
        "hidden_eff_rank_per_layer": train_eff,
        "grad_profile": {
            "effective_rank": grad_eff_rank,
            "rank_90": int(grad_profile["rank_90"]),
            "rank_95": int(grad_profile["rank_95"]),
            "rank_99": int(grad_profile["rank_99"]),
            "singular_values": grad_sv,
            "n_samples": int(grad_profile["n_samples"]),
            "input_dim": int(grad_profile["input_dim"]),
        },
        "l_star": int(l_star),
        "layer_seq_idx": int(layer_seq_idx),
        "svd_top10": svd_top10,
        "ranks": {str(r): rank_results[r] for r in RANKS},
        "verdict": v,
        "verdict_label": v["verdict"],
        "train_wall_sec": float(train_wall),
        "total_wall_sec": float(total_wall),
        "baselines": {
            "cycle_280_block1_rank2_retrain_ppl": 49.03,
            "cycle_280_float_ppl": 53.895,
            "cycle_296_spirals_rank2_retrain_acc": 0.8207,
            "cycle_296_spirals_float_acc": 0.8270,
        },
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    # Markdown summary
    lines = []
    lines.append("# Cycle 298 — SDF rank census + LASER replication (item 308)")
    lines.append("")
    lines.append(f"**Verdict: {v['verdict']}**")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- MLP({INPUT_DIM}, {HIDDEN}, 1, depth={DEPTH}, ReLU); Adam lr={LR_TRAIN}; full-batch MSE")
    lines.append(f"- mesh={MESH_NAME}; N={N_POINTS} points (train_frac={TRAIN_FRAC})")
    lines.append(f"- surface_ratio={SURFACE_RATIO}, surface_noise={SURFACE_NOISE}, seed={SEED}")
    lines.append(f"- max_epochs={MAX_EPOCHS}, plateau_patience={PLATEAU_PATIENCE}")
    lines.append(f"- LASER: l* = argmin eff_rank over l in [1..3]; SVD W (256x256); ranks {RANKS}")
    lines.append(f"- Retrain: {RETRAIN_STEPS} Adam steps lr={LR_RETRAIN}, eval held every {EVAL_EVERY}")
    lines.append(f"- Trainable per fit: 256(r+1)")
    lines.append("")
    lines.append("## Verdict logic (item 308 verbatim)")
    lines.append("- CONFIRMED iff (a) hidden eff_rank ∈ [2.5, 3.5] AND (b) grad eff_rank ∈ [1.8, 2.5] AND (c) rank-3 retrain MSE ≤ 1.1× float MSE")
    lines.append("- REFUTED iff hidden eff_rank ≥ 8 OR rank-3 retrain MSE ≥ 2× float")
    lines.append("- INCONCLUSIVE between")
    lines.append("")
    lines.append("## Training")
    lines.append(f"- final train MSE: {final_train_mse:.6e}")
    lines.append(f"- float held-out MSE: **{float_held_mse:.6e}**")
    lines.append(f"- best epoch: {best_epoch} (plateau-stop or max)")
    lines.append(f"- params: {n_params}")
    lines.append("")
    lines.append("## Hidden eff_rank census")
    for i, e in enumerate(train_eff):
        marker = " ← l*" if i == l_star else ""
        lines.append(f"- layer {i}: eff_rank = {e:.4f}{marker}")
    lines.append(f"- mean over hidden 1..3: **{v['hidden_mean_eff_rank']:.4f}** (target [2.5, 3.5])")
    lines.append("")
    lines.append("## Gradient ensemble (jacobian_rank_profile)")
    lines.append(f"- effective_rank: **{grad_eff_rank:.4f}** (target [1.8, 2.5])")
    lines.append(f"- rank_90: {grad_profile['rank_90']}, rank_95: {grad_profile['rank_95']}, rank_99: {grad_profile['rank_99']}")
    lines.append(f"- top singular values: {[round(x,4) for x in grad_sv[:5]]}")
    lines.append(f"- n_samples: {grad_profile['n_samples']}, input_dim: {grad_profile['input_dim']}")
    lines.append("")
    lines.append("## LASER sweep at l* = " + str(l_star))
    lines.append(f"- SVD top-10 of W_l*: {svd_top10}")
    lines.append("")
    lines.append("| r | no_retrain_held_mse | retrain_best_held_mse | retrain_best_step | ratio_vs_float |")
    lines.append("|--:|--------------------:|----------------------:|------------------:|---------------:|")
    for r in RANKS:
        rr = rank_results[r]
        ratio = rr["retrain_best_held_mse"] / float_held_mse
        lines.append(
            f"| {r} | {rr['no_retrain_held_mse']:.6e} | {rr['retrain_best_held_mse']:.6e} "
            f"| {rr['retrain_best_step']} | {ratio:.4f} |"
        )
    lines.append("")
    lines.append("## Verdict reasoning")
    lines.append(
        f"- hidden mean eff_rank = {v['hidden_mean_eff_rank']:.4f} in [2.5, 3.5]? **{v['hidden_in_range']}**"
    )
    lines.append(
        f"- grad eff_rank = {v['grad_eff_rank']:.4f} in [1.8, 2.5]? **{v['grad_in_range']}**"
    )
    lines.append(
        f"- rank-3 retrain ratio = {v['rank3_retrain_ratio']:.4f} ≤ 1.1? **{v['rank3_ratio_ok']}**"
    )
    lines.append("")
    lines.append(f"**Final verdict: {v['verdict']}**")
    lines.append("")
    lines.append(f"Total wall: {total_wall:.1f}s")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")

    log_lines.append(f"total wall: {total_wall:.1f}s")
    LOG_PATH.write_text("\n".join(log_lines) + "\n")

    print()
    print(f"VERDICT: {v['verdict']}")
    print(f"  hidden mean eff_rank = {v['hidden_mean_eff_rank']:.4f}")
    print(f"  grad  eff_rank       = {v['grad_eff_rank']:.4f}")
    print(f"  rank-3 retrain ratio = {v['rank3_retrain_ratio']:.4f}")
    print(f"  float held MSE       = {float_held_mse:.6e}")
    print(f"total wall: {total_wall:.1f}s")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {LOG_PATH}")


if __name__ == "__main__":
    main()
