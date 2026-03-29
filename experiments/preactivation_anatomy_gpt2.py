"""
Cycle 108: Pre-activation distribution anatomy — GPT-2 FFN
Item 119

Null hypothesis: Pre-activation concentration in |z|<0.5 is fully explained
by per-channel variance; Gaussian N(μ_j, σ²_j) predicts zone fractions within 5%.

Collects pre-activations z = W_in @ LN(x) + b_in for all 12 FFN blocks,
fits per-channel Gaussians, computes zone fractions, compares empirical vs Gaussian.
"""
import json
import sys
import math
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Use the library's GPT-2 loading infrastructure for correct PPL
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aleph.lm_geometric_compression import (
    build_probe_corpus,
    load_nanochat_teacher,
    ProbeCorpusSpec,
    TeacherLoadSpec,
)

# ── paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
DATA_FILE = REPO / "data" / "openwebtext_sample.txt"
OUT_JSON = REPO / "autoresearch" / "results" / "cycle_108_preactivation_anatomy.json"
OUT_PLOT = REPO / "autoresearch" / "results" / "cycle_108_preactivation_anatomy.png"

# ── config ─────────────────────────────────────────────────────────────────────
N_TOKENS = 4096
SEQ_LEN = 128
BATCH_SIZE = 4
PPL_SEQS = 200
THRESHOLDS = [0.25, 0.5, 1.0, 1.5, 2.0]

print("Loading GPT-2 via library...", flush=True)
teacher = load_nanochat_teacher(
    TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device="auto")
)
model = teacher.model
DEVICE = teacher.device
print(f"Device: {DEVICE}", flush=True)


# ── PPL baseline ───────────────────────────────────────────────────────────────

def compute_ppl(teacher, n_seqs=PPL_SEQS) -> float:
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(DATA_FILE),
            sequence_length=SEQ_LEN,
            probe_train_sequences=50,
            probe_holdout_sequences=50,
            eval_val_sequences=n_seqs,
        )
    )
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, corpus.eval_val.shape[0], BATCH_SIZE):
            batch = corpus.eval_val[i : i + BATCH_SIZE].to(DEVICE)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            out = model(inp)
            logits = out.logits if hasattr(out, "logits") else out[0]
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt.reshape(-1),
                reduction="mean",
            )
            total_loss += loss.item() * tgt.numel()
            total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ── pre-activation collection ─────────────────────────────────────────────────

def collect_preactivations(n_tokens: int) -> dict[int, torch.Tensor]:
    """Hook c_fc output (pre-GELU) for all 12 blocks."""
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(DATA_FILE),
            sequence_length=SEQ_LEN,
            probe_train_sequences=50,
            probe_holdout_sequences=n_tokens // SEQ_LEN + 10,
            eval_val_sequences=50,
        )
    )
    # probe_holdout shape: (N_seqs, SEQ_LEN+1)
    seqs = corpus.probe_holdout  # (N_seqs, SEQ_LEN+1)

    preacts: dict[int, list[torch.Tensor]] = {i: [] for i in range(12)}
    token_counts = [0] * 12

    def make_hook(layer_idx):
        def hook_fn(module, inputs, output):
            if token_counts[layer_idx] >= n_tokens:
                return
            z = output.detach().float()  # (B, T, 3072)
            flat = z.reshape(-1, z.shape[-1])
            remaining = n_tokens - token_counts[layer_idx]
            take = min(flat.shape[0], remaining)
            preacts[layer_idx].append(flat[:take].cpu())
            token_counts[layer_idx] += take
        return hook_fn

    handles = []
    for i in range(12):
        h = model.transformer.h[i].mlp.c_fc.register_forward_hook(make_hook(i))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for i in range(0, seqs.shape[0], BATCH_SIZE):
            batch = seqs[i : i + BATCH_SIZE].to(DEVICE)
            inp = batch[:, :-1]
            _ = model(inp)
            if all(c >= n_tokens for c in token_counts):
                break

    for h in handles:
        h.remove()

    return {i: torch.cat(preacts[i], dim=0) for i in range(12)}


# ── importance scores ─────────────────────────────────────────────────────────

def compute_importance_scores(preacts: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    """Reuse pre-activations to compute importance: std(h)*||w_out||*(|mean(h)|+|mean(deriv)|)."""
    scores = {}
    for i in range(12):
        z = preacts[i]  # (N, 3072)
        h = torch.nn.functional.gelu(z)
        # GPT-2 uses Conv1D: c_proj.weight is (3072, 768) — rows = hidden units
        w_out = model.transformer.h[i].mlp.c_proj.weight.float().cpu()  # (3072, 768)
        output_norms = w_out.norm(dim=1)  # (3072,) — one norm per hidden unit
        std_h = h.std(dim=0)
        mean_h = h.mean(dim=0)
        # Sigmoid-weighted derivative of GELU (approximation used in library)
        deriv = torch.sigmoid(1.702 * z) * (1 + 1.702 * z * (1 - torch.sigmoid(1.702 * z)))
        mean_deriv = deriv.mean(dim=0)
        scores[i] = std_h * output_norms * (mean_h.abs() + mean_deriv.abs() + 1e-6)
    return scores


# ── per-channel stats ─────────────────────────────────────────────────────────

def gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def compute_block_stats(z: torch.Tensor, block_id: int, imp_scores: torch.Tensor) -> dict:
    N, C = z.shape
    mu = z.mean(dim=0)      # (C,)
    sigma = z.std(dim=0)    # (C,)

    # Zone fractions
    zone_results = {}
    for T in THRESHOLDS:
        empirical_frac = (z.abs() < T).float().mean(dim=0)  # (C,)
        safe_sigma = sigma.clamp(min=1e-8)
        gauss_frac = gaussian_cdf((T - mu) / safe_sigma) - gaussian_cdf((-T - mu) / safe_sigma)
        emp_mean = empirical_frac.mean().item()
        gauss_mean = gauss_frac.mean().item()
        rel_excess = (emp_mean - gauss_mean) / max(gauss_mean, 1e-8) * 100.0
        zone_results[str(T)] = {
            "empirical": round(emp_mean, 5),
            "gaussian": round(gauss_mean, 5),
            "relative_excess_pct": round(rel_excess, 3),
        }

    # KS statistic on sample of 128 channels
    rng = np.random.default_rng(42)
    sample_ch = rng.choice(C, size=min(128, C), replace=False)
    z_np = z.numpy()
    ks_stats = []
    for j in sample_ch:
        zj = z_np[:, j]
        mj = float(mu[j].item())
        sj = float(sigma[j].item())
        if sj < 1e-8:
            continue
        d, _ = stats.kstest(zj, "norm", args=(mj, sj))
        ks_stats.append(d)
    ks_mean = float(np.mean(ks_stats)) if ks_stats else float("nan")
    ks_max = float(np.max(ks_stats)) if ks_stats else float("nan")

    # σ and μ histograms
    sigma_np = sigma.numpy()
    mu_np = mu.numpy()
    s_counts, s_edges = np.histogram(sigma_np, bins=50)
    m_counts, m_edges = np.histogram(mu_np, bins=50)

    # Bump-dominated channels: σ < 0.5 AND |μ| < 0.5
    bump_mask = (sigma < 0.5) & (mu.abs() < 0.5)
    bump_count = int(bump_mask.sum().item())
    bump_fraction = round(bump_count / C, 4)

    # Spearman ρ(σ_j, importance_j)
    rho, _ = stats.spearmanr(sigma_np, imp_scores.detach().numpy())

    max_rel_excess = max(abs(zone_results[str(T)]["relative_excess_pct"]) for T in THRESHOLDS)

    return {
        "block": block_id,
        "n_tokens": N,
        "n_channels": C,
        "sigma_mean": round(float(sigma_np.mean()), 5),
        "sigma_median": round(float(np.median(sigma_np)), 5),
        "mu_mean": round(float(mu_np.mean()), 5),
        "ks_mean": round(ks_mean, 5),
        "ks_max": round(ks_max, 5),
        "zone_fractions": zone_results,
        "max_relative_excess_pct": round(max_rel_excess, 3),
        "sigma_histogram": {"counts": s_counts.tolist(), "edges": s_edges.tolist()},
        "mu_histogram": {"counts": m_counts.tolist(), "edges": m_edges.tolist()},
        "bump_dominated_count": bump_count,
        "bump_dominated_fraction": bump_fraction,
        "spearman_rho_sigma_importance": round(float(rho), 5),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plot(block_stats: list[dict], spearman_rhos: list[float]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Cycle 108: Pre-activation Distribution Anatomy GPT-2 FFN", fontsize=13)

    blocks = list(range(12))

    # (a) σ mean/median by block
    ax = axes[0, 0]
    sigma_means = [s["sigma_mean"] for s in block_stats]
    sigma_medians = [s["sigma_median"] for s in block_stats]
    ax.plot(blocks, sigma_means, "o-", label="mean σ")
    ax.plot(blocks, sigma_medians, "s--", label="median σ")
    ax.set_xlabel("Block")
    ax.set_ylabel("σ")
    ax.set_title("(a) Per-channel σ by block")
    ax.legend(fontsize=8)
    ax.set_xticks(blocks)

    # (b) Zone fraction comparison at T=0.5
    ax = axes[0, 1]
    T_str = "0.5"
    emp = [s["zone_fractions"][T_str]["empirical"] for s in block_stats]
    gauss = [s["zone_fractions"][T_str]["gaussian"] for s in block_stats]
    ax.plot(blocks, emp, "o-", label="Empirical")
    ax.plot(blocks, gauss, "s--", label="Gaussian")
    ax.set_xlabel("Block")
    ax.set_ylabel("Fraction |z| < 0.5")
    ax.set_title("(b) Zone fractions at T=0.5")
    ax.legend(fontsize=8)
    ax.set_xticks(blocks)

    # (c) Bump-dominated fraction
    ax = axes[1, 0]
    bump_fracs = [s["bump_dominated_fraction"] for s in block_stats]
    ax.bar(blocks, bump_fracs, color="steelblue")
    ax.set_xlabel("Block")
    ax.set_ylabel("Fraction of channels")
    ax.set_title("(c) Bump-dominated channels (σ<0.5, |μ|<0.5)")
    ax.set_xticks(blocks)

    # (d) Spearman ρ
    ax = axes[1, 1]
    ax.bar(blocks, spearman_rhos, color="darkorange")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Block")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("(d) Spearman ρ(σ_j, importance_j)")
    ax.set_xticks(blocks)

    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {OUT_PLOT}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Abort check: PPL
    print("Computing baseline PPL...", flush=True)
    ppl = compute_ppl(teacher)
    print(f"  Baseline PPL = {ppl:.2f}", flush=True)
    if not (25 <= ppl <= 65):
        print(f"ABORT: PPL {ppl:.2f} outside [25, 65]", flush=True)
        sys.exit(1)

    # Collect pre-activations
    print("Collecting pre-activations...", flush=True)
    preacts = collect_preactivations(N_TOKENS)
    for i, z in preacts.items():
        if torch.isnan(z).any():
            print(f"ABORT: NaN in block {i}", flush=True)
            sys.exit(1)
        print(f"  Block {i:2d}: z shape {z.shape}", flush=True)

    # Importance scores (reuse pre-activations — same token set)
    print("Computing importance scores...", flush=True)
    importance = compute_importance_scores(preacts)

    # Per-block stats
    print("Computing per-block stats...", flush=True)
    block_stats = []
    spearman_rhos = []
    for i in range(12):
        s = compute_block_stats(preacts[i], i, importance[i])
        spearman_rhos.append(s["spearman_rho_sigma_importance"])
        print(
            f"  Block {i:2d}: σ_mean={s['sigma_mean']:.3f} "
            f"KS_mean={s['ks_mean']:.4f} "
            f"bump_frac={s['bump_dominated_fraction']:.3f} "
            f"max_excess={s['max_relative_excess_pct']:.2f}% "
            f"ρ={s['spearman_rho_sigma_importance']:.3f}",
            flush=True,
        )
        block_stats.append(s)

    # Verdict
    match_blocks = sum(1 for s in block_stats if s["max_relative_excess_pct"] <= 5.0)
    excess_blocks = sum(1 for s in block_stats if s["max_relative_excess_pct"] > 20.0)
    if match_blocks >= 10:
        verdict = "CONFIRMED"
    elif excess_blocks >= 6:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nVerdict: {verdict} (match_blocks={match_blocks}/12, excess_blocks={excess_blocks}/12)", flush=True)

    result = {
        "cycle": 108,
        "item": 119,
        "baseline_ppl": round(ppl, 3),
        "n_tokens": N_TOKENS,
        "verdict": verdict,
        "match_blocks": match_blocks,
        "excess_blocks": excess_blocks,
        "block_stats": block_stats,
        "spearman_rhos": spearman_rhos,
        "thresholds": THRESHOLDS,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON saved: {OUT_JSON}", flush=True)

    make_plot(block_stats, spearman_rhos)

    return result


if __name__ == "__main__":
    main()
