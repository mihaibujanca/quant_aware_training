"""
Cycle 068: FFN Jacobian effective rank census — GPT-2

For each of the 12 FFN blocks, compute the per-token Jacobian
    J(x) = W_out @ diag(GELU'(z)) @ W_in   shape (768, 768)
over N=512 probe tokens × 3 seeds. Report effective rank at 95%/99% energy thresholds.
Correlate with cycle 048 P1 Frobenius std and single-block compression damage.
"""
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
)

OUTPUT_DIR = ROOT / "autoresearch" / "results"
DATA_PATH = ROOT / "data" / "shakespeare.txt"

SEEDS = [42, 123, 7]
N_TOKENS = 512  # probe tokens per seed
SEQ_LEN = 128   # token window length
N_SEQUENCES = N_TOKENS // SEQ_LEN  # 4 sequences × 128 = 512 tokens
N_BLOCKS = 12
DEVICE = "cpu"

# ---------------------------------------------------------------------------
# GELU derivative (GPT-2 tanh approximation)
# ---------------------------------------------------------------------------
def gelu_grad(z: torch.Tensor) -> torch.Tensor:
    """Element-wise derivative of GPT-2 GELU (tanh approximation)."""
    k = math.sqrt(2.0 / math.pi)
    t = torch.tanh(k * (z + 0.044715 * z ** 3))
    dt = 1.0 - t ** 2
    inner_deriv = k * (1.0 + 3.0 * 0.044715 * z ** 2)
    return 0.5 * (1.0 + t) + 0.5 * z * dt * inner_deriv


# ---------------------------------------------------------------------------
# Load GPT-2 model (shared across seeds, weights are fixed)
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
t0 = time.time()
teacher = load_nanochat_teacher(TeacherLoadSpec(
    source="hf_pretrained",
    hf_model_name="gpt2",
    device=DEVICE,
))
model = teacher.model
tokenizer = teacher.tokenizer
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s")

# Extract and validate weight matrices (shared across all seeds)
W_ins = []   # W_in[b] shape (3072, 768)
W_outs = []  # W_out[b] shape (768, 3072)
for b in range(N_BLOCKS):
    c_fc = model.transformer.h[b].mlp.c_fc
    c_proj = model.transformer.h[b].mlp.c_proj
    # Conv1D: c_fc.weight.shape = (768, 3072), so W_in = c_fc.weight.T
    W_in = c_fc.weight.T.detach().float()   # (3072, 768)
    W_out = c_proj.weight.T.detach().float()  # (768, 3072)
    assert W_in.shape == (3072, 768), f"Block {b}: W_in shape {W_in.shape} != (3072, 768)"
    assert W_out.shape == (768, 3072), f"Block {b}: W_out shape {W_out.shape} != (768, 3072)"
    W_ins.append(W_in)
    W_outs.append(W_out)
print(f"  Weight shapes validated: W_in={W_ins[0].shape}, W_out={W_outs[0].shape}")


# ---------------------------------------------------------------------------
# Helper: effective rank from singular values
# ---------------------------------------------------------------------------
def effective_rank(s: torch.Tensor, threshold: float) -> int:
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    idx = (energy >= threshold).nonzero(as_tuple=False)
    if len(idx) == 0:
        return len(s)
    return int(idx[0, 0].item()) + 1


# ---------------------------------------------------------------------------
# Random matrix control: rank_99 should be >= 750
# ---------------------------------------------------------------------------
print("Running random matrix control...")
rng_ctrl = torch.Generator()
rng_ctrl.manual_seed(0)
ctrl_ranks = []
for _ in range(10):
    R = torch.randn(768, 768, generator=rng_ctrl)
    s = torch.linalg.svdvals(R)
    ctrl_ranks.append(effective_rank(s, 0.99))
ctrl_ranks_arr = ctrl_ranks
print(f"  Random matrix rank_99 (768x768): min={min(ctrl_ranks)}, max={max(ctrl_ranks)}, mean={sum(ctrl_ranks)/len(ctrl_ranks):.1f}")
# Random 768x768 Gaussian matrix rank_99 ~ 592 (Marchenko-Pastur, not ~768).
# Assert >= 550 to confirm SVD/energy logic is working correctly.
assert min(ctrl_ranks) >= 550, f"Control FAILED: min rank_99={min(ctrl_ranks)} < 550"
print(f"  Control PASSED (random matrix rank_99 ~ 592, threshold 550)")


# ---------------------------------------------------------------------------
# Main loop: 3 seeds × 12 blocks × N_TOKENS token positions
# ---------------------------------------------------------------------------
# per_seed_results[seed_idx][block_idx] = {"rank_95": [...], "rank_99": [...]}
per_seed_results = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")
    torch.manual_seed(seed)

    # Build probe corpus — each seed uses a different shuffle
    corpus = build_probe_corpus(ProbeCorpusSpec(
        text_path=str(DATA_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=N_SEQUENCES,
        probe_holdout_sequences=1,
        tokenizer=tokenizer,
        shuffle_seed=seed,
        dataset_name="shakespeare",
    ))
    probe_tokens = corpus.probe_train  # (N_SEQUENCES, SEQ_LEN)
    actual_tokens = probe_tokens.numel()
    print(f"  probe_tokens shape: {probe_tokens.shape}, total tokens: {actual_tokens}")

    # Collect pre-GELU activations z via hooks
    ffn_z: dict[int, list] = {b: [] for b in range(N_BLOCKS)}
    hooks = []

    def make_hook(block_idx):
        def hook(module, input, output):
            ffn_z[block_idx].append(output.detach().cpu().float())
        return hook

    for b in range(N_BLOCKS):
        h = model.transformer.h[b].mlp.c_fc.register_forward_hook(make_hook(b))
        hooks.append(h)

    with torch.no_grad():
        for seq_idx in range(probe_tokens.shape[0]):
            inp = probe_tokens[seq_idx:seq_idx+1].to(DEVICE)
            model(inp, use_cache=False)

    for h in hooks:
        h.remove()

    # Concatenate: z shape (T_total, 3072)
    for b in range(N_BLOCKS):
        z_cat = torch.cat(ffn_z[b], dim=0)  # may be (N_SEQ, SEQ_LEN, 3072) or (T, 3072)
        if z_cat.dim() == 3:
            z_cat = z_cat.reshape(-1, z_cat.shape[-1])
        ffn_z[b] = z_cat  # (T, 3072)

    T = ffn_z[0].shape[0]
    print(f"  T (total tokens in z): {T}")
    assert ffn_z[0].shape[1] == 3072, f"Unexpected hidden size: {ffn_z[0].shape[1]}"

    seed_results = {b: {"rank_95": [], "rank_99": []} for b in range(N_BLOCKS)}

    for b in range(N_BLOCKS):
        z_all = ffn_z[b]  # (T, 3072)
        W_in = W_ins[b]   # (3072, 768)
        W_out = W_outs[b]  # (768, 3072)

        for t_idx in range(T):
            z_t = z_all[t_idx]          # (3072,)
            sigma_prime = gelu_grad(z_t)  # (3072,)

            # J = W_out @ diag(sigma_prime) @ W_in
            # = (W_out * sigma_prime.unsqueeze(0)) @ W_in
            # shape: (768, 3072) * (1, 3072) -> (768, 3072) @ (3072, 768) -> (768, 768)
            J = (W_out * sigma_prime.unsqueeze(0)) @ W_in  # (768, 768)
            assert J.shape == (768, 768), f"J shape {J.shape} != (768, 768)"

            s = torch.linalg.svdvals(J)
            assert not torch.isnan(s).any(), f"NaN in SVD at block {b}, token {t_idx}"

            seed_results[b]["rank_95"].append(effective_rank(s, 0.95))
            seed_results[b]["rank_99"].append(effective_rank(s, 0.99))

        r95 = seed_results[b]["rank_95"]
        r99 = seed_results[b]["rank_99"]
        print(f"  Block {b:2d}: rank_95 mean={sum(r95)/len(r95):.1f}, rank_99 mean={sum(r99)/len(r99):.1f}  ({T} tokens)")

    per_seed_results.append(seed_results)
    print(f"  Seed {seed} done.")


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------
print("\nAggregating results...")
block_stats = {}
for b in range(N_BLOCKS):
    all_r95 = []
    all_r99 = []
    seed_means_r99 = []
    for seed_idx in range(len(SEEDS)):
        r95 = per_seed_results[seed_idx][b]["rank_95"]
        r99 = per_seed_results[seed_idx][b]["rank_99"]
        all_r95.extend(r95)
        all_r99.extend(r99)
        seed_means_r99.append(sum(r99) / len(r99))

    block_stats[b] = {
        "mean_rank_95": float(sum(all_r95) / len(all_r95)),
        "mean_rank_99": float(sum(all_r99) / len(all_r99)),
        "std_rank_95": float(torch.tensor(all_r95, dtype=torch.float).std().item()),
        "std_rank_99": float(torch.tensor(all_r99, dtype=torch.float).std().item()),
        "min_rank_99": int(min(all_r99)),
        "max_rank_99": int(max(all_r99)),
        "seed_means_rank_99": seed_means_r99,
        # Cross-seed consistency: std of per-seed means
        "cross_seed_std_rank_99": float(torch.tensor(seed_means_r99).std().item()),
    }
    print(f"  Block {b:2d}: rank_99 mean={block_stats[b]['mean_rank_99']:.1f} ± {block_stats[b]['std_rank_99']:.1f}  (cross-seed std={block_stats[b]['cross_seed_std_rank_99']:.2f})")

# Cross-seed consistency check
for b in range(N_BLOCKS):
    mean_r99 = block_stats[b]["mean_rank_99"]
    cs_std = block_stats[b]["cross_seed_std_rank_99"]
    if mean_r99 > 0 and cs_std / mean_r99 > 0.20:
        print(f"  WARNING: Block {b} cross-seed std {cs_std:.2f} > 20% of mean {mean_r99:.1f}")


# ---------------------------------------------------------------------------
# Correlations with cycle 048 data
# ---------------------------------------------------------------------------
print("\nComputing correlations with cycle 048...")
with open(ROOT / "autoresearch" / "results" / "cycle_048_local_complexity_gpt2.json") as f:
    c048 = json.load(f)

p1_frob_std = [c048["proxies_by_layer"][str(b)]["p1_frob_std"] for b in range(N_BLOCKS)]
damage_8x = c048["damage_8x"]  # list of 12 floats

mean_rank_99 = [block_stats[b]["mean_rank_99"] for b in range(N_BLOCKS)]

rho_p1, pval_p1 = spearmanr(mean_rank_99, p1_frob_std)
rho_dmg, pval_dmg = spearmanr(mean_rank_99, damage_8x)

print(f"  Spearman(rank_99, P1_frob_std): rho={rho_p1:.3f}, p={pval_p1:.4f}")
print(f"  Spearman(rank_99, damage_8x):   rho={rho_dmg:.3f}, p={pval_dmg:.4f}")


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------
n_low_rank = sum(1 for b in range(N_BLOCKS) if block_stats[b]["mean_rank_99"] <= 500)
# Random 768x768 Gaussian matrix has rank_99 ~ 592 (Marchenko-Pastur).
# "Refuted" threshold of 700 in plan exceeds 592, so we use 580 (just below random baseline)
n_above_random = sum(1 for b in range(N_BLOCKS) if block_stats[b]["mean_rank_99"] > 580)
print(f"\n  Blocks with rank_99 <= 500: {n_low_rank}/12")
print(f"  Blocks with rank_99 > 580 (above random baseline ~592): {n_above_random}/12")
print(f"  Blocks with rank_99 > 700: {sum(1 for b in range(N_BLOCKS) if block_stats[b]['mean_rank_99'] > 700)}/12")

if n_low_rank >= 10 and rho_p1 >= 0.5:
    outcome = "confirmed"
    outcome_reason = (
        f"{n_low_rank}/12 blocks have rank_99 <= 500 AND "
        f"Spearman(rank_99, P1_frob_std)={rho_p1:.3f} >= 0.5"
    )
elif n_above_random > 6:
    outcome = "refuted"
    outcome_reason = (
        f"Majority of blocks ({n_above_random}/12) have rank_99 > 580 "
        f"(at or above random matrix baseline ~592 — FFN not meaningfully low-rank)"
    )
else:
    outcome = "inconclusive"
    outcome_reason = (
        f"{n_low_rank}/12 blocks low-rank OR "
        f"Spearman(rank_99, P1_frob_std)={rho_p1:.3f} below 0.5 threshold"
    )

print(f"\n  OUTCOME: {outcome.upper()} — {outcome_reason}")


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
result = {
    "cycle": 68,
    "item": 80,
    "seeds": SEEDS,
    "n_tokens_per_seed": T,
    "block_stats": {str(b): block_stats[b] for b in range(N_BLOCKS)},
    "correlations": {
        "spearman_rank99_vs_p1_frob_std": {"rho": float(rho_p1), "pval": float(pval_p1)},
        "spearman_rank99_vs_damage_8x": {"rho": float(rho_dmg), "pval": float(pval_dmg)},
    },
    "reference_p1_frob_std": p1_frob_std,
    "reference_damage_8x": damage_8x,
    "random_matrix_control": {
        "n_matrices": 10,
        "matrix_size": "768x768",
        "min_rank_99": int(min(ctrl_ranks_arr)),
        "max_rank_99": int(max(ctrl_ranks_arr)),
        "all_ranks": ctrl_ranks_arr,
    },
    "outcome": outcome,
    "outcome_reason": outcome_reason,
}

json_path = OUTPUT_DIR / "cycle_068_jacobian_rank_census.json"
with open(json_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved JSON: {json_path}")


# ---------------------------------------------------------------------------
# Plot: bar chart of mean rank_99 with std error bars, overlay P1 frob std
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 5))

blocks = list(range(N_BLOCKS))
means_99 = [block_stats[b]["mean_rank_99"] for b in blocks]
stds_99 = [block_stats[b]["std_rank_99"] for b in blocks]
means_95 = [block_stats[b]["mean_rank_95"] for b in blocks]

bars = ax1.bar(blocks, means_99, yerr=stds_99, capsize=4,
               color="steelblue", alpha=0.7, label="rank_99 mean ± std")
ax1.bar(blocks, means_95, color="cornflowerblue", alpha=0.5, label="rank_95 mean")
ax1.axhline(500, color="red", linestyle="--", lw=1, label="threshold 500")
ax1.axhline(700, color="orange", linestyle="--", lw=1, label="threshold 700")
ax1.set_xlabel("FFN Block")
ax1.set_ylabel("Effective Rank (768×768 Jacobian)")
ax1.set_ylim(0, 800)
ax1.set_xticks(blocks)
ax1.legend(loc="upper left", fontsize=8)

ax2 = ax1.twinx()
ax2.plot(blocks, p1_frob_std, "D--", color="darkorange", lw=1.5,
         markersize=5, label="P1 Frob std (C048)")
ax2.set_ylabel("P1 Frobenius std", color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")
ax2.legend(loc="upper right", fontsize=8)

title = (
    f"GPT-2 FFN Jacobian Effective Rank — Cycle 068\n"
    f"Spearman(rank_99, P1_frob_std): rho={rho_p1:.3f} p={pval_p1:.4f}  |  "
    f"Spearman(rank_99, damage_8x): rho={rho_dmg:.3f} p={pval_dmg:.4f}  |  "
    f"Outcome: {outcome}"
)
plt.title(title, fontsize=9)
plt.tight_layout()

plot_path = OUTPUT_DIR / "cycle_068_jacobian_rank_census.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved plot: {plot_path}")

print("\nDone.")
