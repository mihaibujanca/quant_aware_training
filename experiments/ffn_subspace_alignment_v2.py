"""
Cycle 079: FFN subspace alignment v2 — GPT-2 (bug-fixed hooks)

Fix from cycle 069: use register_forward_hook on mlp.c_fc to capture
the true FFN input (post-attention + ln_2), not block-level hidden states.

k=384, 256 tokens per seed, 3 seeds [42, 123, 7].
Per token: truncated SVD → top-384 right singular vectors.
Alignment: sample 1000 random pairs, mean svdvals(Vh_i @ Vh_j.T).
"""
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "autoresearch" / "results"
DATA_PATH = ROOT / "data" / "shakespeare.txt"

SEEDS = [42, 123, 7]
N_TOKENS = 256       # tokens per seed
SEQ_LEN = 128        # token window length
N_SEQUENCES = N_TOKENS // SEQ_LEN  # 2 sequences × 128 = 256 tokens
N_BLOCKS = 12
K = 384              # top-k subspace dimension
DEVICE = "cpu"

N_PAIRS = 1000       # pairwise alignment samples
N_RANDOM_PAIRS = 200 # random baseline pairs

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
t0 = time.time()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s")

# Extract weight matrices
W_ins = []   # (3072, 768)
W_outs = []  # (768, 3072)
b_ins = []   # (3072,)
for b in range(N_BLOCKS):
    c_fc = model.h[b].mlp.c_fc
    c_proj = model.h[b].mlp.c_proj
    # Conv1D: weight.shape = (in_features, out_features) = (768, 3072)
    # So W_in = weight.T has shape (3072, 768)
    W_in = c_fc.weight.T.detach().float()    # (3072, 768)
    W_out = c_proj.weight.T.detach().float() # (768, 3072)
    b_in = c_fc.bias.detach().float()        # (3072,)
    assert W_in.shape == (3072, 768), f"Block {b}: W_in {W_in.shape}"
    assert W_out.shape == (768, 3072), f"Block {b}: W_out {W_out.shape}"
    W_ins.append(W_in)
    W_outs.append(W_out)
    b_ins.append(b_in)
print(f"  W_in={W_ins[0].shape}, W_out={W_outs[0].shape}")


# ---------------------------------------------------------------------------
# Tokenize Shakespeare corpus helper
# ---------------------------------------------------------------------------
def load_sequences(seed: int) -> torch.Tensor:
    """Return (N_SEQUENCES, SEQ_LEN) token tensor from Shakespeare."""
    text = DATA_PATH.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    rng = torch.Generator()
    rng.manual_seed(seed)
    total = len(ids) - SEQ_LEN
    starts = torch.randperm(total, generator=rng)[:N_SEQUENCES].tolist()
    seqs = [ids[s:s+SEQ_LEN] for s in starts]
    return torch.tensor(seqs, dtype=torch.long)  # (N_SEQUENCES, SEQ_LEN)


# ---------------------------------------------------------------------------
# Alignment computation
# ---------------------------------------------------------------------------
def subspace_alignment(Vh_list: list, n_pairs: int, rng: torch.Generator) -> tuple:
    """Mean cosine-like alignment over n_pairs random pairs.

    Vh_list: list of (K, 768) tensors (one per token).
    Returns (mean, std) over pairs.
    """
    N = len(Vh_list)
    # Stack for fast indexing
    Vh_stack = torch.stack(Vh_list, dim=0)  # (N, K, 768)

    scores = []
    for _ in range(n_pairs):
        i, j = torch.randint(N, (2,), generator=rng).tolist()
        while j == i:
            j = torch.randint(N, (1,), generator=rng).item()
        # svdvals of (K×768) @ (768×K) = (K×K)
        M = Vh_stack[i] @ Vh_stack[j].T  # (K, K)
        sv = torch.linalg.svdvals(M)      # (K,)
        scores.append(sv.mean().item())

    t = torch.tensor(scores)
    return float(t.mean()), float(t.std())


def random_baseline_alignment(k: int, d: int, n_pairs: int, rng: torch.Generator) -> tuple:
    """Random baseline: k-dim subspaces in R^d."""
    scores = []
    for _ in range(n_pairs):
        A = torch.randn(d, k, generator=rng)
        B = torch.randn(d, k, generator=rng)
        Qa, _ = torch.linalg.qr(A)  # (d, k)
        Qb, _ = torch.linalg.qr(B)
        M = Qa.T @ Qb  # (k, k)
        sv = torch.linalg.svdvals(M)
        scores.append(sv.mean().item())
    t = torch.tensor(scores)
    return float(t.mean()), float(t.std())


# ---------------------------------------------------------------------------
# Main loop: 3 seeds × 12 blocks
# ---------------------------------------------------------------------------
# per_seed_Vhs[seed_idx][block_idx] = list of (K, 768) tensors (256 tokens)
per_seed_Vhs = []

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")
    torch.manual_seed(seed)

    # Load sequences
    sequences = load_sequences(seed)  # (N_SEQUENCES, SEQ_LEN)
    print(f"  sequences shape: {sequences.shape}")

    # Set up hooks: capture ffn INPUT (post-attention + ln_2) via c_fc input
    ffn_inputs: dict[int, list] = {b: [] for b in range(N_BLOCKS)}
    hooks = []

    def make_hook(block_idx):
        def hook(module, input, output):
            # input[0] is the true FFN input: post-attention, post-ln_2
            ffn_inputs[block_idx].append(input[0].detach().cpu().float())
        return hook

    for b in range(N_BLOCKS):
        h = model.h[b].mlp.c_fc.register_forward_hook(make_hook(b))
        hooks.append(h)

    with torch.no_grad():
        for seq_idx in range(sequences.shape[0]):
            inp = sequences[seq_idx:seq_idx+1].to(DEVICE)
            model(inp, use_cache=False)

    for h in hooks:
        h.remove()

    # Concatenate: ffn_input shape (T, 768)
    ffn_input_all: dict[int, torch.Tensor] = {}
    for b in range(N_BLOCKS):
        cat = torch.cat(ffn_inputs[b], dim=0)  # could be (N_SEQ, SEQ_LEN, 768)
        if cat.dim() == 3:
            cat = cat.reshape(-1, cat.shape[-1])
        ffn_input_all[b] = cat  # (T, 768)
    T = ffn_input_all[0].shape[0]
    print(f"  T (tokens per block): {T}, hidden dim: {ffn_input_all[0].shape[1]}")
    assert ffn_input_all[0].shape[1] == 768, f"Unexpected hidden dim: {ffn_input_all[0].shape[1]}"

    # Compute Jacobians and truncated SVD per block
    seed_Vhs = {}
    for b in range(N_BLOCKS):
        h_all = ffn_input_all[b]  # (T, 768)
        W_in = W_ins[b]           # (3072, 768)
        W_out = W_outs[b]         # (768, 3072)
        b_in = b_ins[b]           # (3072,)

        Vh_list = []
        t_block = time.time()
        for t_idx in range(T):
            h_t = h_all[t_idx]           # (768,)
            z = W_in @ h_t + b_in        # (3072,) — this is c_fc(h_t)

            # GELU derivative via autograd
            z_req = z.clone().requires_grad_(True)
            g = F.gelu(z_req, approximate='tanh')
            sigma_prime = torch.autograd.grad(g.sum(), z_req)[0]  # (3072,)

            # J = W_out @ diag(sigma_prime) @ W_in — (768, 768)
            J = (W_out * sigma_prime.unsqueeze(0)) @ W_in  # (768, 768)

            # Truncated SVD: top-K right singular vectors
            # Use svd_lowrank for speed (K+20 oversampling)
            _, _, Vh = torch.svd_lowrank(J, q=K + 20)
            # svd_lowrank returns V (768, q), so Vh = V.T[:K]
            Vh_k = Vh.T[:K]  # (K, 768)
            Vh_list.append(Vh_k)

        elapsed = time.time() - t_block
        print(f"  Block {b:2d}: {T} Jacobians+SVD in {elapsed:.1f}s")
        seed_Vhs[b] = Vh_list

    per_seed_Vhs.append(seed_Vhs)


# ---------------------------------------------------------------------------
# Alignment computation per block, aggregated across seeds
# ---------------------------------------------------------------------------
print("\nComputing pairwise alignment...")

block_stats = {}
rng_align = torch.Generator()
rng_align.manual_seed(999)

# Compute random baseline once (shared across blocks and seeds)
print("  Computing random baseline (k=384, d=768, 200 pairs)...")
random_mean, random_std = random_baseline_alignment(K, 768, N_RANDOM_PAIRS, rng_align)
print(f"  Random baseline: mean={random_mean:.4f}, std={random_std:.4f}")
assert random_mean < 0.8, f"Random baseline {random_mean:.4f} >= 0.8 — sanity check failed"

for b in range(N_BLOCKS):
    seed_means = []
    seed_stds = []
    for seed_idx in range(len(SEEDS)):
        Vh_list = per_seed_Vhs[seed_idx][b]
        m, s = subspace_alignment(Vh_list, N_PAIRS, rng_align)
        seed_means.append(m)
        seed_stds.append(s)

    actual_mean = float(torch.tensor(seed_means).mean())
    actual_std = float(torch.tensor(seed_means).std())
    cross_seed_std = actual_std

    # Normalized excess
    excess = (actual_mean - random_mean) / (1.0 - random_mean)

    block_stats[b] = {
        "actual_mean_cos": actual_mean,
        "actual_std": actual_std,
        "cross_seed_std": cross_seed_std,
        "per_seed_means": seed_means,
        "random_mean_cos": random_mean,
        "random_std": random_std,
        "excess": excess,
        "is_trivial": excess < 0.1,  # below 0.1 = trivially confirmed or not confirmed
    }
    flag = "TRIVIAL" if excess < 0.1 else ("HIGH" if actual_mean >= 0.8 else "LOW")
    print(f"  Block {b:2d}: actual={actual_mean:.4f}±{cross_seed_std:.4f}  random={random_mean:.4f}  excess={excess:.4f}  [{flag}]")


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
for b in range(N_BLOCKS):
    st = block_stats[b]
    if st["actual_mean_cos"] < st["random_mean_cos"]:
        print(f"  ERROR: Block {b} actual < random — possible bug!")
    if st["cross_seed_std"] > 0.05:
        print(f"  WARNING: Block {b} cross-seed std {st['cross_seed_std']:.4f} > 0.05")

# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
n_high = sum(1 for b in range(N_BLOCKS) if block_stats[b]["actual_mean_cos"] >= 0.8)
n_genuinely_confirmed = sum(
    1 for b in range(N_BLOCKS)
    if block_stats[b]["actual_mean_cos"] >= 0.8 and not block_stats[b]["is_trivial"]
)
n_trivially_confirmed = sum(
    1 for b in range(N_BLOCKS)
    if block_stats[b]["actual_mean_cos"] >= 0.8 and block_stats[b]["is_trivial"]
)
n_low = sum(1 for b in range(N_BLOCKS) if block_stats[b]["actual_mean_cos"] < 0.5)

print(f"\n  Blocks with actual >= 0.8: {n_high}/12")
print(f"  Genuinely confirmed (>= 0.8 AND excess >= 0.1): {n_genuinely_confirmed}/12")
print(f"  Trivially confirmed (>= 0.8 but excess < 0.1): {n_trivially_confirmed}/12")
print(f"  Blocks with actual < 0.5: {n_low}/12")

# Agenda criterion: confirmed if >= 10/12 blocks >= 0.8
if n_high >= 10:
    if n_genuinely_confirmed >= 10:
        outcome = "confirmed"
        outcome_reason = (
            f"{n_genuinely_confirmed}/12 blocks genuinely confirmed "
            f"(actual >= 0.8 AND excess >= 0.1) — low-rank basis globally shared"
        )
    elif n_high >= 10:
        outcome = "trivially_confirmed"
        outcome_reason = (
            f"{n_high}/12 blocks with actual >= 0.8, but only {n_genuinely_confirmed} genuinely confirmed "
            f"(excess >= 0.1); {n_trivially_confirmed} trivially confirmed (excess < 0.1)"
        )
elif n_low > 6:
    outcome = "refuted"
    outcome_reason = (
        f"Majority ({n_low}/12) blocks have actual < 0.5 — "
        f"subspace alignment is too low for globally shared structure"
    )
else:
    outcome = "inconclusive"
    outcome_reason = (
        f"{n_high}/12 blocks >= 0.8 (need >= 10); {n_low}/12 < 0.5"
    )

print(f"\n  OUTCOME: {outcome.upper()} — {outcome_reason}")


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
result = {
    "cycle": 79,
    "item": 91,
    "seeds": SEEDS,
    "n_tokens_per_seed": N_TOKENS,
    "k": K,
    "random_baseline": {"mean": random_mean, "std": random_std},
    "block_stats": {str(b): block_stats[b] for b in range(N_BLOCKS)},
    "summary": {
        "n_high": n_high,
        "n_genuinely_confirmed": n_genuinely_confirmed,
        "n_trivially_confirmed": n_trivially_confirmed,
        "n_low": n_low,
    },
    "outcome": outcome,
    "outcome_reason": outcome_reason,
}

json_path = OUTPUT_DIR / "cycle_079_subspace_alignment.json"
with open(json_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved JSON: {json_path}")


# ---------------------------------------------------------------------------
# Plot: actual vs random per block, with excess annotations
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

blocks = list(range(N_BLOCKS))
actuals = [block_stats[b]["actual_mean_cos"] for b in blocks]
stds = [block_stats[b]["cross_seed_std"] for b in blocks]
randoms = [block_stats[b]["random_mean_cos"] for b in blocks]
excesses = [block_stats[b]["excess"] for b in blocks]

x = torch.arange(N_BLOCKS).float()
width = 0.35
ax.bar(x - width/2, actuals, width, yerr=stds, capsize=4,
       color="steelblue", alpha=0.8, label="Actual alignment (mean ± cross-seed std)")
ax.bar(x + width/2, randoms, width,
       color="gray", alpha=0.5, label=f"Random baseline (k={K}, d=768)")

ax.axhline(0.8, color="red", linestyle="--", lw=1.5, label="Threshold 0.8")

# Annotate excess
for b_idx in range(N_BLOCKS):
    ax.text(b_idx - width/2, actuals[b_idx] + stds[b_idx] + 0.005,
            f"e={excesses[b_idx]:.2f}", ha="center", va="bottom", fontsize=7,
            color="navy")

ax.set_xlabel("FFN Block")
ax.set_ylabel("Mean subspace alignment (mean svdvals)")
ax.set_ylim(0, 1.05)
ax.set_xticks(list(range(N_BLOCKS)))
ax.legend(fontsize=8)

title = (
    f"GPT-2 FFN Subspace Alignment — Cycle 079 (v2, correct hooks)\n"
    f"k={K}, d=768, seeds={SEEDS}, random={random_mean:.3f}  |  "
    f"Outcome: {outcome} ({n_high}/12 blocks >= 0.8, {n_genuinely_confirmed} genuine)"
)
plt.title(title, fontsize=9)
plt.tight_layout()

plot_path = OUTPUT_DIR / "cycle_079_subspace_alignment.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Saved plot: {plot_path}")

print("\nDone.")
