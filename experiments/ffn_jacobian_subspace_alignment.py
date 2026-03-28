"""
Cycle 069 — Item 81: Shared vs token-specific Jacobian spectral structure.
Measures pairwise subspace alignment of top-k right singular vectors of J(x)
across tokens for each GPT-2 FFN block.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

RESULTS_DIR = Path("autoresearch/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 7]
N_TOKENS = 256
SEQ_LEN = 128
N_SEQUENCES = 2
N_PAIRS = 1000
N_RANDOM_PAIRS = 200
RANK_JSON = RESULTS_DIR / "cycle_068_jacobian_rank_census.json"
OUT_JSON = RESULTS_DIR / "cycle_069_jacobian_subspace_alignment.json"
OUT_PNG = RESULTS_DIR / "cycle_069_jacobian_subspace_alignment.png"
SHAKESPEARE = Path("experiments/shakespeare.txt")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_rank_info():
    with open(RANK_JSON) as f:
        d = json.load(f)
    bs = d["block_stats"]
    k99 = {b: max(1, round(bs[str(b)]["mean_rank_99"])) for b in range(12)}
    k95 = {b: max(1, round(bs[str(b)]["mean_rank_95"])) for b in range(12)}
    return k99, k95


def build_corpus(tokenizer, seed, n_tokens=N_TOKENS, seq_len=SEQ_LEN, n_seq=N_SEQUENCES):
    """Return list of input_ids tensors (n_seq × seq_len) covering ~n_tokens unique tokens."""
    rng = random.Random(seed)
    if SHAKESPEARE.exists():
        text = SHAKESPEARE.read_text(encoding="utf-8", errors="ignore")
    else:
        # fallback: repeat lorem ipsum style
        text = ("The quick brown fox jumps over the lazy dog. " * 500)
    words = text.split()
    rng.shuffle(words)
    joined = " ".join(words)
    ids = tokenizer.encode(joined, add_special_tokens=False)
    # sample n_seq non-overlapping chunks
    chunks = []
    start = 0
    for _ in range(n_seq):
        if start + seq_len > len(ids):
            start = 0
        chunks.append(torch.tensor(ids[start: start + seq_len], dtype=torch.long).unsqueeze(0))
        start += seq_len
    return chunks


def get_ffn_weights(model, block_idx):
    """Return (W_in [3072×768], W_out [768×3072]) for block block_idx."""
    blk = model.h[block_idx]
    # Conv1D: weight.shape = (in_features, out_features) — opposite of nn.Linear
    W_in = blk.mlp.c_fc.weight.T.detach().float()    # (3072, 768)
    W_out = blk.mlp.c_proj.weight.T.detach().float()  # (768, 3072)
    b_in = blk.mlp.c_fc.bias.detach().float()         # (3072,)
    return W_in, W_out, b_in


def collect_token_jacobians(model, tokenizer, block_idx, W_in, W_out, b_in, seeds):
    """
    Return list (over tokens) of Jacobian matrices J in R^{768×768}.
    Actually we want top-k right singular vectors (Vh), so we compute J and SVD.
    J = W_out @ diag(gelu'(z)) @ W_in  shape (768, 768)
    We return the pre-activation z per token so we can build J lazily.
    """
    device = next(model.parameters()).device
    all_z = []  # (3072,) pre-activations per token
    for seed in seeds:
        chunks = build_corpus(tokenizer, seed)
        for chunk in chunks:
            chunk = chunk.to(device)
            with torch.no_grad():
                # collect hidden states at the input to this block
                out = model(chunk, output_hidden_states=True)
                hidden = out.hidden_states[block_idx]  # (1, seq_len, 768)
                h = hidden[0].float()  # (seq_len, 768)
                z = h @ W_in.T + b_in  # (seq_len, 3072)
                all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)  # (total_tokens, 3072)
    return all_z


def gelu_deriv(z: torch.Tensor) -> torch.Tensor:
    """Derivative of GELU element-wise via autograd."""
    z2 = z.detach().requires_grad_(True)
    g = F.gelu(z2)
    sigma_prime = torch.autograd.grad(g.sum(), z2)[0]
    return sigma_prime.detach()


def compute_top_k_Vh(W_in, W_out, z_token, k):
    """
    z_token: (3072,) pre-activation for one token.
    Returns Vh_k: (k, 768) top-k right singular vectors of J(x).
    J = W_out @ diag(sigma_prime) @ W_in  (768 × 768)
    """
    sigma_prime = gelu_deriv(z_token)  # (3072,)
    # Scale rows of W_in by sigma_prime, then multiply W_out
    # J = W_out @ (sigma_prime.unsqueeze(1) * W_in)  — (768, 768)
    scaled_Win = sigma_prime.unsqueeze(1) * W_in  # (3072, 768)
    J = W_out @ scaled_Win  # (768, 768)
    # SVD — J is 768×768, manageable
    try:
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)  # Vh: (768, 768)
    except Exception:
        # fallback: randomized SVD via lowrank
        U, S, Vh_T = torch.svd_lowrank(J, q=min(k + 20, 768))
        Vh = Vh_T.T
    return Vh[:k, :]  # (k, 768)


def subspace_alignment(Vh_i, Vh_j):
    """Mean cosine of principal angles between k-dim subspaces in R^768."""
    M = Vh_i @ Vh_j.T  # (k, k)
    svals = torch.linalg.svdvals(M)  # (k,) singular values = cosines
    return svals.mean().item()


def random_subspace_alignment(k, d=768, n_pairs=N_RANDOM_PAIRS, rng=None):
    """Mean pairwise alignment of random k-dim subspaces in R^d."""
    if rng is None:
        rng = torch.Generator()
    vals = []
    for _ in range(n_pairs):
        A = torch.randn(d, k, generator=rng)
        B = torch.randn(d, k, generator=rng)
        Q_A, _ = torch.linalg.qr(A)
        Q_B, _ = torch.linalg.qr(B)
        M = Q_A.T @ Q_B  # (k, k)
        svals = torch.linalg.svdvals(M)
        vals.append(svals.mean().item())
    mean_r = sum(vals) / len(vals)
    std_r = (sum((v - mean_r) ** 2 for v in vals) / len(vals)) ** 0.5
    return mean_r, std_r


def run_block(model, tokenizer, block_idx, W_in, W_out, b_in, k_99, k_95, seeds):
    """Run alignment analysis for one block, both k settings."""
    print(f"  Block {block_idx}: collecting tokens...", flush=True)
    t0 = time.time()
    all_z = collect_token_jacobians(model, tokenizer, block_idx, W_in, W_out, b_in, seeds)
    n_tok = all_z.shape[0]
    print(f"  Block {block_idx}: {n_tok} tokens collected in {time.time()-t0:.1f}s", flush=True)

    results = {}
    for k_label, k in [("k_99", k_99), ("k_95", k_95)]:
        print(f"    k={k} ({k_label}): computing Vh for {n_tok} tokens...", flush=True)
        t1 = time.time()
        # Compute Vh for each token
        Vhs = []
        for i in range(n_tok):
            z_i = all_z[i]
            Vhs.append(compute_top_k_Vh(W_in, W_out, z_i, k))
        print(f"    SVDs done in {time.time()-t1:.1f}s", flush=True)

        # Sample 1000 pairs
        rng_pairs = random.Random(block_idx * 1000 + k)
        indices = list(range(n_tok))
        pairs = [(rng_pairs.randint(0, n_tok-1), rng_pairs.randint(0, n_tok-1))
                 for _ in range(N_PAIRS)]
        pairs = [(i, j) for i, j in pairs if i != j][:N_PAIRS]

        actual_vals = []
        for i, j in pairs:
            val = subspace_alignment(Vhs[i], Vhs[j])
            actual_vals.append(val)
        actual_mean = sum(actual_vals) / len(actual_vals)
        actual_std = (sum((v - actual_mean) ** 2 for v in actual_vals) / len(actual_vals)) ** 0.5

        # Random baseline
        rng_torch = torch.Generator()
        rng_torch.manual_seed(block_idx * 7 + k)
        rand_mean, rand_std = random_subspace_alignment(k, rng=rng_torch)

        # Normalized excess
        excess = (actual_mean - rand_mean) / (1.0 - rand_mean + 1e-9)
        is_trivial = (actual_mean >= 0.8 and excess < 0.1)

        print(f"    {k_label}: actual={actual_mean:.4f}±{actual_std:.4f}, "
              f"random={rand_mean:.4f}±{rand_std:.4f}, excess={excess:.4f}, "
              f"trivial={is_trivial}", flush=True)

        results[k_label] = {
            "k": k,
            "actual_mean_cos": actual_mean,
            "actual_std_cos": actual_std,
            "random_mean_cos": rand_mean,
            "random_std_cos": rand_std,
            "excess": excess,
            "is_trivial": is_trivial,
            "n_pairs": len(pairs),
        }
    return results


def determine_verdict(block_results):
    """Apply two-layer verdict logic."""
    confirmed_raw = 0
    genuinely_confirmed = 0
    trivially_confirmed = 0
    refuted = 0

    for b in range(12):
        br = block_results[b]
        # Check either k setting passes raw >= 0.8
        raw_pass = any(br[k_lbl]["actual_mean_cos"] >= 0.8 for k_lbl in ["k_95", "k_99"])
        raw_fail = all(br[k_lbl]["actual_mean_cos"] < 0.5 for k_lbl in ["k_95", "k_99"])

        if raw_pass:
            confirmed_raw += 1
            # Check if genuinely confirmed (excess >= 0.1)
            genuine = any(
                br[k_lbl]["actual_mean_cos"] >= 0.8 and br[k_lbl]["excess"] >= 0.1
                for k_lbl in ["k_95", "k_99"]
            )
            if genuine:
                genuinely_confirmed += 1
            else:
                trivially_confirmed += 1
        elif raw_fail:
            refuted += 1

    # Agenda verdict
    if confirmed_raw >= 10:
        agenda_verdict = "Confirmed"
    elif refuted > 6:
        agenda_verdict = "Refuted"
    else:
        agenda_verdict = "Inconclusive"

    # Geometric verdict
    if genuinely_confirmed >= 10:
        geometric_outcome = "confirmed"
    elif confirmed_raw >= 10 and genuinely_confirmed == 0:
        geometric_outcome = "trivially_confirmed"
    elif confirmed_raw >= 10:
        geometric_outcome = "partially_confirmed_partially_trivial"
    else:
        geometric_outcome = "inconclusive"

    return {
        "agenda_verdict": agenda_verdict,
        "geometric_outcome": geometric_outcome,
        "confirmed_raw": confirmed_raw,
        "genuinely_confirmed": genuinely_confirmed,
        "trivially_confirmed": trivially_confirmed,
        "refuted": refuted,
    }


def plot_results(block_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, k_label in enumerate(["k_95", "k_99"]):
        ax = axes[ax_idx]
        blocks = list(range(12))
        actual = [block_results[b][k_label]["actual_mean_cos"] for b in blocks]
        rand = [block_results[b][k_label]["random_mean_cos"] for b in blocks]
        actual_std = [block_results[b][k_label]["actual_std_cos"] for b in blocks]
        trivial = [block_results[b][k_label]["is_trivial"] for b in blocks]

        x = range(12)
        width = 0.35
        bars1 = ax.bar([xi - width/2 for xi in x], actual, width,
                       label="Actual", color="steelblue", alpha=0.8,
                       yerr=actual_std, capsize=3)
        bars2 = ax.bar([xi + width/2 for xi in x], rand, width,
                       label="Random", color="coral", alpha=0.8)

        # Annotate trivial blocks
        for xi, is_t in enumerate(trivial):
            if is_t:
                ax.text(xi - width/2, actual[xi] + 0.02, "T", ha="center",
                        fontsize=8, color="red")

        ax.axhline(0.8, color="black", linestyle="--", linewidth=1, label="Threshold 0.8")
        ax.set_xlabel("GPT-2 Block")
        ax.set_ylabel("Mean Cosine of Principal Angles")
        k_val = block_results[0][k_label]["k"]
        ax.set_title(f"{k_label} (k≈{k_val})")
        ax.set_xticks(list(x))
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("GPT-2 FFN Jacobian Subspace Alignment\n(T = trivially confirmed, excess < 0.1)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading rank info from cycle 068...", flush=True)
    k99_map, k95_map = load_rank_info()
    print(f"k_99 range: {min(k99_map.values())}-{max(k99_map.values())}", flush=True)
    print(f"k_95 range: {min(k95_map.values())}-{max(k95_map.values())}", flush=True)

    print("Loading GPT-2...", flush=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    # Run on CPU (768×768 SVD is manageable)
    device = torch.device("cpu")
    model = model.to(device)

    block_results = {}
    for b in range(12):
        print(f"\n=== Block {b} ===", flush=True)
        W_in, W_out, b_in = get_ffn_weights(model, b)
        k_99 = k99_map[b]
        k_95 = k95_map[b]
        block_results[b] = run_block(
            model, tokenizer, b, W_in, W_out, b_in,
            k_99, k_95, SEEDS
        )

    # Verdict
    verdict = determine_verdict(block_results)
    print(f"\n=== VERDICT ===")
    for key, val in verdict.items():
        print(f"  {key}: {val}")

    # Save JSON
    output = {
        "cycle": 69,
        "item": 81,
        "seeds": SEEDS,
        "n_tokens_per_seed_per_seq": N_TOKENS,
        "seq_len": SEQ_LEN,
        "n_sequences": N_SEQUENCES,
        "block_results": {str(b): block_results[b] for b in range(12)},
        "verdict": verdict,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved to {OUT_JSON}")

    # Plot
    plot_results(block_results, OUT_PNG)

    print("\nDone.")
    return verdict


if __name__ == "__main__":
    main()
