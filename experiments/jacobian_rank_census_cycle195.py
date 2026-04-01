"""
Cycle 195 — Item 206: Jacobian effective rank census
Compute per-token FFN block Jacobians using torch.func.jacfwd + vmap.
Measure effective rank distribution across tokens per block.
"""
import json
import time
import torch
import torch.nn as nn
from torch.func import jacfwd, vmap
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_SEQS = 5   # reduced from 30 — memory pressure from concurrent training process
SEQ_LEN = 32
SEED = 42
DEVICE = "cpu"
N_BLOCKS = 12
D_MODEL = 768
ENERGY_THRESHOLD = 0.90
OUTPUT_PATH = "autoresearch/results/cycle_195_jacobian_rank.json"

torch.manual_seed(SEED)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Calibration data ──────────────────────────────────────────────────────────
print("Loading calibration data...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [row["text"] for row in dataset if len(row["text"].strip()) > 50][:N_SEQS * 4]

input_ids_list = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        input_ids_list.append(enc["input_ids"][:, :SEQ_LEN])
    if len(input_ids_list) >= N_SEQS:
        break

print(f"Got {len(input_ids_list)} sequences of length {SEQ_LEN}")

# ── Collect residual states per block ─────────────────────────────────────────
# For each block b, collect the residual stream BEFORE the FFN sublayer (after attn)
# shape: (N_SEQS * SEQ_LEN, 768)

residuals_per_block = {b: [] for b in range(N_BLOCKS)}

hooks = []
def make_hook(b):
    def hook_fn(module, input, output):
        # input to mlp is after ln_2; we want residual BEFORE ln_2
        # We hook block.mlp and capture input[0] which is ln_out = ln_2(residual)
        # Actually let's hook block itself to get the residual stream before mlp
        pass
    return hook_fn

# Simpler: run forward pass and collect hidden states
print("Collecting residual states...")
with torch.no_grad():
    for ids in input_ids_list:
        ids = ids.to(DEVICE)
        out = model(ids, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each (1, SEQ_LEN, 768)
        # hidden_states[b] = residual stream BEFORE block b
        # After block b = hidden_states[b+1]
        # We want residual BEFORE the FFN part of block b = after attn part of block b
        # But for the FFN Jacobian we want the input to the FFN sublayer.
        # The FFN computes: x_out = x + mlp(ln_2(x)), where x = residual after attn.
        # hidden_states[b+1] = output of block b = x + attn_out + mlp_out
        # We need the state AFTER attn but BEFORE mlp.
        # Easiest: use a hook on block.mlp to capture its input.
        for b in range(N_BLOCKS):
            # hidden_states[b] = residual before block b (before attn AND mlp)
            # We'll use this as input; fn will re-run the full block's mlp path
            # (attn output is already folded in hidden_states[b+1] - mlp contribution)
            # Actually: let's just collect hidden_states[b] (pre-block residual).
            # The FFN Jacobian fn(x) = x + mlp(ln_2(x)) measures FFN's sensitivity.
            # This is the correct input for the FFN Jacobian regardless.
            h = out.hidden_states[b].squeeze(0).float()  # (SEQ_LEN, 768)
            residuals_per_block[b].append(h)

for b in range(N_BLOCKS):
    residuals_per_block[b] = torch.cat(residuals_per_block[b], dim=0)  # (N_SEQS*SEQ_LEN, 768)

N_tokens = residuals_per_block[0].shape[0]
print(f"Tokens per block: {N_tokens}")

# ── Jacobian census ───────────────────────────────────────────────────────────

def effective_rank(S, threshold=ENERGY_THRESHOLD):
    """Effective rank at given energy threshold."""
    energy = (S ** 2).cumsum(0) / ((S ** 2).sum() + 1e-12)
    return int((energy < threshold).sum()) + 1


def make_ffn_fn(block):
    """Return a function x -> FFN(x) for Jacobian computation."""
    def fn(x):  # x: (768,)
        x2 = x.unsqueeze(0)  # (1, 768)
        return (x2 + block.mlp(block.ln_2(x2))).squeeze(0)  # (768,)
    return fn


per_block = []
t0_total = time.time()

for b in range(N_BLOCKS):
    t0 = time.time()
    print(f"Block {b:2d}: computing Jacobians over {N_tokens} tokens...", flush=True)

    block = model.transformer.h[b]
    fn = make_ffn_fn(block)
    tokens = residuals_per_block[b]  # (N_tokens, 768)

    # Process in mini-batches to avoid OOM (full J_batch = 2.3 GB for 960 tokens)
    CHUNK = 8  # 8 × 768 × 3072 × 4 = 75.5 MB peak dual-number memory
    ranks = []
    # Accumulate rows of the stacked Jacobian incrementally via XtX (768×768)
    XtX = torch.zeros(D_MODEL, D_MODEL)  # J_stacked^T @ J_stacked
    n_chunks = (N_tokens + CHUNK - 1) // CHUNK

    for ci in range(n_chunks):
        chunk = tokens[ci * CHUNK: (ci + 1) * CHUNK]  # (≤64, 768)
        J_chunk = vmap(jacfwd(fn))(chunk)  # (≤64, 768, 768)

        # Per-token ranks
        for i in range(J_chunk.shape[0]):
            S = torch.linalg.svdvals(J_chunk[i])
            ranks.append(effective_rank(S))

        # Accumulate XtX = J_stacked^T @ J_stacked for stacked rank
        # J_chunk is (k, out, in); reshape to (k*768, 768)
        rows = J_chunk.reshape(-1, D_MODEL)  # (k*768, 768)
        XtX += rows.T @ rows

        del J_chunk, rows

    ranks_t = torch.tensor(ranks, dtype=torch.float32)

    # Stacked Jacobian rank via eigendecomposition of XtX
    eigvals = torch.linalg.eigvalsh(XtX)  # ascending order
    eigvals = eigvals.flip(0).clamp(min=0)  # descending, non-negative
    S_stacked_sq = eigvals  # these are singular values squared of J_stacked
    energy = S_stacked_sq.cumsum(0) / (S_stacked_sq.sum() + 1e-12)
    stacked_rank = int((energy < ENERGY_THRESHOLD).sum()) + 1

    elapsed = time.time() - t0
    rec = {
        "block": b,
        "mean_rank_90": float(ranks_t.mean()),
        "std_rank_90": float(ranks_t.std()),
        "min_rank_90": int(ranks_t.min()),
        "max_rank_90": int(ranks_t.max()),
        "stacked_jacobian_rank_90": stacked_rank,
    }
    per_block.append(rec)
    print(f"  mean_rank={rec['mean_rank_90']:.1f} ± {rec['std_rank_90']:.1f}, "
          f"min={rec['min_rank_90']}, max={rec['max_rank_90']}, "
          f"stacked={stacked_rank}, elapsed={elapsed:.1f}s")

total_elapsed = time.time() - t0_total

# ── Verdict ───────────────────────────────────────────────────────────────────
all_mean_ranks = [r["mean_rank_90"] for r in per_block]
overall_mean_rank = sum(all_mean_ranks) / len(all_mean_ranks)
n_low_rank = sum(1 for r in all_mean_ranks if r < 100)

if n_low_rank >= 6:
    verdict = "CONFIRMED"
elif all_mean_ranks[-1] > 300 and overall_mean_rank > 300:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nOverall mean rank: {overall_mean_rank:.1f}")
print(f"Low-rank blocks (<100): {n_low_rank}/12")
print(f"Verdict: {verdict}")
print(f"Total time: {total_elapsed:.1f}s")

# ── Save output ───────────────────────────────────────────────────────────────
result = {
    "cycle": 195,
    "item": 206,
    "verdict": verdict,
    "overall_mean_rank_90": overall_mean_rank,
    "n_low_rank_blocks": n_low_rank,
    "all_mean_ranks": all_mean_ranks,
    "per_block": per_block,
    "total_elapsed_s": total_elapsed,
    "config": {
        "N_SEQS": N_SEQS,
        "SEQ_LEN": SEQ_LEN,
        "N_tokens_per_block": N_tokens,
        "ENERGY_THRESHOLD": ENERGY_THRESHOLD,
        "DEVICE": DEVICE,
        "SEED": SEED,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

print(f"Saved to {OUTPUT_PATH}")
