"""
Cycle 201 — Item 211: Block 11 geometry deep-dive.
Block 11 has anomalous delta_norm=335 (11× median) and borderline droppability.
Three sub-analyses: output rank, token-type delta breakdown, delta direction coherence.
"""
import json
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_SEQS = 50
SEQ_LEN = 64
SEED = 42
ENERGY_THRESH = 0.90
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FOCUS_BLOCKS = [3, 7, 10, 11]
OUTPUT_PATH = "autoresearch/results/cycle_201_block11_geometry.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

print("Loading data...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
calib_ids = []
token_ids_flat = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        ids = enc["input_ids"][:, :SEQ_LEN]
        calib_ids.append(ids.to(DEVICE))
        token_ids_flat.extend(ids[0].tolist())
    if len(calib_ids) >= N_SEQS:
        break
N_tokens = len(calib_ids) * SEQ_LEN
print(f"Calibration: {len(calib_ids)} seqs × {SEQ_LEN} = {N_tokens} tokens")

# ── Helper ────────────────────────────────────────────────────────────────────

def effective_rank(X, threshold=ENERGY_THRESH):
    X_c = X - X.mean(0)
    S = torch.linalg.svdvals(X_c)
    energy = (S ** 2).cumsum(0) / ((S ** 2).sum() + 1e-12)
    return int((energy < threshold).sum()) + 1


def median(lst):
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


# ── Sub-analysis 1: Input/Output/Delta rank ───────────────────────────────────
print("Collecting block I/O...")
block_in = {b: [] for b in FOCUS_BLOCKS}
block_out = {b: [] for b in FOCUS_BLOCKS}
hooks = []
for b in FOCUS_BLOCKS:
    def mk_hook(bb):
        def fn(m, inp, out):
            block_in[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
            out_t = out[0] if isinstance(out, tuple) else out
            block_out[bb].append(out_t.detach().float().cpu().reshape(-1, 768))
        return fn
    hooks.append(model.transformer.h[b].register_forward_hook(mk_hook(b)))

with torch.no_grad():
    for ids in calib_ids:
        model(ids)

for h in hooks:
    h.remove()

for b in FOCUS_BLOCKS:
    block_in[b] = torch.cat(block_in[b], dim=0)   # (N_tokens, 768)
    block_out[b] = torch.cat(block_out[b], dim=0)

print("Computing ranks...")
per_focus_block = []
for b in FOCUS_BLOCKS:
    inp_rank = effective_rank(block_in[b])
    out_rank = effective_rank(block_out[b])
    delta = block_out[b] - block_in[b]
    d_rank = effective_rank(delta)
    d_norms = delta.norm(dim=1)
    print(f"  Block {b:2d}: in_rank={inp_rank}, out_rank={out_rank}, delta_rank={d_rank}, mean_delta={d_norms.mean():.3f}", flush=True)
    per_focus_block.append({
        "block": b,
        "input_rank_90": inp_rank,
        "output_rank_90": out_rank,
        "delta_rank_90": d_rank,
        "mean_delta_norm": float(d_norms.mean().item()),
        "std_delta_norm": float(d_norms.std().item()),
    })

# ── Sub-analysis 2: Token-type breakdown for block 11 ─────────────────────────
print("\nToken-type analysis for block 11...")
delta_11 = block_out[11] - block_in[11]   # (N_tokens, 768)
delta_norms_11 = delta_11.norm(dim=1)     # (N_tokens,)

# By position (0..SEQ_LEN-1)
delta_by_position = []
for pos in range(SEQ_LEN):
    pos_norms = delta_norms_11[pos::SEQ_LEN]
    delta_by_position.append(float(pos_norms.mean().item()))

# By token type
token_delta = {}  # token_id -> list of delta norms
for i, tok in enumerate(token_ids_flat):
    if tok not in token_delta:
        token_delta[tok] = []
    token_delta[tok].append(float(delta_norms_11[i].item()))

# Mean delta per token type
mean_delta_per_token = {tok: sum(v) / len(v) for tok, v in token_delta.items() if len(v) >= 3}

# Top 10 tokens by mean delta
top10_by_delta = sorted(mean_delta_per_token.items(), key=lambda x: -x[1])[:10]
top10_records = []
for tok_id, mean_d in top10_by_delta:
    tok_str = tokenizer.decode([tok_id])
    top10_records.append({"token_id": tok_id, "token": tok_str, "mean_delta_norm": mean_d, "count": len(token_delta[tok_id])})

top20_mean = float(sum(v for _, v in sorted(mean_delta_per_token.items(), key=lambda x: -x[1])[:20]) / min(20, len(mean_delta_per_token)))
overall_mean = float(delta_norms_11.mean().item())
delta_token_concentration = top20_mean > 2 * overall_mean

print(f"  Overall mean delta: {overall_mean:.3f}")
print(f"  Top-20 token mean delta: {top20_mean:.3f}")
print(f"  Concentration (top20 > 2x overall): {delta_token_concentration}")
print(f"  Top tokens by delta norm:")
for r in top10_records[:5]:
    print(f"    '{r['token']}' (id={r['token_id']}): mean={r['mean_delta_norm']:.3f}, n={r['count']}")

# ── Sub-analysis 3: Delta direction coherence ─────────────────────────────────
print("\nDelta direction analysis for block 11...")
delta_centered = delta_11 - delta_11.mean(0)
# SVD of delta matrix
U, S_sv, Vh = torch.linalg.svd(delta_centered, full_matrices=False)
v1 = Vh[0]  # top right singular vector: (768,)

# Cosine similarity of each token's delta with top direction
cos_v1 = F.cosine_similarity(delta_11, v1.unsqueeze(0).expand_as(delta_11), dim=1)
mean_cos_v1 = float(cos_v1.abs().mean().item())
delta_dir_coherent = mean_cos_v1 > 0.5

delta_rank = effective_rank(delta_centered)
top3_energy = float(((S_sv[:3]**2).sum() / (S_sv**2).sum() + 1e-12).item())

print(f"  Delta rank_90: {delta_rank}")
print(f"  Top-3 SV energy fraction: {top3_energy:.3f}")
print(f"  Mean |cos(delta_i, v_1)|: {mean_cos_v1:.3f}")
print(f"  Direction coherent (>0.5): {delta_dir_coherent}")

# ── Verdict ───────────────────────────────────────────────────────────────────
b11 = next(r for r in per_focus_block if r["block"] == 11)
others = [r for r in per_focus_block if r["block"] != 11]
median_other_out_rank = median([r["output_rank_90"] for r in others])
output_rank_anomaly = b11["output_rank_90"] > 3 * median_other_out_rank

outlier_flags = {
    "output_rank_anomaly": output_rank_anomaly,
    "delta_token_concentration": delta_token_concentration,
    "delta_dir_coherent": delta_dir_coherent,
}
outlier_count = sum(outlier_flags.values())

if outlier_count >= 2:
    verdict = "CONFIRMED"
elif outlier_count == 0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nBlock 11 outlier flags: {outlier_flags} (count={outlier_count})")
print(f"Verdict: {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "cycle": 201,
    "item": 211,
    "verdict": verdict,
    "outlier_count": outlier_count,
    "outlier_flags": outlier_flags,
    "per_focus_block": per_focus_block,
    "block11_token_analysis": {
        "delta_by_position": delta_by_position,
        "top20_token_mean_delta": top20_mean,
        "overall_mean_delta": overall_mean,
        "delta_token_concentration": delta_token_concentration,
        "top10_tokens_by_delta": top10_records,
    },
    "block11_delta_direction": {
        "delta_rank_90": delta_rank,
        "top3_sv_energy_frac": top3_energy,
        "mean_cos_top_sv": mean_cos_v1,
        "delta_dir_coherent": delta_dir_coherent,
    },
    "config": {
        "N_SEQS": N_SEQS,
        "SEQ_LEN": SEQ_LEN,
        "ENERGY_THRESH": ENERGY_THRESH,
        "SEED": SEED,
        "FOCUS_BLOCKS": FOCUS_BLOCKS,
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUTPUT_PATH}")
