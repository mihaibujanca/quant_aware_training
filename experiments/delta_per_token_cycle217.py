"""
Cycle 217 — Item 231: Per-token Δ(x) = GELU_FFN(x) − ReLU_FFN(x) for GPT-2.

For each of 12 blocks:
  - Capture mlp.c_fc input (post-LN2 residual) over 200 × 128 = 25600 OWT/Wikitext
    calibration tokens.
  - Compute per-token Δ_out = c_proj(GELU(pre) − ReLU(pre)).
  - SVD(Δ_out), report rank_95, rank_99, ||E[Δ]||² / E[||Δ||²] (TI/TD ratio),
    top-3 right singular vectors projected through lm_head (vocab impact).
  - Cross-block alignment: principal cosines of top-k right SVs of Δ_l with
    top-20 left SVs of next block's c_fc weight.

Verdict:
  - low_rank := rank_95 ≤ 64 in ≥ 6 of 12 blocks
  - interpretable := ≥ 3 blocks where any of (SV 0,1,2 top-10 boost/supp) contains
    3+ alphanumeric word tokens (heuristic).
  - CONFIRMED := low_rank AND interpretable
  - REFUTED   := rank_95 > 400 in ALL blocks
  - INCONCLUSIVE otherwise
"""
import json
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_JSON = "autoresearch/results/cycle_217_delta_per_token.json"
OUTPUT_MD = "autoresearch/results/cycle_217_summary.md"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Calibration data (matches cycle 206 loader) ──────────────────────────────
print("Loading wikitext-2 validation split...")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB:
        break

calib_ids = [t.to(DEVICE) for t in all_ids[:N_CALIB]]
print(f"Calibration: {len(calib_ids)} sequences × {SEQ_LEN} tokens")

# ── Forward pass with c_fc input hooks ───────────────────────────────────────
N_BLOCKS = 12
stash = {l: [] for l in range(N_BLOCKS)}

def make_hook(l):
    def fn(module, input, output):
        stash[l].append(input[0].detach().float().cpu())
    return fn

hooks = []
for l in range(N_BLOCKS):
    h = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_hook(l))
    hooks.append(h)

print("Running forward pass...")
with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)
for h in hooks:
    h.remove()

# Concatenate and reshape: (N*T, 768)
mlp_inputs = {}
for l in range(N_BLOCKS):
    cat = torch.cat(stash[l], dim=0)  # (N, T, 768) or flat
    mlp_inputs[l] = cat.reshape(-1, 768)
    assert mlp_inputs[l].shape == (N_CALIB * SEQ_LEN, 768), \
        f"block {l} shape {mlp_inputs[l].shape}"
stash = None

print(f"Captured inputs: {mlp_inputs[0].shape} per block")

# ── Per-block Δ analysis ─────────────────────────────────────────────────────
# lm_head on CPU for vocab projection
lm_head_W = model.lm_head.weight.detach().float().cpu()  # (50257, 768)

def is_word_token(tok: str) -> bool:
    """Heuristic: token is a word if stripped form is ≥2 alphanumeric chars."""
    s = tok.strip()
    if len(s) < 2:
        return False
    # Require at least one letter
    return bool(re.search(r"[A-Za-z]", s)) and bool(re.match(r"^[A-Za-z0-9'\-]+$", s))

def count_word_tokens(tokens):
    return sum(1 for t in tokens if is_word_token(t))

blocks_out = []
interpretable_block_count = 0

for l in range(N_BLOCKS):
    print(f"\n── Block {l} ──")
    x = mlp_inputs[l]  # (25600, 768) CPU float32
    block = model.transformer.h[l]

    # Move c_fc weights to CPU for this computation (avoids mps memory pressure)
    W_fc = block.mlp.c_fc.weight.detach().float().cpu()   # (768, 3072) — Conv1D
    b_fc = block.mlp.c_fc.bias.detach().float().cpu()     # (3072,)
    W_proj = block.mlp.c_proj.weight.detach().float().cpu()  # (3072, 768) — Conv1D

    # Chunked preact + delta_out to keep memory in check
    chunk = 4096
    delta_out_chunks = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i+chunk]
        preact = xc @ W_fc + b_fc  # (c, 3072)
        gelu_post = F.gelu(preact, approximate="tanh")  # GPT-2 new GELU
        relu_post = F.relu(preact)
        delta_post = gelu_post - relu_post
        do = delta_post @ W_proj  # (c, 768) — bias cancels
        delta_out_chunks.append(do)
    delta_out = torch.cat(delta_out_chunks, dim=0)  # (25600, 768)
    delta_out_chunks = None

    # ── SVD ──
    U, S, Vh = torch.linalg.svd(delta_out, full_matrices=False)  # S (768,), Vh (768, 768)
    energy = (S ** 2)
    cum = energy.cumsum(0) / energy.sum()
    rank_95 = int((cum < 0.95).sum().item()) + 1
    rank_99 = int((cum < 0.99).sum().item()) + 1

    # ── TI/TD ratio ──
    mean_delta = delta_out.mean(0)  # (768,)
    per_token_norm_sq = (delta_out ** 2).sum(1)  # (25600,)
    delta_norm_mean = per_token_norm_sq.sqrt().mean().item()
    ti_td_ratio = (mean_delta.norm() ** 2 / per_token_norm_sq.mean()).item()

    print(f"  rank_95={rank_95}, rank_99={rank_99}, ||Δ||_mean={delta_norm_mean:.3f}, "
          f"TI/TD={ti_td_ratio:.4f}")

    # ── Top-3 singular vectors → vocab ──
    top_svs = []
    interpretable_any = False
    for k in range(3):
        v = Vh[k]  # (768,)
        logit_impact = lm_head_W @ v  # (50257,)
        top_boost = torch.topk(logit_impact, 10)
        top_supp = torch.topk(-logit_impact, 10)
        boost_tokens = [tokenizer.decode([i]) for i in top_boost.indices.tolist()]
        supp_tokens = [tokenizer.decode([i]) for i in top_supp.indices.tolist()]
        boost_scores = [float(s) for s in top_boost.values.tolist()]
        supp_scores = [float(-s) for s in top_supp.values.tolist()]

        # Interpretability heuristic: ≥3 alphanumeric word tokens in boost or supp
        if count_word_tokens(boost_tokens) >= 3 or count_word_tokens(supp_tokens) >= 3:
            interpretable_any = True

        top_svs.append({
            "sv_idx": k,
            "singular_value": float(S[k].item()),
            "boost_tokens": boost_tokens,
            "boost_scores": boost_scores,
            "supp_tokens": supp_tokens,
            "supp_scores": supp_scores,
        })
        if k == 0:
            print(f"  SV0 boost: {boost_tokens[:5]}")
            print(f"  SV0 supp:  {supp_tokens[:5]}")

    if interpretable_any:
        interpretable_block_count += 1

    blocks_out.append({
        "block": l,
        "delta_norm_mean": delta_norm_mean,
        "rank_95": rank_95,
        "rank_99": rank_99,
        "top20_singular_values": [float(s) for s in S[:20].tolist()],
        "ti_td_ratio": ti_td_ratio,
        "top_singular_vectors": top_svs,
    })

# ── Cross-block alignment (Vh_l top-k vs U of next block's c_fc) ─────────────
print("\n── Cross-block alignment ──")
cross_out = []
# Also compute random baseline (3 seeds × top-3 random unit vectors in R^768)
def rand_unit_mat(seed, k=3, d=768):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(k, d, generator=g)
    m = m / m.norm(dim=1, keepdim=True)
    return m

random_mats_top3 = [rand_unit_mat(s, k=3) for s in (42, 137, 2024)]

# Recompute Vh per block (cheap; avoid holding delta_out in memory)
# We already saved top3 SV metadata, but we need Vh[:3] and Vh[:20].
# Store only Vh[:20] per block.
vh_top20 = {}
for l in range(N_BLOCKS):
    x = mlp_inputs[l]
    W_fc = model.transformer.h[l].mlp.c_fc.weight.detach().float().cpu()
    b_fc = model.transformer.h[l].mlp.c_fc.bias.detach().float().cpu()
    W_proj = model.transformer.h[l].mlp.c_proj.weight.detach().float().cpu()
    chunk = 4096
    parts = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i+chunk]
        preact = xc @ W_fc + b_fc
        gelu_post = F.gelu(preact, approximate="tanh")
        relu_post = F.relu(preact)
        delta_post = gelu_post - relu_post
        parts.append(delta_post @ W_proj)
    dout = torch.cat(parts, dim=0)
    _, _, Vh = torch.linalg.svd(dout, full_matrices=False)
    vh_top20[l] = Vh[:20].clone()

for l in range(N_BLOCKS - 1):
    # c_fc weight of block l+1: (768, 3072) — Conv1D
    W_next = model.transformer.h[l + 1].mlp.c_fc.weight.detach().float().cpu()
    # Left singular vectors of W_next (the 768-dim reading subspace)
    U_next, _, _ = torch.linalg.svd(W_next, full_matrices=False)  # U_next (768, 768)
    U_next20 = U_next[:, :20]  # (768, 20)

    Vh_top3 = vh_top20[l][:3]   # (3, 768)
    Vh_top20 = vh_top20[l]       # (20, 768)

    # Principal cosines: singular values of (Vh_top3 @ U_next20)
    prod_top3 = Vh_top3 @ U_next20  # (3, 20)
    cos_top3 = torch.linalg.svdvals(prod_top3)  # (3,)

    prod_top20 = Vh_top20 @ U_next20  # (20, 20)
    cos_top20 = torch.linalg.svdvals(prod_top20)  # (20,)

    # Random baselines: top-3 random unit vectors × U_next20
    rand_top3_means = []
    for rm in random_mats_top3:
        rp = rm @ U_next20  # (3, 20)
        rc = torch.linalg.svdvals(rp)  # (3,)
        rand_top3_means.append(rc.mean().item())
    random_top3_mean = sum(rand_top3_means) / len(rand_top3_means)

    cross_out.append({
        "l": l,
        "lp1": l + 1,
        "principal_cos_top3": [float(x) for x in cos_top3.tolist()],
        "principal_cos_mean_top3": float(cos_top3.mean().item()),
        "principal_cos_top20": [float(x) for x in cos_top20.tolist()],
        "principal_cos_top20_mean": float(cos_top20.mean().item()),
        "random_baseline_top3_mean": random_top3_mean,
    })
    print(f"  l={l}→{l+1}: cos_top3_mean={cos_top3.mean().item():.4f}, "
          f"cos_top20_mean={cos_top20.mean().item():.4f}, "
          f"random={random_top3_mean:.4f}")

# ── Verdict ──────────────────────────────────────────────────────────────────
rank95_list = [b["rank_95"] for b in blocks_out]
n_lowrank = sum(1 for r in rank95_list if r <= 64)
max_rank95 = max(rank95_list)
median_rank95 = sorted(rank95_list)[len(rank95_list) // 2]
low_rank = n_lowrank >= 6

interpretable = interpretable_block_count >= 3

if low_rank and interpretable:
    verdict = "CONFIRMED"
elif all(r > 400 for r in rank95_list):
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n── Verdict ──")
print(f"rank_95 per block: {rank95_list}")
print(f"n blocks rank_95 ≤ 64: {n_lowrank}/12 (low_rank={low_rank})")
print(f"interpretable blocks (SV0-2 word-token heuristic): {interpretable_block_count}")
print(f"median rank_95 = {median_rank95}, max = {max_rank95}")
print(f"VERDICT: {verdict}")

# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 217,
    "item": 231,
    "verdict": verdict,
    "config": {
        "n_sequences": N_CALIB,
        "seq_len": SEQ_LEN,
        "n_tokens": N_CALIB * SEQ_LEN,
        "model": "gpt2",
    },
    "summary": {
        "rank_95_per_block": rank95_list,
        "n_blocks_rank95_le_64": n_lowrank,
        "median_rank_95": median_rank95,
        "max_rank_95": max_rank95,
        "low_rank": low_rank,
        "interpretable_block_count": interpretable_block_count,
        "interpretable": interpretable,
    },
    "blocks": blocks_out,
    "cross_block_alignment": cross_out,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON → {OUTPUT_JSON}")

# ── Summary markdown ─────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 217 — Item 231: Per-token Δ = GELU_FFN − ReLU_FFN")
lines.append("")
lines.append(f"**Verdict:** {verdict}")
lines.append("")
lines.append("## Headline")
lines.append(f"- Median rank_95 across 12 blocks: **{median_rank95}** (max {max_rank95})")
lines.append(f"- Blocks with rank_95 ≤ 64: **{n_lowrank} / 12** (low_rank criterion: ≥ 6)")
lines.append(f"- Blocks with interpretable top SVs (word-token heuristic): **{interpretable_block_count} / 12**")
lines.append("")
lines.append("## Per-block")
lines.append("")
lines.append("| Block | rank_95 | rank_99 | ||Δ||_mean | TI/TD | SV0 top boost |")
lines.append("|------:|--------:|--------:|-----------:|------:|:--------------|")
for b in blocks_out:
    sv0_boost = b["top_singular_vectors"][0]["boost_tokens"][:3]
    sv0_str = " ".join(repr(t) for t in sv0_boost)
    lines.append(f"| {b['block']} | {b['rank_95']} | {b['rank_99']} | "
                 f"{b['delta_norm_mean']:.3f} | {b['ti_td_ratio']:.4f} | {sv0_str} |")
lines.append("")

# Cross-block alignment
lines.append("## Cross-block alignment — Δ_l top SVs vs c_fc_{l+1} left SVs (top-20)")
lines.append("")
lines.append("| l→l+1 | cos_top3_mean | cos_top20_mean | random_baseline_top3_mean |")
lines.append("|:------|-------------:|--------------:|--------------------------:|")
mean_cos_top3_all = sum(c["principal_cos_mean_top3"] for c in cross_out) / len(cross_out)
mean_cos_top20_all = sum(c["principal_cos_top20_mean"] for c in cross_out) / len(cross_out)
mean_random_all = sum(c["random_baseline_top3_mean"] for c in cross_out) / len(cross_out)
for c in cross_out:
    lines.append(f"| {c['l']}→{c['lp1']} | {c['principal_cos_mean_top3']:.4f} | "
                 f"{c['principal_cos_top20_mean']:.4f} | {c['random_baseline_top3_mean']:.4f} |")
lines.append(f"| **mean** | **{mean_cos_top3_all:.4f}** | **{mean_cos_top20_all:.4f}** | "
             f"**{mean_random_all:.4f}** |")
lines.append("")
lines.append(f"Cycle 216 W_down→c_fc top-5 principal cos mean baseline: **0.649** "
             f"(mid-blocks 4-7 hit 0.70-0.78).")
lines.append("")

# Striking blocks: pick 3 with lowest rank_95 and up to 2 with highest
sorted_by_rank = sorted(blocks_out, key=lambda b: b["rank_95"])
striking = sorted_by_rank[:3] + sorted_by_rank[-2:]
seen = set()
striking_unique = []
for b in striking:
    if b["block"] not in seen:
        seen.add(b["block"])
        striking_unique.append(b)

lines.append("## Striking blocks")
for b in striking_unique:
    lines.append(f"### Block {b['block']} — rank_95={b['rank_95']}, "
                 f"TI/TD={b['ti_td_ratio']:.4f}, ||Δ||={b['delta_norm_mean']:.3f}")
    for sv in b["top_singular_vectors"]:
        lines.append(f"- **SV{sv['sv_idx']}** (σ={sv['singular_value']:.2f})")
        lines.append(f"  - boost: {sv['boost_tokens']}")
        lines.append(f"  - supp:  {sv['supp_tokens']}")
    lines.append("")

# Plain language
lines.append("## Plain-language summary")
if verdict == "CONFIRMED":
    lines.append(
        f"The GELU−ReLU difference Δ(x) lives in a low-rank subspace in "
        f"{n_lowrank} of 12 blocks (median rank_95 = {median_rank95}). "
        f"Top singular directions project to interpretable vocabulary clusters, "
        f"meaning the 'bump' is a structured, token-dependent message — not a "
        f"full-dimensional correction. Cross-block principal cosines vs random baseline: "
        f"{mean_cos_top3_all:.3f} vs {mean_random_all:.3f}."
    )
elif verdict == "REFUTED":
    lines.append(
        f"Δ(x) is essentially full-rank in every block (min rank_95 = {min(rank95_list)}, "
        f"max = {max_rank95}). The bump is not a low-rank structured signal."
    )
else:
    lines.append(
        f"Mixed evidence. {n_lowrank}/12 blocks meet the low-rank criterion "
        f"(median rank_95 = {median_rank95}, max = {max_rank95}). "
        f"Interpretable blocks: {interpretable_block_count}/12 (heuristic). "
        f"Cross-block principal cosines: {mean_cos_top3_all:.3f} (top-3) / "
        f"{mean_cos_top20_all:.3f} (top-20) vs random {mean_random_all:.3f}."
    )

with open(OUTPUT_MD, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved MD   → {OUTPUT_MD}")
print("Done.")
