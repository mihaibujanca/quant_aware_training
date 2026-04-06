"""
Cycle 218 — Item 232: Where does Δ point? — CENTRED Δ̃ alignment with
next block's reading directions.

Extends cycle 217. For each block l, SVD the CENTRED Δ̃_l = Δ_l − mean(Δ_l)
and measure cross-block alignment of its top-20 right singular vectors with
  - block l+1 c_fc left singular subspace (top-20)
  - block l+1 c_attn Q, K, V left singular subspaces (top-20 each)
  - block l+1 residual-stream input top-20 principal components

Random baseline: 10 seeds of Gaussian orthonormal (20, 768) bases.
Significance: cos_sum > random_mean + 3 * random_std (~ p < 0.01).

Verdict:
  - CONFIRMED  := n_significant_pairs >= 8 (out of 11 cross-block pairs)
  - REFUTED    := n_significant_pairs == 0
  - INCONCLUSIVE otherwise
"""
import json
import re
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB = 200
SEQ_LEN = 128
SEED = 42
N_RANDOM_SEEDS = 10
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_JSON = "autoresearch/results/cycle_218_delta_centred_alignment.json"
OUTPUT_MD = "autoresearch/results/cycle_218_summary.md"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Calibration data (matches cycle 217 loader) ──────────────────────────────
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

# ── Hook: capture c_fc input AND block-input (pre-LN1) per block ─────────────
N_BLOCKS = 12
cfc_stash = {l: [] for l in range(N_BLOCKS)}
blockin_stash = {l: [] for l in range(N_BLOCKS)}

def make_cfc_hook(l):
    def fn(module, input, output):
        cfc_stash[l].append(input[0].detach().float().cpu())
    return fn

def make_blockin_prehook(l):
    def fn(module, args):
        # args is a tuple; args[0] is the block input (batch, seq, 768)
        blockin_stash[l].append(args[0].detach().float().cpu())
    return fn

hooks = []
for l in range(N_BLOCKS):
    h1 = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_cfc_hook(l))
    h2 = model.transformer.h[l].register_forward_pre_hook(make_blockin_prehook(l))
    hooks.append(h1)
    hooks.append(h2)

print("Running forward pass...")
with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)
for h in hooks:
    h.remove()

mlp_inputs = {}
block_inputs = {}
for l in range(N_BLOCKS):
    mlp_inputs[l] = torch.cat(cfc_stash[l], dim=0).reshape(-1, 768)
    block_inputs[l] = torch.cat(blockin_stash[l], dim=0).reshape(-1, 768)
    assert mlp_inputs[l].shape == (N_CALIB * SEQ_LEN, 768)
    assert block_inputs[l].shape == (N_CALIB * SEQ_LEN, 768)

cfc_stash = None
blockin_stash = None
print(f"Captured c_fc inputs: {mlp_inputs[0].shape}")
print(f"Captured block inputs: {block_inputs[0].shape}")

# ── Per-block centred Δ̃ analysis ────────────────────────────────────────────
lm_head_W = model.lm_head.weight.detach().float().cpu()  # (50257, 768)

def is_word_token(tok: str) -> bool:
    s = tok.strip()
    if len(s) < 2:
        return False
    return bool(re.search(r"[A-Za-z]", s)) and bool(re.match(r"^[A-Za-z0-9'\-]+$", s))

def count_word_tokens(tokens):
    return sum(1 for t in tokens if is_word_token(t))

blocks_out = []
vh_top20 = {}  # cache centred Δ̃ top-20 right SVs per block
sigma0_list = []
sigma1_list = []
rank95_list = []
rank99_list = []
interpretable_block_count = 0

for l in range(N_BLOCKS):
    print(f"\n── Block {l} (centred Δ̃) ──")
    x = mlp_inputs[l]
    block = model.transformer.h[l]

    W_fc = block.mlp.c_fc.weight.detach().float().cpu()
    b_fc = block.mlp.c_fc.bias.detach().float().cpu()
    W_proj = block.mlp.c_proj.weight.detach().float().cpu()

    chunk = 4096
    parts = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i+chunk]
        preact = xc @ W_fc + b_fc
        gelu_post = F.gelu(preact, approximate="tanh")
        relu_post = F.relu(preact)
        delta_post = gelu_post - relu_post
        parts.append(delta_post @ W_proj)
    delta_out = torch.cat(parts, dim=0)  # (25600, 768)
    parts = None

    mean_delta = delta_out.mean(0)  # (768,)
    mean_delta_norm = float(mean_delta.norm().item())
    delta_centred = delta_out - mean_delta
    delta_out = None

    U, S, Vh = torch.linalg.svd(delta_centred, full_matrices=False)
    energy = S ** 2
    cum = energy.cumsum(0) / energy.sum()
    rank_95 = int((cum < 0.95).sum().item()) + 1
    rank_99 = int((cum < 0.99).sum().item()) + 1
    sigma0 = float(S[0].item())
    sigma1 = float(S[1].item())
    sigma2 = float(S[2].item())

    rank95_list.append(rank_95)
    rank99_list.append(rank_99)
    sigma0_list.append(sigma0)
    sigma1_list.append(sigma1)

    print(f"  rank_95={rank_95}, rank_99={rank_99}, σ₀={sigma0:.2f}, σ₁={sigma1:.2f}, "
          f"σ₂={sigma2:.2f}, ||mean_Δ||={mean_delta_norm:.3f}")

    vh_top20[l] = Vh[:20].clone()  # (20, 768)

    # Vocab interpretability on centred Δ̃
    top_svs = []
    interpretable_any = False
    for k in range(3):
        v = Vh[k]
        logit_impact = lm_head_W @ v
        top_boost = torch.topk(logit_impact, 10)
        top_supp = torch.topk(-logit_impact, 10)
        boost_tokens = [tokenizer.decode([i]) for i in top_boost.indices.tolist()]
        supp_tokens = [tokenizer.decode([i]) for i in top_supp.indices.tolist()]
        boost_scores = [float(s) for s in top_boost.values.tolist()]
        supp_scores = [float(-s) for s in top_supp.values.tolist()]
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
        "mean_delta_norm": mean_delta_norm,
        "rank_95_centred": rank_95,
        "rank_99_centred": rank_99,
        "sigma0": sigma0,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "top_singular_vectors": top_svs,
    })

# Raw-rank_95 from cycle 217 for comparison
RAW_RANK95_C217 = [10, 132, 1, 1, 1, 1, 1, 8, 73, 110, 1, 1]

# ── Reader bases per block l+1 (in R^768) ────────────────────────────────────
print("\n── Building downstream reader bases ──")

def top_left_svs(W, k=20):
    """Top-k left singular vectors of W as columns. Returns (d, k)."""
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    return U[:, :k].contiguous()

reader_bases = {}  # reader_bases[lp1][reader_name] -> (768, 20)
for lp1 in range(1, N_BLOCKS):
    block = model.transformer.h[lp1]

    # c_fc: Conv1D stores weight as (768, 3072)
    W_fc = block.mlp.c_fc.weight.detach().float().cpu()
    assert W_fc.shape == (768, 3072), f"block {lp1} W_fc shape {W_fc.shape}"
    U_fc = top_left_svs(W_fc, 20)
    assert U_fc.shape == (768, 20)

    # c_attn: Conv1D stores weight as (768, 2304); slice into Q/K/V
    W_attn = block.attn.c_attn.weight.detach().float().cpu()
    assert W_attn.shape == (768, 2304), f"block {lp1} W_attn shape {W_attn.shape}"
    Q = W_attn[:, 0:768]
    K = W_attn[:, 768:1536]
    V = W_attn[:, 1536:2304]
    assert Q.shape == (768, 768) and K.shape == (768, 768) and V.shape == (768, 768)
    U_q = top_left_svs(Q, 20)
    U_k = top_left_svs(K, 20)
    U_v = top_left_svs(V, 20)

    # Residual stream PCs at block lp1 input (centred)
    R = block_inputs[lp1]  # (n_tokens, 768)
    R_mean = R.mean(0)
    R_c = R - R_mean
    _, _, Vh_R = torch.linalg.svd(R_c, full_matrices=False)
    # PCs in R^768 are right singular vectors = Vh rows. Stack top-20 as columns.
    U_res = Vh_R[:20].T.contiguous()  # (768, 20)
    assert U_res.shape == (768, 20)

    reader_bases[lp1] = {
        "c_fc": U_fc,
        "Q": U_q,
        "K": U_k,
        "V": U_v,
        "resPC": U_res,
    }
    print(f"  lp1={lp1}: reader bases built (c_fc, Q, K, V, resPC)")

# ── Random baseline: 10 seeds of Gaussian orthonormal (20, 768) ──────────────
print("\n── Random baseline (10 seeds) ──")

def random_orthonormal(seed, k=20, d=768):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(d, k, generator=g)  # (768, 20)
    Q, _ = torch.linalg.qr(G)  # Q (768, 20) orthonormal columns
    return Q.T.contiguous()  # (20, 768) orthonormal rows

random_V20 = [random_orthonormal(42 + s * 1000 + 7) for s in range(N_RANDOM_SEEDS)]

# Collect random cos_sums over all (seed, lp1, reader)
random_stats = {name: {"top20": [], "top5": []} for name in ("c_fc", "Q", "K", "V", "resPC")}
for lp1 in range(1, N_BLOCKS):
    rb = reader_bases[lp1]
    for name in ("c_fc", "Q", "K", "V", "resPC"):
        B = rb[name]  # (768, 20)
        for Vr in random_V20:
            M = Vr @ B  # (20, 20)
            cos = torch.linalg.svdvals(M)  # (20,) sorted descending
            random_stats[name]["top20"].append(cos.sum().item())
            random_stats[name]["top5"].append(cos[:5].sum().item())

random_baseline = {}
for name, stats in random_stats.items():
    t20 = torch.tensor(stats["top20"])
    t5 = torch.tensor(stats["top5"])
    random_baseline[name] = {
        "mean_top20": float(t20.mean().item()),
        "std_top20": float(t20.std().item()),
        "mean_top5": float(t5.mean().item()),
        "std_top5": float(t5.std().item()),
    }
    print(f"  {name}: top20 mean={t20.mean():.4f} ± {t20.std():.4f} | "
          f"top5 mean={t5.mean():.4f} ± {t5.std():.4f}")

# ── Cross-block alignment ────────────────────────────────────────────────────
print("\n── Cross-block alignment (centred Δ̃_l → reader(l+1)) ──")
cross_out = []
pair_sig = []

for l in range(N_BLOCKS - 1):
    V20 = vh_top20[l]  # (20, 768)
    rb = reader_bases[l + 1]
    row = {"l": l, "lp1": l + 1}
    sig_row = {"l": l, "lp1": l + 1}
    any_sig = False
    for name in ("c_fc", "Q", "K", "V", "resPC"):
        B = rb[name]  # (768, 20)
        M = V20 @ B  # (20, 20)
        cos = torch.linalg.svdvals(M)
        cs20 = float(cos.sum().item())
        cs5 = float(cos[:5].sum().item())
        row[f"{name}_top20_cos_sum"] = cs20
        row[f"{name}_top5_cos_sum"] = cs5
        row[f"{name}_top20_cos"] = [float(x) for x in cos.tolist()]
        rb_mean = random_baseline[name]["mean_top20"]
        rb_std = random_baseline[name]["std_top20"]
        threshold = rb_mean + 3 * rb_std
        sig = cs20 > threshold
        sig_row[f"{name}_sig"] = bool(sig)
        sig_row[f"{name}_cos_sum"] = cs20
        sig_row[f"{name}_threshold"] = float(threshold)
        if sig:
            any_sig = True
    sig_row["any_sig"] = bool(any_sig)
    pair_sig.append(sig_row)
    cross_out.append(row)
    print(f"  l={l}→{l+1}: c_fc={row['c_fc_top20_cos_sum']:.3f} "
          f"Q={row['Q_top20_cos_sum']:.3f} K={row['K_top20_cos_sum']:.3f} "
          f"V={row['V_top20_cos_sum']:.3f} resPC={row['resPC_top20_cos_sum']:.3f} "
          f"any_sig={any_sig}")

n_significant_pairs = sum(1 for s in pair_sig if s["any_sig"])
if n_significant_pairs >= 8:
    verdict = "CONFIRMED"
elif n_significant_pairs == 0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nn_significant_pairs = {n_significant_pairs}/11")
print(f"VERDICT: {verdict}")

# Per-reader significance breakdown
reader_sig_counts = {}
for name in ("c_fc", "Q", "K", "V", "resPC"):
    reader_sig_counts[name] = sum(1 for s in pair_sig if s[f"{name}_sig"])
print(f"Reader sig counts: {reader_sig_counts}")

# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 218,
    "item": 232,
    "verdict": verdict,
    "config": {
        "n_sequences": N_CALIB,
        "seq_len": SEQ_LEN,
        "n_tokens": N_CALIB * SEQ_LEN,
        "n_random_seeds": N_RANDOM_SEEDS,
        "model": "gpt2",
    },
    "raw_rank_95_cycle217": RAW_RANK95_C217,
    "centred_rank_95_cycle218": rank95_list,
    "centred_rank_99_cycle218": rank99_list,
    "blocks": blocks_out,
    "cross_block_alignment": cross_out,
    "random_baseline": random_baseline,
    "significance": {
        "pair_significance_per_pair": pair_sig,
        "n_significant_pairs": n_significant_pairs,
        "reader_sig_counts": reader_sig_counts,
        "verdict": verdict,
    },
    "interpretability": {
        "n_blocks_with_word_tokens": interpretable_block_count,
    },
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON → {OUTPUT_JSON}")

# ── Summary markdown ─────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 218 — Item 232: Centred Δ̃ alignment with downstream readers")
lines.append("")
lines.append(f"**Verdict:** {verdict}")
lines.append("")
lines.append("## Headline")
lines.append(f"- **Significant cross-block pairs**: {n_significant_pairs} / 11 "
             f"(CONFIRMED ≥ 8, REFUTED = 0)")
lines.append(f"- Per-reader significant pair counts: {reader_sig_counts}")
lines.append(f"- Blocks with word-token SVs (heuristic): {interpretable_block_count} / 12")
lines.append("")
lines.append("## Rank of centred Δ̃ vs raw Δ (cycle 217)")
lines.append("")
lines.append("| Block | rank_95 RAW (c217) | rank_95 CENTRED | rank_99 CENTRED | σ₀ | σ₁ | σ₂ |")
lines.append("|------:|-------------------:|----------------:|----------------:|---:|---:|---:|")
for b in blocks_out:
    l = b["block"]
    lines.append(f"| {l} | {RAW_RANK95_C217[l]} | {b['rank_95_centred']} | "
                 f"{b['rank_99_centred']} | {b['sigma0']:.1f} | {b['sigma1']:.1f} | "
                 f"{b['sigma2']:.1f} |")
lines.append("")
raw_med = sorted(RAW_RANK95_C217)[len(RAW_RANK95_C217) // 2]
cen_med = sorted(rank95_list)[len(rank95_list) // 2]
lines.append(f"Median rank_95: raw {raw_med} → centred {cen_med} "
             f"(centring removes per-block constant bias).")
lines.append("")

lines.append("## Random baseline (10 seeds × 11 pairs × 5 readers)")
lines.append("")
lines.append("| Reader | mean top20 | std top20 | mean top5 | std top5 |")
lines.append("|:-------|-----------:|----------:|----------:|---------:|")
for name in ("c_fc", "Q", "K", "V", "resPC"):
    rb = random_baseline[name]
    lines.append(f"| {name} | {rb['mean_top20']:.4f} | {rb['std_top20']:.4f} | "
                 f"{rb['mean_top5']:.4f} | {rb['std_top5']:.4f} |")
lines.append("")

lines.append("## Cross-block alignment (top-20 principal cosine sums)")
lines.append("")
lines.append("| l→l+1 | c_fc | Q | K | V | resPC | any_sig |")
lines.append("|:------|-----:|--:|--:|--:|------:|:-------:|")
for row, sig in zip(cross_out, pair_sig):
    lines.append(f"| {row['l']}→{row['lp1']} | "
                 f"{row['c_fc_top20_cos_sum']:.3f} | {row['Q_top20_cos_sum']:.3f} | "
                 f"{row['K_top20_cos_sum']:.3f} | {row['V_top20_cos_sum']:.3f} | "
                 f"{row['resPC_top20_cos_sum']:.3f} | "
                 f"{'✓' if sig['any_sig'] else '·'} |")
lines.append("")
lines.append(f"**Significance**: cos_sum > random_mean + 3·std (≈ p < 0.01). "
             f"Per-reader thresholds ≈ c_fc {random_baseline['c_fc']['mean_top20'] + 3*random_baseline['c_fc']['std_top20']:.3f}, "
             f"Q {random_baseline['Q']['mean_top20'] + 3*random_baseline['Q']['std_top20']:.3f}, "
             f"K {random_baseline['K']['mean_top20'] + 3*random_baseline['K']['std_top20']:.3f}, "
             f"V {random_baseline['V']['mean_top20'] + 3*random_baseline['V']['std_top20']:.3f}, "
             f"resPC {random_baseline['resPC']['mean_top20'] + 3*random_baseline['resPC']['std_top20']:.3f}.")
lines.append("")

# Vocab interpretability striking blocks
lines.append("## Vocab interpretability — centred Δ̃ top-3 SVs")
lines.append("")
sorted_blocks = sorted(blocks_out, key=lambda b: b["rank_95_centred"])
striking = sorted_blocks[:3] + sorted_blocks[-2:]
seen = set()
unique_striking = []
for b in striking:
    if b["block"] not in seen:
        seen.add(b["block"])
        unique_striking.append(b)
for b in unique_striking:
    lines.append(f"### Block {b['block']} — rank_95 centred = {b['rank_95_centred']}, "
                 f"σ₀={b['sigma0']:.1f}, σ₁={b['sigma1']:.1f}")
    for sv in b["top_singular_vectors"]:
        lines.append(f"- **SV{sv['sv_idx']}** (σ={sv['singular_value']:.2f})")
        lines.append(f"  - boost: {sv['boost_tokens']}")
        lines.append(f"  - supp:  {sv['supp_tokens']}")
    lines.append("")

# Plain-language conclusion
lines.append("## Plain-language summary")
if verdict == "CONFIRMED":
    lines.append(
        f"After centring (subtracting the per-block mean Δ), the token-dependent "
        f"residual Δ̃ writes into directions that downstream blocks actually read. "
        f"{n_significant_pairs}/11 cross-block pairs exceed the random baseline "
        f"by > 3σ against at least one downstream reader "
        f"({', '.join(f'{k}:{v}' for k, v in reader_sig_counts.items())}). "
        f"Removing the mean promotes rank_95 from {raw_med} (raw, c217) to "
        f"{cen_med} (centred), confirming the constant bias hid the true per-token "
        f"geometry. The bump IS structured inter-block communication; downstream "
        f"components explicitly depend on it."
    )
elif verdict == "REFUTED":
    lines.append(
        f"Centred Δ̃ does NOT preferentially align with any downstream reader "
        f"subspace: 0/11 cross-block pairs beat the random baseline. Despite the "
        f"rank_95 jump from raw {raw_med} → centred {cen_med} showing the centring "
        f"exposes new structure, that structure is not preferentially read by the "
        f"next block's c_fc / Q / K / V / residual PCs. The bump's per-token "
        f"component is not inter-block communication — it may be within-block "
        f"computation that the residual stream discards."
    )
else:
    lines.append(
        f"Weak but present alignment. {n_significant_pairs}/11 cross-block pairs "
        f"exceed the random baseline against ≥1 reader "
        f"({', '.join(f'{k}:{v}' for k, v in reader_sig_counts.items())}). Centring "
        f"lifts rank_95 from raw median {raw_med} → centred {cen_med}. The mean "
        f"removal exposes some structured per-token residual, but most cross-block "
        f"pairs either don't clear the 3σ bar or clear it against the residual "
        f"stream PC baseline (structural geometry) rather than weight-specific "
        f"readers. Not clean confirmation of inter-block communication."
    )
lines.append("")

lines.append("## Implications for items 233 / 234")
lines.append(
    f"Item 233 (anti-podal channel twin census) depends on item 230 (REFUTED: "
    f"<1.2% twins), so its premise is already weakened regardless of this result. "
    f"Item 234 (surgical bump removal: GELU_noneg vs GELU_nobump) is where the "
    f"functional load of the bump will actually be localised. "
    + (
        f"Today's {verdict} result on Δ̃ alignment means the bump DOES write "
        f"into the next block's reading directions, so 234 is expected to show "
        f"that removing the bump knee (or negative leak) breaks cross-block "
        f"information flow, not just within-block nonlinearity."
        if verdict == "CONFIRMED" else
        f"Today's {verdict} result means we cannot yet attribute the bump's "
        f"functional role to explicit inter-block communication via weight "
        f"readers; item 234 must therefore disambiguate within-block from "
        f"cross-block effect independently, rather than relying on the "
        f"alignment interpretation."
    )
)
lines.append("")

with open(OUTPUT_MD, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved MD   → {OUTPUT_MD}")
print("Done.")
