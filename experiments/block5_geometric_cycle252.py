"""
Cycle 252 — Item 262: Block 5 geometric deep-dive (analysis-only, no training).

For blocks {1,2,3,4,5,6,7,8} compute 6 geometric metrics on post-GELU input to
c_proj and on c_proj row cosines. Compute z-scores for block 5 against cohort
{1,2,4,7,8}. Verdict:
  CONFIRMED if block 5 is >=1.5σ outlier on >=2 metrics
  REFUTED if block 5 within 1σ on ALL metrics
  INCONCLUSIVE otherwise (outlier on exactly 1 metric)
"""
import json
import math
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BLOCKS = [1, 2, 3, 4, 5, 6, 7, 8]
COHORT = [1, 2, 4, 7, 8]  # block 5 vs these
TARGET = 5
TOPK_VALUES = [128, 512]
OUTPUT_JSON = "autoresearch/results/cycle_252_block5_geometric.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Load calibration data (wikitext-2 val, all_ids[:200]) ─────────────────────
print("Loading calibration data...")
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
expected_n_tokens = N_CALIB * SEQ_LEN
print(f"Calibration: {len(calib_ids)} seqs × {SEQ_LEN} tokens = {expected_n_tokens} tokens")

# ── Capture post-GELU activation (input to c_proj) per block ──────────────────
post_gelu = {b: [] for b in BLOCKS}
hooks = []
for b in BLOCKS:
    def mk_hook(bb):
        def fn(m, inp, out):
            # input to c_proj is post-GELU activation, shape (B, T, 3072)
            x = inp[0].detach().float().cpu().reshape(-1, 3072)
            post_gelu[bb].append(x)
        return fn
    hooks.append(model.transformer.h[b].mlp.c_proj.register_forward_hook(mk_hook(b)))

print("Running forward passes...")
t0 = time.time()
with torch.no_grad():
    for ids in calib_ids:
        model(ids)
print(f"  forward done in {time.time()-t0:.1f}s")

for h in hooks:
    h.remove()

for b in BLOCKS:
    post_gelu[b] = torch.cat(post_gelu[b], dim=0)  # (n_tokens, 3072)
    assert post_gelu[b].shape == (expected_n_tokens, 3072), (
        f"block {b} shape {post_gelu[b].shape} != {(expected_n_tokens, 3072)}"
    )
print(f"Captured post_gelu shape: {post_gelu[BLOCKS[0]].shape}")

# ── Metrics ───────────────────────────────────────────────────────────────────

def channel_std_stats(X):
    """X: (n_tokens, 3072) → per-channel std distribution stats."""
    std_per_channel = X.std(dim=0)  # (3072,)
    qs = torch.tensor([0.10, 0.50, 0.90, 0.99])
    pcts = torch.quantile(std_per_channel, qs).tolist()
    return {
        "mean": float(std_per_channel.mean()),
        "std": float(std_per_channel.std()),
        "p10": pcts[0],
        "p50": pcts[1],
        "p90": pcts[2],
        "p99": pcts[3],
    }

def channel_kurtosis_stats(X):
    """Per-channel excess kurtosis: E[(x-μ)^4]/σ^4 − 3."""
    mu = X.mean(dim=0, keepdim=True)
    centered = X - mu
    sigma2 = (centered ** 2).mean(dim=0)
    m4 = (centered ** 4).mean(dim=0)
    kurt = m4 / (sigma2 ** 2 + 1e-12) - 3.0  # (3072,)
    return {
        "mean": float(kurt.mean()),
        "std": float(kurt.std()),
    }

def effective_rank(X):
    """SVD entropy effective rank."""
    # compute singular values
    s = torch.linalg.svdvals(X)  # min(n_tokens, 3072) values
    p = (s ** 2)
    p = p / p.sum()
    H = -(p * (p + 1e-30).log()).sum()
    return float(torch.exp(H)), s

def topk_coverage(X, W_proj, k):
    """Sort channels by std_i * ||c_proj row i||, sum top-k weight² / total weight²."""
    std_per = X.std(dim=0)  # (3072,)
    row_norm = W_proj.norm(dim=1)  # (3072,) — c_proj.weight is (3072, 768)
    w = std_per * row_norm  # (3072,)
    w2 = w ** 2
    total = w2.sum()
    topk = torch.topk(w2, k).values
    return float(topk.sum() / total)

def cproj_row_cosines(W_proj):
    """c_proj.weight (3072, 768). Row-normalize, compute off-diagonal cosine
    mean and max. Because (3072, 3072) is heavy, use a sampling approach for
    max (full for mean via Gram matrix)."""
    W_norm = F.normalize(W_proj, dim=1)  # (3072, 768)
    # Gram matrix chunked to avoid O(3072²) memory blowup
    n = W_norm.shape[0]
    chunk = 512
    off_sum = 0.0
    off_sq_sum = 0.0
    off_count = 0
    max_val = -float("inf")
    for i in range(0, n, chunk):
        G = W_norm[i:i+chunk] @ W_norm.T  # (c, 3072)
        ci = G.shape[0]
        # zero out diagonal entries relative to row index (i..i+ci)
        for r in range(ci):
            G[r, i + r] = 0.0
        # accumulate sum and max
        off_sum += float(G.sum())
        off_sq_sum += float((G ** 2).sum())
        off_count += ci * n - ci  # minus diagonals we zeroed
        m = float(G.max())
        if m > max_val:
            max_val = m
    mean_cos = off_sum / off_count
    # Note: off-diagonal symmetric (pairs counted twice); mean unaffected.
    return {"mean_cos": mean_cos, "max_cos": max_val}

# ── Compute per-block ─────────────────────────────────────────────────────────
per_block = {}
for b in BLOCKS:
    print(f"\n=== Block {b} ===")
    X = post_gelu[b]  # (n_tokens, 3072) on CPU float32
    W_proj = model.transformer.h[b].mlp.c_proj.weight.data.float().cpu()  # (3072, 768)

    std_stats = channel_std_stats(X)
    print(f"  std mean={std_stats['mean']:.4f} std={std_stats['std']:.4f}")

    kurt_stats = channel_kurtosis_stats(X)
    print(f"  kurtosis mean={kurt_stats['mean']:.4f}")

    t0 = time.time()
    eff_rank, sv = effective_rank(X)
    print(f"  effective rank={eff_rank:.2f}  (SVD {time.time()-t0:.1f}s)")

    covs = {}
    for k in TOPK_VALUES:
        covs[f"cov{k}"] = topk_coverage(X, W_proj, k)
        print(f"  topk-{k} coverage={covs[f'cov{k}']:.4f}")

    cos_stats = cproj_row_cosines(W_proj)
    print(f"  c_proj row cos mean={cos_stats['mean_cos']:.4f} max={cos_stats['max_cos']:.4f}")

    # Verification: eff_rank ≤ 3072, cov512 ≥ cov128
    assert eff_rank <= 3072.0 + 1e-6, f"block {b} eff_rank {eff_rank} > 3072"
    assert covs["cov512"] >= covs["cov128"] - 1e-9, (
        f"block {b} cov512 {covs['cov512']} < cov128 {covs['cov128']}"
    )

    per_block[b] = {
        "std_mean": std_stats["mean"],
        "std_std": std_stats["std"],
        "std_p10": std_stats["p10"],
        "std_p50": std_stats["p50"],
        "std_p90": std_stats["p90"],
        "std_p99": std_stats["p99"],
        "kurt_mean": kurt_stats["mean"],
        "kurt_std": kurt_stats["std"],
        "eff_rank": eff_rank,
        "cov128": covs["cov128"],
        "cov512": covs["cov512"],
        "mean_cos": cos_stats["mean_cos"],
        "max_cos": cos_stats["max_cos"],
    }

# ── Z-scores: block 5 vs cohort {1,2,4,7,8} ───────────────────────────────────
METRICS = ["std_mean", "kurt_mean", "eff_rank", "cov128", "cov512", "mean_cos"]
z_scores = {}
print("\n=== Z-scores (block 5 vs cohort {1,2,4,7,8}) ===")
for m in METRICS:
    cohort_vals = torch.tensor([per_block[b][m] for b in COHORT])
    cohort_mean = float(cohort_vals.mean())
    cohort_std = float(cohort_vals.std(unbiased=True))
    assert cohort_std > 0, f"cohort std zero for metric {m}"
    target_val = per_block[TARGET][m]
    z = (target_val - cohort_mean) / cohort_std
    z_scores[m] = {
        "target_val": target_val,
        "cohort_mean": cohort_mean,
        "cohort_std": cohort_std,
        "z": float(z),
    }
    print(f"  {m:12s}: target={target_val:.4f} cohort={cohort_mean:.4f}±{cohort_std:.4f}  z={z:+.3f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
outlier_metrics_15 = [m for m in METRICS if abs(z_scores[m]["z"]) >= 1.5]
outlier_metrics_1 = [m for m in METRICS if abs(z_scores[m]["z"]) >= 1.0]

if len(outlier_metrics_15) >= 2:
    verdict = "CONFIRMED"
elif len(outlier_metrics_1) == 0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== VERDICT: {verdict} ===")
print(f"  |z|≥1.5 metrics: {outlier_metrics_15}")
print(f"  |z|≥1.0 metrics: {outlier_metrics_1}")

# ── Write JSON output ─────────────────────────────────────────────────────────
result = {
    "cycle": 252,
    "item": 262,
    "verdict": verdict,
    "config": {
        "model": "gpt2",
        "n_calib": N_CALIB,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "device": DEVICE,
        "blocks": BLOCKS,
        "cohort": COHORT,
        "target": TARGET,
    },
    "per_block": per_block,
    "z_scores": z_scores,
    "outlier_metrics_15": outlier_metrics_15,
    "outlier_metrics_1": outlier_metrics_1,
    "verdict_logic": f"|z|>=1.5 count {len(outlier_metrics_15)}, |z|>=1.0 count {len(outlier_metrics_1)} -> {verdict}",
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_JSON}")
