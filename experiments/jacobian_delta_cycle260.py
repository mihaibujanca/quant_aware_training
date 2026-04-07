"""
Cycle 260 — Item 271: Jacobian delta ΔJ_l(x) = J_GELU − J_ReLU per token, per block.

For 256 calib tokens × 12 blocks, compute operator-level Jacobian delta
    ΔJ_l(x) = W_proj.T @ diag(σ'_GELU(z(x)) − σ'_ReLU(z(x))) @ W_fc.T  ∈ R^{768×768}

Per-token:  rank_95 via linalg.svdvals (no U/V), median over 256 tokens per block.
Per-block:  DJ_avg = mean_t ΔJ_l(x_t), SVD → top-20 left singular vectors as the
            "mode-1" (stable-across-tokens) subspace of ΔJ_l.
Alignment:  top-5 principal cosine sum with cycle 218's c_fc reading basis of block l+1
            (top-20 left singular vectors of W_fc_{l+1}), compared to a 10-seed random
            Gaussian orthonormal baseline (cycle 218 top-5 baseline was ~1.27).

Verdict (item 271):
  - CONFIRMED iff median rank_95(ΔJ_l) ≤ 100 in ≥6 blocks
              AND top-5 cos sum ≥ 2.0 in ≥6 block-pairs (l=0..10, reader=l+1)
  - REFUTED   iff median rank_95(ΔJ_l) ≥ 400 in ≥6 blocks
  - INCONCLUSIVE otherwise
"""
import json
import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB_SEQS_FOR_LOADER = 200   # match cycle 218 loader for determinism
N_TARGET_TOKENS = 256           # item 271 spec: 256 calib tokens
SEQ_LEN = 128
SEED = 42
N_RANDOM_SEEDS = 10
DEVICE_MODEL = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_LINALG = "cpu"           # linalg on CPU (reliable float32 linalg)
RESULT_PATH = "autoresearch/results/cycle_260_jacobian_delta.json"
SUMMARY_PATH = "autoresearch/results/cycle_260_summary.md"

torch.manual_seed(SEED)
print(f"Model device: {DEVICE_MODEL}", flush=True)
print(f"Linalg device: {DEVICE_LINALG}", flush=True)

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE_MODEL)
model.eval()

# ── Calibration data (matches cycle 218 loader) ──────────────────────────────
print("Loading wikitext-2 validation split...", flush=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB_SEQS_FOR_LOADER:
        break

# Take first 2 sequences -> 2 * 128 = 256 tokens (item 271 spec)
n_seqs_needed = (N_TARGET_TOKENS + SEQ_LEN - 1) // SEQ_LEN
calib_ids = [t.to(DEVICE_MODEL) for t in all_ids[:n_seqs_needed]]
print(f"Calib: using first {len(calib_ids)} sequences × {SEQ_LEN} tokens "
      f"= {len(calib_ids) * SEQ_LEN} tokens (target {N_TARGET_TOKENS})", flush=True)
assert len(calib_ids) * SEQ_LEN >= N_TARGET_TOKENS

# ── Hook c_fc input on every block ───────────────────────────────────────────
N_BLOCKS = 12
cfc_stash = {l: [] for l in range(N_BLOCKS)}

def make_cfc_hook(l):
    def fn(module, input, output):
        cfc_stash[l].append(input[0].detach().float().cpu())
    return fn

hooks = []
for l in range(N_BLOCKS):
    h = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_cfc_hook(l))
    hooks.append(h)

print("Running forward pass to capture c_fc inputs...", flush=True)
with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)
for h in hooks:
    h.remove()

ln2_out_per_block = {}
for l in range(N_BLOCKS):
    x = torch.cat(cfc_stash[l], dim=0).reshape(-1, 768)
    # Take first N_TARGET_TOKENS tokens
    ln2_out_per_block[l] = x[:N_TARGET_TOKENS].contiguous()
    assert ln2_out_per_block[l].shape == (N_TARGET_TOKENS, 768), \
        f"block {l} got {ln2_out_per_block[l].shape}"
cfc_stash = None
print(f"Captured ln2_out: {ln2_out_per_block[0].shape} per block", flush=True)

# ── Cache weights on linalg device ───────────────────────────────────────────
W_fc_per_block = {}
b_fc_per_block = {}
W_proj_per_block = {}
for l in range(N_BLOCKS):
    block = model.transformer.h[l]
    W_fc_per_block[l] = block.mlp.c_fc.weight.detach().float().to(DEVICE_LINALG)    # (768, 3072)
    b_fc_per_block[l] = block.mlp.c_fc.bias.detach().float().to(DEVICE_LINALG)      # (3072,)
    W_proj_per_block[l] = block.mlp.c_proj.weight.detach().float().to(DEVICE_LINALG)  # (3072, 768)
    assert W_fc_per_block[l].shape == (768, 3072)
    assert W_proj_per_block[l].shape == (3072, 768)

# ── c_fc reader basis per block l+1 (cycle 218 convention) ───────────────────
def top_left_svs(W, k=20):
    """Top-k left singular vectors of W as columns. Returns (d, k)."""
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    return U[:, :k].contiguous()

reader_basis = {}
for lp1 in range(1, N_BLOCKS):
    U_fc = top_left_svs(W_fc_per_block[lp1], 20)
    assert U_fc.shape == (768, 20)
    reader_basis[lp1] = U_fc

# ── Per-block ΔJ computation ─────────────────────────────────────────────────
print("\n──────── Computing per-token ΔJ and per-block aggregates ────────", flush=True)

per_block_results = []
dj_avg_per_block = {}

for l in range(N_BLOCKS):
    print(f"\n── Block {l} ──", flush=True)
    x_ln2 = ln2_out_per_block[l].to(DEVICE_LINALG)  # (256, 768)
    W_fc = W_fc_per_block[l]
    b_fc = b_fc_per_block[l]
    W_proj = W_proj_per_block[l]

    # preact: (256, 3072)
    preact = x_ln2 @ W_fc + b_fc

    # σ'_GELU via fresh-leaf autograd
    z = preact.detach().clone().requires_grad_(True)
    y = F.gelu(z, approximate='tanh').sum()
    sigma_gelu_prime = torch.autograd.grad(y, z)[0].detach()  # (256, 3072)

    # σ'_ReLU
    sigma_relu_prime = (preact > 0).float().detach()  # (256, 3072)

    delta_diag = sigma_gelu_prime - sigma_relu_prime  # (256, 3072)
    delta_diag_stats = {
        "mean": float(delta_diag.mean().item()),
        "std": float(delta_diag.std().item()),
        "min": float(delta_diag.min().item()),
        "max": float(delta_diag.max().item()),
        "abs_mean": float(delta_diag.abs().mean().item()),
        "frac_large": float((delta_diag.abs() > 0.1).float().mean().item()),
    }
    print(f"  delta_diag: mean={delta_diag_stats['mean']:+.5f} "
          f"abs_mean={delta_diag_stats['abs_mean']:.5f} "
          f"min={delta_diag_stats['min']:+.4f} max={delta_diag_stats['max']:+.4f} "
          f"frac>0.1={delta_diag_stats['frac_large']:.4f}", flush=True)

    # Per-token loop: build ΔJ, compute rank_95, accumulate running mean
    W_proj_T = W_proj.T.contiguous()  # (768, 3072)
    W_fc_T = W_fc.T.contiguous()      # (3072, 768)

    dj_sum = torch.zeros(768, 768, dtype=torch.float32, device=DEVICE_LINALG)
    rank_95_per_token = []
    fro_per_token = []

    for t in range(N_TARGET_TOKENS):
        dd = delta_diag[t]  # (3072,)
        # ΔJ = (W_proj.T * dd) @ W_fc.T : (768,3072) scaled by (3072,) then @ (3072,768) = (768,768)
        DJ_t = (W_proj_T * dd.unsqueeze(0)) @ W_fc_T
        dj_sum.add_(DJ_t)

        sv = torch.linalg.svdvals(DJ_t)  # (768,) descending
        energy = sv * sv
        total = energy.sum().clamp_min(1e-30)
        cum = energy.cumsum(0) / total
        r95 = int((cum < 0.95).sum().item()) + 1
        rank_95_per_token.append(r95)
        fro_per_token.append(float(sv.norm().item()))

    dj_avg = dj_sum / N_TARGET_TOKENS
    dj_avg_per_block[l] = dj_avg

    rank_95_t = torch.tensor(rank_95_per_token, dtype=torch.float64)
    fro_t = torch.tensor(fro_per_token, dtype=torch.float64)
    r95_p25 = float(torch.quantile(rank_95_t, 0.25).item())
    r95_med = float(torch.quantile(rank_95_t, 0.50).item())
    r95_p75 = float(torch.quantile(rank_95_t, 0.75).item())
    r95_min = int(rank_95_t.min().item())
    r95_max = int(rank_95_t.max().item())

    # SVD of averaged ΔJ for mode-1 subspace
    U_avg, S_avg, _ = torch.linalg.svd(dj_avg, full_matrices=False)
    U_avg_top20 = U_avg[:, :20].contiguous()
    avg_sv_top5 = [float(s.item()) for s in S_avg[:5]]
    avg_energy = (S_avg * S_avg)
    avg_total_e = float(avg_energy.sum().clamp_min(1e-30).item())
    avg_cum = avg_energy.cumsum(0) / max(avg_energy.sum().item(), 1e-30)
    avg_rank_95 = int((avg_cum < 0.95).sum().item()) + 1

    print(f"  rank_95 per-token: p25={r95_p25:.0f} med={r95_med:.0f} p75={r95_p75:.0f} "
          f"(min={r95_min} max={r95_max}) ||ΔJ||_F median={float(fro_t.median().item()):.3f}", flush=True)
    print(f"  avg ΔJ: σ[0:5]={[f'{s:.3f}' for s in avg_sv_top5]} rank_95_avg={avg_rank_95}", flush=True)

    per_block_results.append({
        "block": l,
        "delta_diag_stats": delta_diag_stats,
        "rank_95_per_token": rank_95_per_token,
        "rank_95_p25": r95_p25,
        "rank_95_median": r95_med,
        "rank_95_p75": r95_p75,
        "rank_95_min": r95_min,
        "rank_95_max": r95_max,
        "frobenius_median": float(fro_t.median().item()),
        "frobenius_per_token": fro_per_token,
        "avg_dj_top5_sv": avg_sv_top5,
        "avg_dj_rank_95": avg_rank_95,
    })

# ── Random baseline (10 seeds) ───────────────────────────────────────────────
print("\n──────── Random baseline (10 seeds × 11 pairs) ────────", flush=True)

def random_orthonormal_basis(seed, k=20, d=768):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(d, k, generator=g)  # (768, 20)
    Q, _ = torch.linalg.qr(G)  # Q (768, 20) orthonormal
    return Q.contiguous()

random_top5_sums = []
random_top20_sums = []
for lp1 in range(1, N_BLOCKS):
    B_reader = reader_basis[lp1]  # (768, 20)
    for s in range(N_RANDOM_SEEDS):
        U_r = random_orthonormal_basis(42 + s * 1000 + 7)  # (768, 20)
        M = U_r.T @ B_reader  # (20, 20)
        cos = torch.linalg.svdvals(M)  # (20,) descending
        random_top5_sums.append(float(cos[:5].sum().item()))
        random_top20_sums.append(float(cos.sum().item()))

rt5 = torch.tensor(random_top5_sums, dtype=torch.float64)
rt20 = torch.tensor(random_top20_sums, dtype=torch.float64)
random_baseline = {
    "n_samples": len(random_top5_sums),
    "top5_mean": float(rt5.mean().item()),
    "top5_std": float(rt5.std().item()),
    "top20_mean": float(rt20.mean().item()),
    "top20_std": float(rt20.std().item()),
}
print(f"  Random top5 cos sum: mean={random_baseline['top5_mean']:.4f} "
      f"± {random_baseline['top5_std']:.4f}", flush=True)
print(f"  Random top20 cos sum: mean={random_baseline['top20_mean']:.4f} "
      f"± {random_baseline['top20_std']:.4f}", flush=True)

# ── Cross-block alignment: mode-1 ΔJ_l vs c_fc reader of block l+1 ───────────
print("\n──────── Cross-block alignment (mode-1 ΔJ_l → c_fc reader l+1) ────────", flush=True)
alignment_per_pair = []
for l in range(N_BLOCKS - 1):
    # Mode-1 top-20 LEFT singular vectors of dj_avg
    U_avg, _, _ = torch.linalg.svd(dj_avg_per_block[l], full_matrices=False)
    U_top20 = U_avg[:, :20].contiguous()  # (768, 20)
    B_reader = reader_basis[l + 1]         # (768, 20)
    M = U_top20.T @ B_reader               # (20, 20)
    cos = torch.linalg.svdvals(M)          # (20,) descending
    top5_sum = float(cos[:5].sum().item())
    top20_sum = float(cos.sum().item())
    cos_list = [float(c.item()) for c in cos]
    threshold5 = random_baseline["top5_mean"] + 3 * random_baseline["top5_std"]
    sig = top5_sum > threshold5
    alignment_per_pair.append({
        "l": l,
        "lp1": l + 1,
        "top5_cos_sum": top5_sum,
        "top20_cos_sum": top20_sum,
        "top20_cos": cos_list,
        "passes_threshold_2.0": bool(top5_sum >= 2.0),
        "significant_vs_random_3sigma": bool(sig),
    })
    marker = "✓" if top5_sum >= 2.0 else "·"
    print(f"  l={l}→{l+1}: top5={top5_sum:.3f} top20={top20_sum:.3f} "
          f"thresh2.0={marker} sig3σ={'Y' if sig else 'N'}", flush=True)

# ── Verdict logic ────────────────────────────────────────────────────────────
print("\n──────── Verdict logic ────────", flush=True)
rank95_medians = [b["rank_95_median"] for b in per_block_results]
n_blocks_le_100 = sum(1 for r in rank95_medians if r <= 100)
n_blocks_ge_400 = sum(1 for r in rank95_medians if r >= 400)
n_pairs_ge_2 = sum(1 for p in alignment_per_pair if p["passes_threshold_2.0"])

clause1 = n_blocks_le_100 >= 6
clause2 = n_pairs_ge_2 >= 6
confirmed = clause1 and clause2
refuted = n_blocks_ge_400 >= 6
if confirmed:
    verdict = "CONFIRMED"
elif refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"  Per-block median rank_95: {[f'{r:.0f}' for r in rank95_medians]}", flush=True)
print(f"  n_blocks_with_median_r95_le_100 = {n_blocks_le_100}/12 (need ≥6)", flush=True)
print(f"  n_blocks_with_median_r95_ge_400 = {n_blocks_ge_400}/12 (refuted if ≥6)", flush=True)
print(f"  n_pairs_with_top5_cos_ge_2.0    = {n_pairs_ge_2}/11 (need ≥6)", flush=True)
print(f"  VERDICT: {verdict}", flush=True)

# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 260,
    "item": 271,
    "verdict": verdict,
    "config": {
        "model": "gpt2",
        "n_calib_tokens": N_TARGET_TOKENS,
        "n_calib_seqs_used": len(calib_ids),
        "seq_len": SEQ_LEN,
        "n_blocks": N_BLOCKS,
        "seed": SEED,
        "n_random_seeds": N_RANDOM_SEEDS,
        "device_model": DEVICE_MODEL,
        "device_linalg": DEVICE_LINALG,
    },
    "per_block": per_block_results,
    "random_baseline": random_baseline,
    "alignment_cross_block": alignment_per_pair,
    "summary": {
        "rank_95_medians_per_block": rank95_medians,
        "n_blocks_median_r95_le_100": n_blocks_le_100,
        "n_blocks_median_r95_ge_400": n_blocks_ge_400,
        "n_pairs_top5_cos_ge_2.0": n_pairs_ge_2,
        "n_pairs_total": len(alignment_per_pair),
        "clause1_passes_rank_95_le_100_in_ge_6_blocks": bool(clause1),
        "clause2_passes_cos_ge_2_in_ge_6_pairs": bool(clause2),
        "refuted_clause_rank_95_ge_400_in_ge_6_blocks": bool(refuted),
    },
    "verdict_logic": {
        "confirmed_iff": "n_blocks_median_r95_le_100 >= 6 AND n_pairs_top5_cos_ge_2.0 >= 6",
        "refuted_iff": "n_blocks_median_r95_ge_400 >= 6",
        "cycle218_random_top5_reference": 1.27,
    },
    "trainable_params": 0,
}
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved {RESULT_PATH}", flush=True)

# ── Summary markdown ─────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 260 — Item 271: Jacobian delta ΔJ_l(x) = J_GELU − J_ReLU")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- n_calib_tokens = {N_TARGET_TOKENS}, n_blocks = {N_BLOCKS}, trainable_params = 0")
lines.append(f"- CONFIRMED iff median rank_95(ΔJ_l) ≤ 100 in ≥6 blocks AND "
             f"mode-1 top-5 cos sum with c_fc reader of block l+1 ≥ 2.0 in ≥6 pairs.")
lines.append(f"- REFUTED iff median rank_95(ΔJ_l) ≥ 400 in ≥6 blocks.")
lines.append("")
lines.append("## Per-block rank_95(ΔJ) over 256 tokens")
lines.append("")
lines.append("| Block | r95 p25 | r95 med | r95 p75 | r95 min | r95 max | avg ΔJ rank_95 | ||ΔJ||_F med |")
lines.append("|------:|--------:|--------:|--------:|--------:|--------:|---------------:|-------------:|")
for b in per_block_results:
    lines.append(f"| {b['block']} | {b['rank_95_p25']:.0f} | {b['rank_95_median']:.0f} | "
                 f"{b['rank_95_p75']:.0f} | {b['rank_95_min']} | {b['rank_95_max']} | "
                 f"{b['avg_dj_rank_95']} | {b['frobenius_median']:.3f} |")
lines.append("")
lines.append(f"Blocks with median rank_95 ≤ 100: **{n_blocks_le_100}/12** (need ≥6 for CONFIRMED).")
lines.append(f"Blocks with median rank_95 ≥ 400: **{n_blocks_ge_400}/12** (≥6 would REFUTE).")
lines.append("")
lines.append("## Cross-block alignment: mode-1 ΔJ_l vs c_fc reader of block l+1")
lines.append("")
lines.append(f"Random baseline (10 seeds): top-5 mean = {random_baseline['top5_mean']:.4f} ± "
             f"{random_baseline['top5_std']:.4f}, top-20 mean = {random_baseline['top20_mean']:.4f} ± "
             f"{random_baseline['top20_std']:.4f}.")
lines.append("Cycle 218 reference: random top-5 sum ≈ 1.27; centred Δ̃ top-5 sum ≈ 2.97.")
lines.append("")
lines.append("| l → l+1 | top-5 cos sum | top-20 cos sum | ≥2.0 | sig 3σ |")
lines.append("|:-------:|--------------:|---------------:|:----:|:------:|")
for p in alignment_per_pair:
    check5 = "✓" if p["passes_threshold_2.0"] else "·"
    sig3 = "✓" if p["significant_vs_random_3sigma"] else "·"
    lines.append(f"| {p['l']}→{p['lp1']} | {p['top5_cos_sum']:.3f} | "
                 f"{p['top20_cos_sum']:.3f} | {check5} | {sig3} |")
lines.append("")
lines.append(f"Pairs with top-5 cos sum ≥ 2.0: **{n_pairs_ge_2}/11** (need ≥6 for CONFIRMED).")
lines.append("")
lines.append("## Verdict logic")
lines.append(f"- Clause 1 (rank_95 ≤ 100 in ≥6 blocks): **{'PASS' if clause1 else 'FAIL'}** "
             f"({n_blocks_le_100}/12)")
lines.append(f"- Clause 2 (mode-1 top-5 cos sum ≥ 2.0 in ≥6 pairs): **{'PASS' if clause2 else 'FAIL'}** "
             f"({n_pairs_ge_2}/11)")
lines.append(f"- REFUTED clause (rank_95 ≥ 400 in ≥6 blocks): **{'TRIGGERED' if refuted else 'not triggered'}** "
             f"({n_blocks_ge_400}/12)")
lines.append(f"- **Verdict: {verdict}**")
lines.append("")
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
