"""
Cycle 267 — Item 277: Block 11 c_proj output rank decomposition (analysis-only).

For blocks {1, 5, 11}:
  - Hook c_proj OUTPUT (post-projection residual contribution, shape (N, 768)).
  - Covariance C = mlp_out.T @ mlp_out / N (768x768), SVD -> eff_rank, top-k energy.
  - Top-20 right singular vectors V_mlp (eigenvectors of C).
  - SVD of c_proj.weight (3072, 768) -> top-20 right singular vectors V_wproj (R^768).
  - Cosines of top-20 V_mlp with lm_head rows; restrict to top-200 rows (by row norm).
  - Report median over top-5 V_mlp of (max cosine over top-200 lm_head rows).

Verdict (block 11 only):
  CONFIRMED if eff_rank <= 50 OR med_top5_cos_top200 >= 0.5
  REFUTED   if eff_rank >= 200 AND med_top5_cos_top200 <= 0.2
  INCONCLUSIVE otherwise.
"""
import json
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# Config
N_CALIB = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BLOCKS = [1, 5, 11]
TARGET_BLOCK = 11
TOPK_ENERGY = [1, 5, 20, 50, 100, 200]
TOP_V_FOR_ALIGNMENT = 20
TOP_V_MEDIAN = 5
TOP_LMHEAD_ROWS = 200
OUTPUT_JSON = "autoresearch/results/cycle_267_block11_cproj_decomp.json"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# Load model
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# lm_head weight (50257, 768); in GPT-2 this is tied to wte. Use lm_head.weight.
lm_head_W = model.lm_head.weight.detach().float().cpu()  # (50257, 768)
print(f"lm_head weight shape: {tuple(lm_head_W.shape)}")
assert lm_head_W.shape == (50257, 768), f"lm_head shape {lm_head_W.shape} != (50257, 768)"

# Top-200 lm_head rows by L2 row norm
lm_row_norms = lm_head_W.norm(dim=1)  # (50257,)
top_row_idx = torch.topk(lm_row_norms, TOP_LMHEAD_ROWS).indices  # (200,)
lm_head_top = lm_head_W[top_row_idx]  # (200, 768)
lm_head_top_normed = F.normalize(lm_head_top, dim=1)  # (200, 768)
lm_head_all_normed = F.normalize(lm_head_W, dim=1)  # (50257, 768)
print(f"lm_head row norm range: [{float(lm_row_norms.min()):.4f}, {float(lm_row_norms.max()):.4f}]")
print(f"lm_head top-200 row norm range: [{float(lm_head_top.norm(dim=1).min()):.4f}, {float(lm_head_top.norm(dim=1).max()):.4f}]")

# Load calib data
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
print(f"Calibration: {len(calib_ids)} seqs x {SEQ_LEN} tokens = {expected_n_tokens} tokens")

# Capture c_proj OUTPUT (mlp_out) per block
mlp_out_acts = {b: [] for b in BLOCKS}
hooks = []
for b in BLOCKS:
    def mk_hook(bb):
        def fn(m, inp, out):
            # out: (B, T, 768) — post c_proj, pre residual-add
            x = out.detach().float().cpu().reshape(-1, 768)
            mlp_out_acts[bb].append(x)
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
    mlp_out_acts[b] = torch.cat(mlp_out_acts[b], dim=0)
    assert mlp_out_acts[b].shape == (expected_n_tokens, 768), (
        f"block {b} shape {mlp_out_acts[b].shape} != {(expected_n_tokens, 768)}"
    )
print(f"Captured mlp_out shape: {mlp_out_acts[BLOCKS[0]].shape}")

# Per-block analysis
per_block = {}
for b in BLOCKS:
    print(f"\n=== Block {b} ===")
    X = mlp_out_acts[b]  # (25600, 768) float32 CPU
    N = X.shape[0]

    # Covariance 768x768
    t0 = time.time()
    C = (X.T @ X) / N  # (768, 768)
    # Symmetric eigendecomp gives eigenvalues and eigenvectors of cov
    evals, evecs = torch.linalg.eigh(C)  # evals ascending
    # Descending
    evals = evals.flip(0)  # (768,)
    evecs = evecs.flip(1)  # (768, 768) columns in descending order
    evals = evals.clamp_min(0.0)  # numerical
    total = float(evals.sum())
    p = evals / (evals.sum() + 1e-30)
    H = -(p * (p + 1e-30).log()).sum()
    eff_rank = float(torch.exp(H))
    print(f"  eff_rank = {eff_rank:.2f}  (eigh {time.time()-t0:.1f}s)")

    # Top-k energy
    topk_energy = {}
    for k in TOPK_ENERGY:
        if k <= evals.shape[0]:
            topk_energy[k] = float(evals[:k].sum() / (total + 1e-30))
        else:
            topk_energy[k] = 1.0
        print(f"  top-{k:3d} energy = {topk_energy[k]:.4f}")

    # V_mlp: top-20 eigenvectors (cols of evecs)
    V_mlp = evecs[:, :TOP_V_FOR_ALIGNMENT]  # (768, 20)
    # These are unit-norm since eigh returns orthonormal eigenvectors
    vm_norms = V_mlp.norm(dim=0)
    assert torch.allclose(vm_norms, torch.ones_like(vm_norms), atol=1e-4), (
        f"block {b} V_mlp cols not unit norm: {vm_norms}"
    )

    # SVD of c_proj.weight — HF stores (3072, 768), F.linear does x @ W.T
    W_proj = model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()  # (3072, 768)
    assert W_proj.shape == (3072, 768), f"block {b} c_proj shape {W_proj.shape} != (3072, 768)"
    t0 = time.time()
    U_wp, S_wp, Vh_wp = torch.linalg.svd(W_proj, full_matrices=False)
    # Vh_wp: (768, 768) — rows are right singular vectors; columns of V = Vh_wp.T
    V_wp = Vh_wp.T[:, :TOP_V_FOR_ALIGNMENT]  # (768, 20)
    print(f"  c_proj SVD {time.time()-t0:.1f}s, top sv = {float(S_wp[0]):.4f}, s20 = {float(S_wp[TOP_V_FOR_ALIGNMENT-1]):.4f}")

    # Cosines V_mlp (20, 768) vs lm_head
    V_mlp_T = V_mlp.T  # (20, 768) already unit norm
    # Top-200 lm_head rows
    cos_top200 = V_mlp_T @ lm_head_top_normed.T  # (20, 200)
    max_cos_top200_per_v = cos_top200.abs().max(dim=1).values  # (20,)
    # Top-5 V_mlp median of max cosine
    med_top5_cos_top200 = float(max_cos_top200_per_v[:TOP_V_MEDIAN].median())
    # All lm_head rows
    cos_all = V_mlp_T @ lm_head_all_normed.T  # (20, 50257)
    max_cos_all_per_v = cos_all.abs().max(dim=1).values  # (20,)
    med_top5_cos_all = float(max_cos_all_per_v[:TOP_V_MEDIAN].median())
    print(f"  V_mlp top-5 max-cos top-200 lm_head: {max_cos_top200_per_v[:TOP_V_MEDIAN].tolist()} -> median {med_top5_cos_top200:.4f}")
    print(f"  V_mlp top-5 max-cos ALL   lm_head: {max_cos_all_per_v[:TOP_V_MEDIAN].tolist()} -> median {med_top5_cos_all:.4f}")

    # Same for V_wp (operator-level)
    V_wp_T = V_wp.T  # (20, 768) unit norm
    vw_norms = V_wp_T.norm(dim=1)
    assert torch.allclose(vw_norms, torch.ones_like(vw_norms), atol=1e-4)
    cos_wp_top200 = V_wp_T @ lm_head_top_normed.T  # (20, 200)
    max_cos_wp_top200_per_v = cos_wp_top200.abs().max(dim=1).values
    med_top5_cos_wp_top200 = float(max_cos_wp_top200_per_v[:TOP_V_MEDIAN].median())
    cos_wp_all = V_wp_T @ lm_head_all_normed.T
    max_cos_wp_all_per_v = cos_wp_all.abs().max(dim=1).values
    med_top5_cos_wp_all = float(max_cos_wp_all_per_v[:TOP_V_MEDIAN].median())
    print(f"  V_wp  top-5 max-cos top-200 lm_head: {max_cos_wp_top200_per_v[:TOP_V_MEDIAN].tolist()} -> median {med_top5_cos_wp_top200:.4f}")

    per_block[b] = {
        "eff_rank": eff_rank,
        "total_energy": total,
        "top1_eigval": float(evals[0]),
        "topk_energy": {str(k): v for k, v in topk_energy.items()},
        "top20_mlp_eigvals": evals[:20].tolist(),
        "wproj_top20_sv": S_wp[:20].tolist(),
        # V_mlp alignment
        "max_cos_top200_per_v_mlp": max_cos_top200_per_v.tolist(),
        "max_cos_all_per_v_mlp": max_cos_all_per_v.tolist(),
        "med_top5_cos_top200_v_mlp": med_top5_cos_top200,
        "med_top5_cos_all_v_mlp": med_top5_cos_all,
        # V_wp alignment
        "max_cos_top200_per_v_wp": max_cos_wp_top200_per_v.tolist(),
        "max_cos_all_per_v_wp": max_cos_wp_all_per_v.tolist(),
        "med_top5_cos_top200_v_wp": med_top5_cos_wp_top200,
        "med_top5_cos_all_v_wp": med_top5_cos_wp_all,
    }

# Verdict — block 11 only
b11 = per_block[TARGET_BLOCK]
eff_rank_b11 = b11["eff_rank"]
med_cos_b11 = b11["med_top5_cos_top200_v_mlp"]
print(f"\n=== Verdict (block {TARGET_BLOCK}) ===")
print(f"  eff_rank         = {eff_rank_b11:.2f}")
print(f"  med top5 cos top200 = {med_cos_b11:.4f}")

if eff_rank_b11 <= 50.0 or med_cos_b11 >= 0.5:
    verdict = "CONFIRMED"
elif eff_rank_b11 >= 200.0 and med_cos_b11 <= 0.2:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"  VERDICT: {verdict}")

# Write JSON
result = {
    "cycle": 267,
    "item": 277,
    "verdict": verdict,
    "config": {
        "model": "gpt2",
        "n_calib": N_CALIB,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "device": DEVICE,
        "blocks": BLOCKS,
        "target_block": TARGET_BLOCK,
        "topk_energy": TOPK_ENERGY,
        "top_v_for_alignment": TOP_V_FOR_ALIGNMENT,
        "top_v_median": TOP_V_MEDIAN,
        "top_lmhead_rows": TOP_LMHEAD_ROWS,
    },
    "verdict_logic": {
        "eff_rank_b11": eff_rank_b11,
        "med_top5_cos_top200_b11": med_cos_b11,
        "threshold_confirmed_eff_rank_le": 50.0,
        "threshold_confirmed_cos_ge": 0.5,
        "threshold_refuted_eff_rank_ge": 200.0,
        "threshold_refuted_cos_le": 0.2,
        "verdict": verdict,
    },
    "per_block": {str(b): per_block[b] for b in BLOCKS},
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_JSON}")
