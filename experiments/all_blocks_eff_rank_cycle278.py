"""
Cycle 278 — Item 288: All-blocks c_proj output eff_rank census.

Extends cycle 267 from blocks {1, 5, 11} to all 12 GPT-2 small blocks.

For each block l ∈ {0..11}:
  - Hook c_proj OUTPUT (post-projection residual contribution, shape (N, 768)).
  - Covariance C = mlp_out.T @ mlp_out / N (768x768), SVD -> eff_rank, top-k energy.
  - SVD of c_proj.weight (3072, 768) -> top-20 singular values (operator side).

Verdict (item 288 verbatim):
  CONFIRMED iff (block 4 eff_rank <= 50 OR block 7 eff_rank <= 50)
            AND ALL of {2, 3, 6, 8, 9, 10} eff_rank >= 100
  REFUTED   iff blocks 4 AND 7 BOTH eff_rank >= 100
  INCONCLUSIVE otherwise

Pure analysis. 0 trainable params. No Adam, no retraining.
"""
import json
import time
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# Config
N_CALIB = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BLOCKS = list(range(12))  # all 12 blocks
TOPK_ENERGY = [1, 5, 20, 50, 100, 200]
TOP_V_FOR_REPORT = 20
OUTPUT_JSON = "autoresearch/results/cycle_278_all_blocks_eff_rank.json"

# Cycle 267 reference values (sanity check) for blocks {1, 5, 11}
CYCLE_267_EFF_RANK = {
    1: 1.9448049068450928,
    5: 179.1913299560547,
    11: 22.488744735717773,
}

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")
print(f"BLOCKS: {BLOCKS}")

# Load model
print("Loading GPT-2...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

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

print("Running forward passes (single pass collects all 12 blocks)...")
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
print(f"Captured mlp_out shape per block: {mlp_out_acts[BLOCKS[0]].shape}")

# Per-block analysis
per_block = {}
for b in BLOCKS:
    print(f"\n=== Block {b} ===")
    X = mlp_out_acts[b]  # (25600, 768) float32 CPU
    N = X.shape[0]

    # Covariance 768x768
    t0 = time.time()
    C = (X.T @ X) / N  # (768, 768)
    evals, evecs = torch.linalg.eigh(C)  # ascending
    evals = evals.flip(0)  # descending
    evecs = evecs.flip(1)
    evals = evals.clamp_min(0.0)
    total = float(evals.sum())
    p = evals / (evals.sum() + 1e-30)
    H = -(p * (p + 1e-30).log()).sum()
    eff_rank = float(torch.exp(H))
    print(f"  eff_rank = {eff_rank:.4f}  (eigh {time.time()-t0:.1f}s)")

    # Top-k energy
    topk_energy = {}
    for k in TOPK_ENERGY:
        if k <= evals.shape[0]:
            topk_energy[k] = float(evals[:k].sum() / (total + 1e-30))
        else:
            topk_energy[k] = 1.0
        print(f"  top-{k:3d} energy = {topk_energy[k]:.4f}")

    # SVD of c_proj.weight (3072, 768) -> top-20 sv (operator side)
    W_proj = model.transformer.h[b].mlp.c_proj.weight.detach().float().cpu()
    assert W_proj.shape == (3072, 768), f"block {b} c_proj shape {W_proj.shape} != (3072, 768)"
    t0 = time.time()
    _, S_wp, _ = torch.linalg.svd(W_proj, full_matrices=False)
    print(f"  c_proj SVD {time.time()-t0:.1f}s, sv1={float(S_wp[0]):.4f}, sv20={float(S_wp[TOP_V_FOR_REPORT-1]):.4f}")

    per_block[b] = {
        "eff_rank": eff_rank,
        "total_energy": total,
        "top1_eigval": float(evals[0]),
        "topk_energy": {str(k): v for k, v in topk_energy.items()},
        "top20_mlp_eigvals": evals[:TOP_V_FOR_REPORT].tolist(),
        "wproj_top20_sv": S_wp[:TOP_V_FOR_REPORT].tolist(),
    }

# Sanity check: blocks {1, 5, 11} should reproduce cycle 267 within 1e-3
print("\n=== Sanity check vs cycle 267 ===")
sanity = {}
for b, ref in CYCLE_267_EFF_RANK.items():
    cur = per_block[b]["eff_rank"]
    delta = abs(cur - ref)
    print(f"  block {b:2d}: cycle 267 = {ref:.4f}, cycle 278 = {cur:.4f}, |delta| = {delta:.6f}")
    assert delta < 1e-3, f"block {b} eff_rank diverged from cycle 267 by {delta} (> 1e-3)"
    sanity[b] = {"cycle_267": ref, "cycle_278": cur, "abs_delta": delta}
print("  ALL SANITY CHECKS PASS")

# Verdict — item 288 verbatim
b4_er = per_block[4]["eff_rank"]
b7_er = per_block[7]["eff_rank"]
mid_blocks = [2, 3, 6, 8, 9, 10]
mid_er = {b: per_block[b]["eff_rank"] for b in mid_blocks}

confirmed_first_clause = (b4_er <= 50.0) or (b7_er <= 50.0)
confirmed_second_clause = all(mid_er[b] >= 100.0 for b in mid_blocks)
refuted_clause = (b4_er >= 100.0) and (b7_er >= 100.0)

print(f"\n=== Verdict ===")
print(f"  block 4 eff_rank = {b4_er:.4f}")
print(f"  block 7 eff_rank = {b7_er:.4f}")
print(f"  mid-block eff_ranks: {[(b, round(mid_er[b], 4)) for b in mid_blocks]}")
print(f"  CONFIRMED clause 1 (b4 <= 50 OR b7 <= 50): {confirmed_first_clause}")
print(f"  CONFIRMED clause 2 (mid all >= 100):       {confirmed_second_clause}")
print(f"  REFUTED clause   (b4 AND b7 both >= 100):  {refuted_clause}")

if confirmed_first_clause and confirmed_second_clause:
    verdict = "CONFIRMED"
elif refuted_clause:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"  VERDICT: {verdict}")

# Profile summary
print("\n=== 12-block eff_rank profile ===")
for b in BLOCKS:
    print(f"  block {b:2d}: eff_rank = {per_block[b]['eff_rank']:8.4f}, top1_energy = {per_block[b]['topk_energy']['1']:.4f}")

# Write JSON
result = {
    "cycle": 278,
    "item": 288,
    "verdict": verdict,
    "config": {
        "model": "gpt2",
        "n_calib": N_CALIB,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "device": DEVICE,
        "blocks": BLOCKS,
        "topk_energy": TOPK_ENERGY,
        "top_v_for_report": TOP_V_FOR_REPORT,
    },
    "verdict_logic": {
        "block4_eff_rank": b4_er,
        "block7_eff_rank": b7_er,
        "mid_block_eff_ranks": {str(b): mid_er[b] for b in mid_blocks},
        "threshold_confirmed_block4_or_7_le": 50.0,
        "threshold_confirmed_mid_blocks_ge": 100.0,
        "threshold_refuted_block4_and_7_ge": 100.0,
        "confirmed_first_clause": confirmed_first_clause,
        "confirmed_second_clause": confirmed_second_clause,
        "refuted_clause": refuted_clause,
        "verdict": verdict,
    },
    "sanity_check_vs_cycle_267": {str(b): sanity[b] for b in CYCLE_267_EFF_RANK},
    "per_block": {str(b): per_block[b] for b in BLOCKS},
    "twelve_block_eff_rank_profile": {str(b): per_block[b]["eff_rank"] for b in BLOCKS},
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_JSON}")
