"""
Cycle 261 — Item 272: W_down column geometry vs per-channel sign frequency σ_c.

The rescue-note experiment from 2026-04-02:
    pos_output = Σ_{c: z_c>0} α_c w_c   (w_c are W_down columns)
    neg_output = Σ_{c: z_c<0} (−β_c) w_c

For each block l ∈ {0..11}:
  - σ_c = E_x[1{z_c(x) > 0}] over the calib slice (200×128 wikitext-2 val tokens)
  - Split channels into:
        mostly_neg : σ ∈ [0.0, 0.1]
        balanced   : σ ∈ [0.45, 0.55]
        mostly_pos : σ ∈ [0.9, 1.0]
  - W_down rows = c_proj.weight (3072, 768) — each row is a column of W_down in R^768
  - Per group: SVD spectrum, top-20 sv, rank_95, top-1 SV energy fraction, top-20 right SVs.
  - Principal angles (deg) between mostly_pos and mostly_neg subspaces (top-20 right SVs).
  - Block 2 special: mostly_pos top-1 SV energy fraction (cycle 174 rank-1 check).
  - Null distribution: 100 shuffles of channel labels preserving (n_pos, n_neg).

Verdict (item 272):
  - CONFIRMED iff
        (a) block-2 mostly_pos top-1 SV energy ≥ 0.70
        AND (b) median principal angle (pos vs neg) decreasing monotonically from
                block 1 to block 11 by ≥ 30° (angle[1] − angle[11] ≥ 30°)
        AND (c) each of those angles outside null mean ± 2σ.
  - REFUTED iff all 12 blocks have median angle within ±2σ of null mean.
  - INCONCLUSIVE otherwise.
"""
import json
import os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
N_CALIB_SEQS_FOR_LOADER = 200
SEQ_LEN = 128
SEED = 42
N_NULL_SHUFFLES = 100
DEVICE_MODEL = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_LINALG = "cpu"
N_BLOCKS = 12
D = 768
N_CHANNELS = 3072
TOPK_RIGHT_SV = 20

SIGMA_NEG_HI = 0.10
SIGMA_BAL_LO = 0.45
SIGMA_BAL_HI = 0.55
SIGMA_POS_LO = 0.90

RESULT_PATH = "autoresearch/results/cycle_261_wdown_sign_geometry.json"
SUMMARY_PATH = "autoresearch/results/cycle_261_summary.md"

torch.manual_seed(SEED)
print(f"Model device: {DEVICE_MODEL}", flush=True)
print(f"Linalg device: {DEVICE_LINALG}", flush=True)

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE_MODEL)
model.eval()

# ── Calib data: wikitext-2 val, first 200 sequences × 128 tokens ─────────────
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
calib_ids = [t.to(DEVICE_MODEL) for t in all_ids[:N_CALIB_SEQS_FOR_LOADER]]
N_TOKENS = len(calib_ids) * SEQ_LEN
print(f"Calib: {len(calib_ids)} sequences × {SEQ_LEN} = {N_TOKENS} tokens", flush=True)
assert len(calib_ids) == N_CALIB_SEQS_FOR_LOADER

# ── Hook c_fc input on every block ───────────────────────────────────────────
cfc_stash = {l: [] for l in range(N_BLOCKS)}

def make_cfc_hook(l):
    def fn(module, input, output):
        cfc_stash[l].append(input[0].detach().float().cpu())
    return fn

hooks = []
for l in range(N_BLOCKS):
    h = model.transformer.h[l].mlp.c_fc.register_forward_hook(make_cfc_hook(l))
    hooks.append(h)

print("Running forward passes to capture c_fc inputs (200 seqs)...", flush=True)
with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)
for h in hooks:
    h.remove()

ln2_out_per_block = {}
for l in range(N_BLOCKS):
    x = torch.cat(cfc_stash[l], dim=0).reshape(-1, D)
    assert x.shape == (N_TOKENS, D), f"block {l} got {x.shape}"
    ln2_out_per_block[l] = x.contiguous()
cfc_stash = None
print(f"Captured ln2_out: {ln2_out_per_block[0].shape} per block", flush=True)

# ── Cache weights on linalg device ───────────────────────────────────────────
W_fc_per_block = {}
b_fc_per_block = {}
W_proj_per_block = {}
for l in range(N_BLOCKS):
    block = model.transformer.h[l]
    W_fc_per_block[l] = block.mlp.c_fc.weight.detach().float().to(DEVICE_LINALG)        # (768, 3072)
    b_fc_per_block[l] = block.mlp.c_fc.bias.detach().float().to(DEVICE_LINALG)          # (3072,)
    W_proj_per_block[l] = block.mlp.c_proj.weight.detach().float().to(DEVICE_LINALG)    # (3072, 768)
    assert W_fc_per_block[l].shape == (D, N_CHANNELS)
    assert W_proj_per_block[l].shape == (N_CHANNELS, D)

# ── Free model after we have weights + activations ───────────────────────────
del model
import gc; gc.collect()

# ── Helper: top-20 right singular vectors of a row subset of W_down ──────────
@torch.no_grad()
def group_geometry(W_rows: torch.Tensor, k_right: int = TOPK_RIGHT_SV):
    """W_rows: (n, 768). Returns dict with svdvals, rank_95, top1_energy, V_top
    where V_top is (768, k_used). k_used = min(n, k_right, 768)."""
    n = W_rows.shape[0]
    if n == 0:
        return {
            "n": 0,
            "k_used": 0,
            "svdvals_top20": [],
            "all_svdvals_count": 0,
            "rank_95": 0,
            "top1_energy_fraction": float("nan"),
            "V_top": torch.zeros(D, 0),
        }
    U, S, Vh = torch.linalg.svd(W_rows, full_matrices=False)  # S len min(n, 768)
    energy = (S * S)
    total = float(energy.sum().item())
    cum = energy.cumsum(0) / energy.sum()
    rank_95 = int((cum < 0.95).sum().item()) + 1
    top1_energy = float(energy[0].item() / total) if total > 0 else float("nan")
    k_used = min(n, k_right, D)
    V_top = Vh[:k_used].T.contiguous()  # (768, k_used)
    return {
        "n": n,
        "k_used": k_used,
        "svdvals_top20": S[:20].tolist(),
        "all_svdvals_count": int(S.shape[0]),
        "rank_95": rank_95,
        "top1_energy_fraction": top1_energy,
        "V_top": V_top,
    }


@torch.no_grad()
def principal_angles_deg(V_a: torch.Tensor, V_b: torch.Tensor):
    """V_a (768, k_a), V_b (768, k_b) — both column-orthonormal.
    Returns angles in degrees (length min(k_a, k_b)). Empty tensor if k_a or k_b == 0."""
    if V_a.shape[1] == 0 or V_b.shape[1] == 0:
        return torch.empty(0)
    M = V_a.T @ V_b  # (k_a, k_b)
    cos = torch.linalg.svdvals(M)
    cos = cos.clamp(-1.0, 1.0)
    angles_rad = torch.arccos(cos)
    angles_deg = angles_rad * (180.0 / torch.pi)
    return angles_deg


# ── Per-block analysis ───────────────────────────────────────────────────────
per_block = []
for l in range(N_BLOCKS):
    print(f"\n[block {l}] computing σ_c and group geometry...", flush=True)
    ln2_out = ln2_out_per_block[l].to(DEVICE_LINALG)
    W_fc = W_fc_per_block[l]
    b_fc = b_fc_per_block[l]
    W_down_rows = W_proj_per_block[l]  # (3072, 768)

    # Pre-activation z(x) = ln2_out @ W_fc + b_fc
    preact = ln2_out @ W_fc + b_fc  # (N_TOKENS, 3072)
    sigma_c = (preact > 0).float().mean(dim=0)  # (3072,)
    assert sigma_c.shape == (N_CHANNELS,)

    mask_neg = (sigma_c >= 0.0) & (sigma_c <= SIGMA_NEG_HI)
    mask_bal = (sigma_c >= SIGMA_BAL_LO) & (sigma_c <= SIGMA_BAL_HI)
    mask_pos = (sigma_c >= SIGMA_POS_LO) & (sigma_c <= 1.0)
    n_neg = int(mask_neg.sum().item())
    n_bal = int(mask_bal.sum().item())
    n_pos = int(mask_pos.sum().item())
    print(f"  σ_c stats: min={float(sigma_c.min()):.3f} max={float(sigma_c.max()):.3f} "
          f"mean={float(sigma_c.mean()):.3f}", flush=True)
    print(f"  group sizes: neg={n_neg}  balanced={n_bal}  pos={n_pos}", flush=True)

    W_neg = W_down_rows[mask_neg]
    W_bal = W_down_rows[mask_bal]
    W_pos = W_down_rows[mask_pos]

    g_neg = group_geometry(W_neg)
    g_bal = group_geometry(W_bal)
    g_pos = group_geometry(W_pos)

    # Principal angles between mostly_pos and mostly_neg
    angles = principal_angles_deg(g_pos["V_top"], g_neg["V_top"])
    if angles.numel() > 0:
        median_angle_deg = float(angles.median().item())
        mean_angle_deg = float(angles.mean().item())
    else:
        median_angle_deg = float("nan")
        mean_angle_deg = float("nan")
    print(f"  pos↔neg principal angles (deg): k={angles.numel()}  "
          f"median={median_angle_deg:.2f}  mean={mean_angle_deg:.2f}", flush=True)

    # ── Null distribution: shuffle channel labels preserving (n_pos, n_neg) ──
    null_medians = []
    null_means = []
    n_total = N_CHANNELS
    for s in range(N_NULL_SHUFFLES):
        g_perm = torch.Generator().manual_seed(SEED + s)
        perm = torch.randperm(n_total, generator=g_perm)
        # Partition: first n_neg → neg, next n_pos → pos. Disjoint by construction.
        neg_idx = perm[:n_neg]
        pos_idx = perm[n_neg:n_neg + n_pos]
        W_neg_p = W_down_rows[neg_idx]
        W_pos_p = W_down_rows[pos_idx]
        g_neg_p = group_geometry(W_neg_p)
        g_pos_p = group_geometry(W_pos_p)
        angles_p = principal_angles_deg(g_pos_p["V_top"], g_neg_p["V_top"])
        if angles_p.numel() > 0:
            null_medians.append(float(angles_p.median().item()))
            null_means.append(float(angles_p.mean().item()))
    null_medians_t = torch.tensor(null_medians, dtype=torch.float64)
    null_mean = float(null_medians_t.mean().item()) if null_medians_t.numel() > 0 else float("nan")
    null_std = float(null_medians_t.std(unbiased=False).item()) if null_medians_t.numel() > 0 else float("nan")
    if null_std > 0 and not (median_angle_deg != median_angle_deg):
        z = (median_angle_deg - null_mean) / null_std
    else:
        z = float("nan")
    inside_null_2sigma = (
        not (median_angle_deg != median_angle_deg)
        and (median_angle_deg >= null_mean - 2 * null_std)
        and (median_angle_deg <= null_mean + 2 * null_std)
    )
    print(f"  null (100 shuffles): mean={null_mean:.2f}±{null_std:.3f}  "
          f"z={z:.3f}  inside_2σ={inside_null_2sigma}", flush=True)

    rec = {
        "block": l,
        "n_neg": n_neg,
        "n_bal": n_bal,
        "n_pos": n_pos,
        "sigma_c_stats": {
            "min": float(sigma_c.min().item()),
            "max": float(sigma_c.max().item()),
            "mean": float(sigma_c.mean().item()),
            "median": float(sigma_c.median().item()),
        },
        "mostly_neg": {k: v for k, v in g_neg.items() if k != "V_top"},
        "balanced":   {k: v for k, v in g_bal.items() if k != "V_top"},
        "mostly_pos": {k: v for k, v in g_pos.items() if k != "V_top"},
        "principal_angles_pos_vs_neg_deg": angles.tolist(),
        "median_principal_angle_pos_vs_neg_deg": median_angle_deg,
        "mean_principal_angle_pos_vs_neg_deg": mean_angle_deg,
        "null": {
            "n_shuffles": N_NULL_SHUFFLES,
            "mean_of_medians_deg": null_mean,
            "std_of_medians_deg": null_std,
            "z_score_real_vs_null": z,
            "inside_2sigma": bool(inside_null_2sigma),
        },
    }
    per_block.append(rec)

# ── Block 2 special check: mostly_pos top-1 SV energy fraction ───────────────
b2_pos_top1 = per_block[2]["mostly_pos"]["top1_energy_fraction"]
print(f"\n[block 2] mostly_pos top-1 SV energy fraction: {b2_pos_top1:.4f}", flush=True)

# ── Verdict ──────────────────────────────────────────────────────────────────
median_angles = [pb["median_principal_angle_pos_vs_neg_deg"] for pb in per_block]
print(f"\nMedian principal angles (deg) per block:", flush=True)
for l, a in enumerate(median_angles):
    print(f"  block {l}: {a:.2f}", flush=True)

# Clause (a): block 2 pos top-1 ≥ 0.70
clause_a = (b2_pos_top1 == b2_pos_top1) and (b2_pos_top1 >= 0.70)

# Clause (b): monotone decrease block 1 → block 11 with total ≥ 30°
slice_1_11 = median_angles[1:12]  # blocks 1..11 inclusive
strictly_monotone_decreasing = all(
    slice_1_11[i] >= slice_1_11[i + 1] for i in range(len(slice_1_11) - 1)
)
total_drop = slice_1_11[0] - slice_1_11[-1]
clause_b = strictly_monotone_decreasing and (total_drop >= 30.0)

# Clause (c): each angle in blocks 1..11 outside null mean ± 2σ
clause_c = all(not per_block[l]["null"]["inside_2sigma"] for l in range(1, 12))

confirmed = clause_a and clause_b and clause_c

# REFUTED: all 12 blocks within ±2σ of null
all_inside = all(per_block[l]["null"]["inside_2sigma"] for l in range(N_BLOCKS))
refuted = all_inside

if confirmed:
    verdict = "CONFIRMED"
elif refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nClause (a) block-2 pos top-1 ≥ 0.70: {clause_a} (value={b2_pos_top1:.4f})")
print(f"Clause (b) monotone decrease blocks 1→11 ≥ 30°: {clause_b} "
      f"(monotone={strictly_monotone_decreasing} total_drop={total_drop:.2f}°)")
print(f"Clause (c) all blocks 1..11 outside null ±2σ: {clause_c}")
print(f"REFUTED clause (all 12 inside ±2σ): {all_inside}")
print(f"\nVERDICT: {verdict}")

# ── Save JSON ────────────────────────────────────────────────────────────────
config = {
    "model": "gpt2",
    "n_calib_seqs": N_CALIB_SEQS_FOR_LOADER,
    "seq_len": SEQ_LEN,
    "n_calib_tokens": N_TOKENS,
    "seed": SEED,
    "n_null_shuffles": N_NULL_SHUFFLES,
    "device_model": DEVICE_MODEL,
    "device_linalg": DEVICE_LINALG,
    "n_blocks": N_BLOCKS,
    "d": D,
    "n_channels": N_CHANNELS,
    "topk_right_sv": TOPK_RIGHT_SV,
    "sigma_thresholds": {
        "neg": [0.0, SIGMA_NEG_HI],
        "balanced": [SIGMA_BAL_LO, SIGMA_BAL_HI],
        "pos": [SIGMA_POS_LO, 1.0],
    },
}

verdict_logic = {
    "clause_a_block2_pos_top1_ge_070": bool(clause_a),
    "clause_a_value": float(b2_pos_top1) if b2_pos_top1 == b2_pos_top1 else None,
    "clause_b_monotone_decrease_blocks_1_to_11_ge_30deg": bool(clause_b),
    "clause_b_strictly_monotone": bool(strictly_monotone_decreasing),
    "clause_b_total_drop_deg": float(total_drop),
    "clause_c_all_blocks_1_11_outside_null_2sigma": bool(clause_c),
    "confirmed": bool(confirmed),
    "refuted_all_12_inside_null_2sigma": bool(all_inside),
    "verdict": verdict,
}

result = {
    "cycle": 261,
    "item": 272,
    "verdict": verdict,
    "config": config,
    "blocks": per_block,
    "block_2_pos_top1_energy_fraction": float(b2_pos_top1) if b2_pos_top1 == b2_pos_top1 else None,
    "median_principal_angle_pos_vs_neg_deg_per_block": median_angles,
    "verdict_logic": verdict_logic,
}

os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
with open(RESULT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {RESULT_PATH}", flush=True)

# ── Summary md ───────────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 261 — W_down sign-frequency geometry (Item 272)\n")
lines.append(f"\n**Verdict: {verdict}**\n")
lines.append("\n## Setup\n")
lines.append(
    f"- GPT-2 small, calib = {N_CALIB_SEQS_FOR_LOADER} sequences × {SEQ_LEN} = "
    f"{N_TOKENS} tokens, wikitext-2 val (seed {SEED}).\n"
)
lines.append(
    f"- For each block l ∈ {{0..{N_BLOCKS-1}}}: σ_c = E_x[1{{z_c>0}}], split channels into "
    f"σ ∈ [0, {SIGMA_NEG_HI}] / [{SIGMA_BAL_LO}, {SIGMA_BAL_HI}] / [{SIGMA_POS_LO}, 1.0]; "
    f"SVD of corresponding W_down row subsets; principal angles (top-{TOPK_RIGHT_SV} right SVs) "
    f"between mostly_pos and mostly_neg.\n"
)
lines.append(
    f"- Null: {N_NULL_SHUFFLES} shuffles preserving (n_pos, n_neg) per block.\n"
)
lines.append("\n## Per-block group sizes and angles\n\n")
lines.append("| block | n_neg | n_bal | n_pos | rank95_pos | rank95_neg | pos_top1 | median_angle° | null_mean±std° | inside_2σ |\n")
lines.append("|------:|-----:|-----:|-----:|----------:|----------:|--------:|------------:|---------------:|:--------:|\n")
for pb in per_block:
    l = pb["block"]
    nn, nb, np_ = pb["n_neg"], pb["n_bal"], pb["n_pos"]
    r95p = pb["mostly_pos"]["rank_95"]
    r95n = pb["mostly_neg"]["rank_95"]
    t1p = pb["mostly_pos"]["top1_energy_fraction"]
    ma = pb["median_principal_angle_pos_vs_neg_deg"]
    nm = pb["null"]["mean_of_medians_deg"]
    ns = pb["null"]["std_of_medians_deg"]
    ins = "Y" if pb["null"]["inside_2sigma"] else "N"
    lines.append(
        f"| {l} | {nn} | {nb} | {np_} | {r95p} | {r95n} | {t1p:.3f} | {ma:.2f} | "
        f"{nm:.2f}±{ns:.2f} | {ins} |\n"
    )

lines.append("\n## Block 2 rank-1 check (cycle 174)\n")
lines.append(
    f"- mostly_pos (σ ≥ {SIGMA_POS_LO}) at block 2: n={per_block[2]['n_pos']}, "
    f"top-1 SV energy fraction = **{b2_pos_top1:.4f}**, "
    f"rank_95 = {per_block[2]['mostly_pos']['rank_95']}.\n"
)
lines.append(
    f"- Threshold for clause (a): top-1 ≥ 0.70 → **{'PASS' if clause_a else 'FAIL'}**.\n"
)

lines.append("\n## Verdict logic\n")
lines.append(
    f"- (a) block-2 pos top-1 ≥ 0.70: **{clause_a}** (value {b2_pos_top1:.4f})\n"
)
lines.append(
    f"- (b) median angle blocks 1→11 strictly monotone decreasing AND total drop ≥ 30°: "
    f"**{clause_b}** (monotone={strictly_monotone_decreasing}, total_drop={total_drop:.2f}°)\n"
)
lines.append(
    f"- (c) blocks 1..11 each outside null mean ± 2σ: **{clause_c}**\n"
)
lines.append(
    f"- REFUTED clause (all 12 blocks inside null ±2σ): **{all_inside}**\n"
)
lines.append(f"- **Verdict: {verdict}**\n")

with open(SUMMARY_PATH, "w") as f:
    f.writelines(lines)
print(f"Wrote {SUMMARY_PATH}", flush=True)
print(f"\nDONE — verdict={verdict}", flush=True)
