"""
Cycle 227 — Item 237: Δ̃_l specificity control vs non-adjacent c_fc(j).

Cycle 218 found 11/11 cross-block alignment of centred Δ̃_l with c_fc(l+1)
left singular subspace at +17 to +33σ above random. That test never measured
non-adjacent destinations, so it cannot distinguish "Δ̃ writes specifically to
the next block" from "Δ̃ leaks into a universal reading basis shared by every
deeper block".

This script reuses the cycle 218 calibration pipeline (same model, data loader,
hooks, centred Δ̃_l construction). For each l ∈ {0..10} we compute the top-20
right singular vectors of centred Δ̃_l and measure the top-5 (and top-20)
principal cosine sums against c_fc(j) for ALL j > l (not just j=l+1). We then
compare each row's max-at-j=l+1 against the row's mean over j > l+1
(non-adjacent destinations).

Verdict (row-aggregated):
  - CONFIRMED  if median_specificity ≥ 1.0 AND ≥7/10 rows have ALL non-adjacent
                entries within mean+3σ of random baseline
  - REFUTED    if median_specificity < 0.3
  - INCONCLUSIVE otherwise
"""
import json
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
OUTPUT_JSON = "autoresearch/results/cycle_227_delta_specificity.json"
OUTPUT_MD = "autoresearch/results/cycle_227_summary.md"

# Cycle 218 baseline (mean of c_fc top5 cos_sum across the 11 adjacent pairs)
CYCLE_218_TOP5_MEAN_ADJACENT = 2.9687

torch.manual_seed(SEED)
print(f"Device: {DEVICE}")

# ── Load model + tokenizer ───────────────────────────────────────────────────
print("Loading GPT-2 and tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ── Calibration data (matches cycle 218 loader exactly) ──────────────────────
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

# ── Hook: capture c_fc input per block (only — drop block_in pre-hook) ───────
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

print("Running forward pass...")
with torch.no_grad():
    for ids in calib_ids:
        _ = model(ids)
for h in hooks:
    h.remove()

mlp_inputs = {}
for l in range(N_BLOCKS):
    mlp_inputs[l] = torch.cat(cfc_stash[l], dim=0).reshape(-1, 768)
    assert mlp_inputs[l].shape == (N_CALIB * SEQ_LEN, 768)

cfc_stash = None
print(f"Captured c_fc inputs: {mlp_inputs[0].shape}")

# ── Centred Δ̃_l → top-20 right SVs (reuse cycle 218 path exactly) ───────────
vh_top20 = {}  # l -> (20, 768)
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
    delta_out = torch.cat(parts, dim=0)
    parts = None

    mean_delta = delta_out.mean(0)
    delta_centred = delta_out - mean_delta
    delta_out = None

    U, S, Vh = torch.linalg.svd(delta_centred, full_matrices=False)
    vh_top20[l] = Vh[:20].clone()  # (20, 768)
    print(f"  σ₀={float(S[0]):.2f}, σ₁={float(S[1]):.2f}, ||mean_Δ||={float(mean_delta.norm()):.3f}")

mlp_inputs = None

# ── c_fc(j) reader bases for ALL j ∈ {1..11} ─────────────────────────────────
def top_left_svs(W, k=20):
    """Top-k left singular vectors of W as columns. Returns (d, k)."""
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    return U[:, :k].contiguous()

print("\n── Building c_fc reader bases for j=1..11 ──")
U_fc = {}  # j -> (768, 20)
for j in range(1, N_BLOCKS):
    block = model.transformer.h[j]
    W_fc = block.mlp.c_fc.weight.detach().float().cpu()
    assert W_fc.shape == (768, 3072), f"block {j} W_fc shape {W_fc.shape}"
    U_fc[j] = top_left_svs(W_fc, 20)
    assert U_fc[j].shape == (768, 20)

# ── Random baseline: 10 seeds × Gaussian QR-orthonormal (20, 768) ────────────
print("\n── Random baseline (10 seeds) ──")

def random_orthonormal(seed, k=20, d=768):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(d, k, generator=g)
    Q, _ = torch.linalg.qr(G)
    return Q.T.contiguous()  # (20, 768) orthonormal rows

random_V20 = [random_orthonormal(42 + s * 1000 + 7) for s in range(N_RANDOM_SEEDS)]

# Per-j baseline (top5 + top20 cos_sum)
random_baseline_per_j = []
random_top5_mean = {}
random_top5_std = {}
random_top20_mean = {}
random_top20_std = {}
for j in range(1, N_BLOCKS):
    Bj = U_fc[j]  # (768, 20)
    top5_acc = []
    top20_acc = []
    for Vr in random_V20:
        M = Vr @ Bj  # (20, 20)
        cos = torch.linalg.svdvals(M)
        top5_acc.append(cos[:5].sum().item())
        top20_acc.append(cos.sum().item())
    t5 = torch.tensor(top5_acc)
    t20 = torch.tensor(top20_acc)
    random_top5_mean[j] = float(t5.mean().item())
    random_top5_std[j] = float(t5.std().item())
    random_top20_mean[j] = float(t20.mean().item())
    random_top20_std[j] = float(t20.std().item())
    random_baseline_per_j.append({
        "j": j,
        "top5_mean": random_top5_mean[j],
        "top5_std": random_top5_std[j],
        "top20_mean": random_top20_mean[j],
        "top20_std": random_top20_std[j],
    })
    print(f"  j={j}: top5={random_top5_mean[j]:.4f}±{random_top5_std[j]:.4f} "
          f"top20={random_top20_mean[j]:.4f}±{random_top20_std[j]:.4f}")

# ── Heatmap: cos_sum_top5[l, j] and cos_sum_top20[l, j] for ALL l < j ────────
print("\n── Cross-block heatmap ──")
heatmap_top5 = [[None] * N_BLOCKS for _ in range(N_BLOCKS - 1)]   # rows l=0..10, cols j=0..11
heatmap_top20 = [[None] * N_BLOCKS for _ in range(N_BLOCKS - 1)]
# Note: store as 11×12 grid (l=0..10, j=0..11). Entries with j ≤ l stay None.

for l in range(N_BLOCKS - 1):
    V20 = vh_top20[l]  # (20, 768)
    row_str = []
    for j in range(l + 1, N_BLOCKS):
        Bj = U_fc[j]
        M = V20 @ Bj  # (20, 20)
        cos = torch.linalg.svdvals(M)
        cs5 = float(cos[:5].sum().item())
        cs20 = float(cos.sum().item())
        heatmap_top5[l][j] = cs5
        heatmap_top20[l][j] = cs20
        row_str.append(f"j={j}:{cs5:.2f}")
    if l == 0 or l == 10:
        print(f"  l={l}: " + " ".join(row_str))

# ── Per-row analysis ─────────────────────────────────────────────────────────
print("\n── Per-row specificity (top5) ──")
row_analysis = []
specificity_values = []  # only rows with ≥1 non-adjacent entry (l ∈ 0..9)
for l in range(N_BLOCKS - 1):  # l=0..10
    max_at_lp1 = heatmap_top5[l][l + 1]
    nonadj_vals = [heatmap_top5[l][j] for j in range(l + 2, N_BLOCKS)]
    if len(nonadj_vals) == 0:
        # l=10: no non-adjacent
        mean_nonadj = None
        spec = None
        all_quiet = None  # excluded from quiet count
    else:
        mean_nonadj = float(sum(nonadj_vals) / len(nonadj_vals))
        spec = float(max_at_lp1 - mean_nonadj)
        # Quiet check: ALL non-adjacent entries within mean+3σ of random
        all_quiet = all(
            heatmap_top5[l][j] <= random_top5_mean[j] + 3 * random_top5_std[j]
            for j in range(l + 2, N_BLOCKS)
        )
        specificity_values.append(spec)
    row_analysis.append({
        "l": l,
        "max_at_lp1": float(max_at_lp1),
        "mean_nonadj": mean_nonadj,
        "specificity": spec,
        "all_quiet_nonadj": all_quiet,
    })
    if mean_nonadj is None:
        print(f"  l={l}: max@l+1={max_at_lp1:.3f}  (no non-adjacent)")
    else:
        print(f"  l={l}: max@l+1={max_at_lp1:.3f}  mean_nonadj={mean_nonadj:.3f}  "
              f"spec={spec:+.3f}  quiet={all_quiet}")

# Median specificity
sorted_spec = sorted(specificity_values)
n_spec = len(sorted_spec)
if n_spec % 2 == 1:
    median_specificity = sorted_spec[n_spec // 2]
else:
    median_specificity = 0.5 * (sorted_spec[n_spec // 2 - 1] + sorted_spec[n_spec // 2])
median_specificity = float(median_specificity)

# Quiet rows count (l=0..9 only — l=10 has no non-adjacent)
n_rows_with_quiet_nonadj = sum(
    1 for r in row_analysis if r["all_quiet_nonadj"] is True
)

# Significance count of non-adjacent (l, j) entries above 3σ
n_significant_nonadj = 0
for l in range(N_BLOCKS - 1):
    for j in range(l + 2, N_BLOCKS):
        if heatmap_top5[l][j] > random_top5_mean[j] + 3 * random_top5_std[j]:
            n_significant_nonadj += 1

# ── Verdict ──────────────────────────────────────────────────────────────────
if median_specificity >= 1.0 and n_rows_with_quiet_nonadj >= 7:
    verdict = "confirmed"
elif median_specificity < 0.3:
    verdict = "refuted"
else:
    verdict = "inconclusive"

print(f"\nmedian_specificity = {median_specificity:.4f}")
print(f"n_rows_with_quiet_nonadj = {n_rows_with_quiet_nonadj} / 10")
print(f"n_significant_nonadj = {n_significant_nonadj} (out of {sum(N_BLOCKS - 1 - l - 1 for l in range(N_BLOCKS - 1))} entries)")
print(f"VERDICT: {verdict.upper()}")

# Sanity check vs cycle 218 baseline (mean over l of cos_sum_top5[l][l+1])
adj_top5 = [heatmap_top5[l][l + 1] for l in range(N_BLOCKS - 1)]
adj_mean_top5 = sum(adj_top5) / len(adj_top5)
print(f"\nSanity (adjacent top5 mean): cycle227={adj_mean_top5:.4f} cycle218={CYCLE_218_TOP5_MEAN_ADJACENT:.4f} "
      f"Δ={adj_mean_top5 - CYCLE_218_TOP5_MEAN_ADJACENT:+.4f}")
sanity_ok = abs(adj_mean_top5 - CYCLE_218_TOP5_MEAN_ADJACENT) < 0.05 * CYCLE_218_TOP5_MEAN_ADJACENT
if not sanity_ok:
    print("  WARNING: cycle 218 sanity deviation > 5%")

# ── Save JSON ────────────────────────────────────────────────────────────────
result = {
    "cycle": 227,
    "item": 237,
    "verdict": verdict,
    "config": {
        "n_sequences": N_CALIB,
        "seq_len": SEQ_LEN,
        "n_tokens": N_CALIB * SEQ_LEN,
        "n_random_seeds": N_RANDOM_SEEDS,
        "model": "gpt2",
        "metric": "principal_cosine_sum_top5_and_top20",
    },
    "heatmap_top5": heatmap_top5,
    "heatmap_top20": heatmap_top20,
    "random_baseline_per_j": random_baseline_per_j,
    "row_analysis": row_analysis,
    "median_specificity": median_specificity,
    "n_rows_with_quiet_nonadj": n_rows_with_quiet_nonadj,
    "n_significant_nonadj": n_significant_nonadj,
    "cycle_218_baseline_top5_mean": CYCLE_218_TOP5_MEAN_ADJACENT,
    "cycle_227_adjacent_top5_mean": float(adj_mean_top5),
    "sanity_check_ok": bool(sanity_ok),
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved JSON → {OUTPUT_JSON}")

# ── Markdown summary ─────────────────────────────────────────────────────────
def fmt_cell(v):
    return "  ·  " if v is None else f"{v:5.2f}"

lines = []
lines.append(f"# Cycle 227 — Item 237: Δ̃_l specificity control vs non-adjacent c_fc(j)")
lines.append("")
lines.append(f"**Verdict:** {verdict.upper()}")
lines.append("")
lines.append("## Headline")
lines.append(f"- median_specificity (max@l+1 − mean_nonadj, top-5 cos_sum): **{median_specificity:+.3f}** "
             f"(CONFIRMED ≥ 1.0, REFUTED < 0.3)")
lines.append(f"- rows where all non-adjacent (l, j) sit within mean+3σ of random: "
             f"**{n_rows_with_quiet_nonadj} / 10** (CONFIRMED ≥ 7)")
lines.append(f"- non-adjacent entries above 3σ random: **{n_significant_nonadj} / 55**")
lines.append(f"- sanity check (cycle 227 adjacent top5 mean = {adj_mean_top5:.4f} vs cycle 218 {CYCLE_218_TOP5_MEAN_ADJACENT:.4f}): "
             f"{'OK' if sanity_ok else 'FAIL'}")
lines.append("")

lines.append("## Heatmap — top-5 principal cosine sum (rows l, cols j)")
lines.append("")
header = "| l\\j | " + " | ".join(f"{j}" for j in range(1, N_BLOCKS)) + " |"
sep = "|---:|" + "------:|" * (N_BLOCKS - 1)
lines.append(header)
lines.append(sep)
for l in range(N_BLOCKS - 1):
    row_cells = []
    for j in range(1, N_BLOCKS):
        v = heatmap_top5[l][j]
        row_cells.append("·" if v is None else f"{v:.2f}")
    lines.append(f"| {l} | " + " | ".join(row_cells) + " |")
lines.append("")

lines.append("## Heatmap — top-20 principal cosine sum")
lines.append("")
lines.append(header)
lines.append(sep)
for l in range(N_BLOCKS - 1):
    row_cells = []
    for j in range(1, N_BLOCKS):
        v = heatmap_top20[l][j]
        row_cells.append("·" if v is None else f"{v:.2f}")
    lines.append(f"| {l} | " + " | ".join(row_cells) + " |")
lines.append("")

lines.append("## Per-row specificity (top-5 cos_sum)")
lines.append("")
lines.append("| l | max @ l+1 | mean over j>l+1 | specificity | all non-adj quiet? |")
lines.append("|---:|---------:|----------------:|------------:|:-----------------:|")
for r in row_analysis:
    mean_str = "—" if r["mean_nonadj"] is None else f"{r['mean_nonadj']:.3f}"
    spec_str = "—" if r["specificity"] is None else f"{r['specificity']:+.3f}"
    quiet_str = "—" if r["all_quiet_nonadj"] is None else ("✓" if r["all_quiet_nonadj"] else "·")
    lines.append(f"| {r['l']} | {r['max_at_lp1']:.3f} | {mean_str} | {spec_str} | {quiet_str} |")
lines.append("")
lines.append(f"Median specificity (rows l=0..9): **{median_specificity:+.3f}**")
lines.append("")

lines.append("## Random baseline — per j (10 seeds Gaussian QR-orthonormal 20×768)")
lines.append("")
lines.append("| j | top5 mean | top5 std | top5 mean+3σ | top20 mean | top20 std |")
lines.append("|---:|---------:|--------:|------------:|----------:|---------:|")
for rb in random_baseline_per_j:
    j = rb["j"]
    thresh5 = rb["top5_mean"] + 3 * rb["top5_std"]
    lines.append(f"| {j} | {rb['top5_mean']:.4f} | {rb['top5_std']:.4f} | {thresh5:.4f} | "
                 f"{rb['top20_mean']:.4f} | {rb['top20_std']:.4f} |")
lines.append("")

lines.append("## Plain-language summary")
if verdict == "confirmed":
    lines.append(
        f"The cycle 218 alignment IS adjacency-specific. The cos_sum at j=l+1 sits {median_specificity:.2f} "
        f"above the average non-adjacent destination (top-5, median across rows), and "
        f"{n_rows_with_quiet_nonadj}/10 rows show ALL non-adjacent (l, j) entries within 3σ of the random baseline. "
        f"Centred Δ̃_l preferentially writes into block l+1's c_fc reading subspace and not into any later block's, "
        f"ruling out the 'universal reading basis' alternative."
    )
elif verdict == "refuted":
    lines.append(
        f"The cycle 218 alignment is NOT adjacency-specific. The cos_sum gap between j=l+1 and the average "
        f"non-adjacent destination is only {median_specificity:.2f} (top-5, median), well below the 0.3 threshold. "
        f"Δ̃_l writes into a basis that all later blocks happen to read — a generic 'universal c_fc reading subspace' "
        f"interpretation, not an explicit l→l+1 communication channel. Cycle 218's CONFIRMED finding survives only "
        f"as a statement about alignment with c_fc subspaces in general, not as evidence of inter-block targeting."
    )
else:
    lines.append(
        f"Mixed evidence. Median specificity = {median_specificity:.2f} (in the 0.3–1.0 inconclusive band), "
        f"and {n_rows_with_quiet_nonadj}/10 rows have all non-adjacent entries within 3σ of random. The peak at "
        f"j=l+1 IS larger than the average non-adjacent value, but the gap is small enough that the 'generic reading basis' "
        f"interpretation cannot be ruled out from this measurement alone. A direct functional test (e.g. swap c_fc(l+1) "
        f"vs c_fc(j>l+1) and measure block l write→read effect) is needed to disambiguate."
    )
lines.append("")

lines.append("## Tie back to cycle 218")
lines.append(
    f"Cycle 218 reported 11/11 cross-block pairs at +17–33σ above random for the c_fc(l+1) reader. "
    f"This cycle reproduces the adjacent-pair signal (cycle 227 mean cos_sum_top5 over (l, l+1) = "
    f"{adj_mean_top5:.4f}, cycle 218 = {CYCLE_218_TOP5_MEAN_ADJACENT:.4f}, "
    f"Δ = {adj_mean_top5 - CYCLE_218_TOP5_MEAN_ADJACENT:+.4f}) and asks the missing question: "
    f"does that alignment also appear at j > l+1? The {verdict.upper()} verdict above answers it."
)
lines.append("")

with open(OUTPUT_MD, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved MD   → {OUTPUT_MD}")
print("Done.")
