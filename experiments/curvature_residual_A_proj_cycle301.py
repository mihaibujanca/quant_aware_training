"""
Cycle 301 — Item 311: Curvature residual decomposition vs cycle 280's rank-2
column space.

Geometric object: for activation pair (sigma, sigma_lin) = (GELU, ReLU) and
each block l in {0, 1, 2, 11} of GPT-2 small,
    z(x)  = (LN2(x) @ W_fc + b_fc)        in R^3072  (per token)
    delta = GELU(z) - ReLU(z)             in R^3072
    Dy    = delta @ W_proj                in R^768   (residual stream)
where W_proj = c_proj.weight has HF Conv1D shape (3072, 768).

Cycle 280's "kept 2D output subspace" at block 1 is span(Vh[:2, :]) where
W_proj = U @ diag(S) @ Vh, with Vh[:2, :] in R^(2, 768) orthonormal rows.
This script projects Dy(x) onto Vh[:2, :] and reports
    E_kept = ||Dy @ Vh[:2,:]^T||_F^2 / ||Dy||_F^2,
the energy fraction of the curvature residual that survives the rank-2
truncation. Random-2D-subspace null is reported for context.

No training. ~10 minute run cap.
"""
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = [0, 1, 2, 11]
RANK = 2
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
N_RANDOM = 100
NULL_SEED = 20260406
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_LINALG = "cpu"

FLOAT_BASELINE_REF = 53.895

OUTPUT_JSON = "autoresearch/results/cycle_301_decomp.json"
OUTPUT_HIST_PLOT = "autoresearch/results/cycle_301_cos_histograms.png"
SUMMARY_PATH = "autoresearch/results/cycle_301_summary.md"

torch.manual_seed(SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load model + data (identical slice to cycle 300) ──────────────────────────
print("Loading GPT-2 and wikitext-2 val...", flush=True)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
texts = [r["text"] for r in ds if len(r["text"].strip()) > 50]
all_ids = []
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
    if enc["input_ids"].shape[1] >= SEQ_LEN:
        all_ids.append(enc["input_ids"][:, :SEQ_LEN])
    if len(all_ids) >= N_CALIB + N_EVAL:
        break

calib_ids_cpu = [t for t in all_ids[:N_CALIB]]
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB:N_CALIB + N_EVAL]]
print(f"Calib: {len(calib_ids_cpu)} seqs  Eval: {len(eval_ids)} seqs", flush=True)


def eval_ppl(model, eval_ids):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for ids in eval_ids:
            out = model(ids, labels=ids)
            total_loss += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens)


t0 = time.time()
print("Loading model...", flush=True)
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

print("Float PPL on eval slice...", flush=True)
t0 = time.time()
float_ppl = eval_ppl(model, eval_ids)
print(f"  float PPL = {float_ppl:.3f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from cycle reference {FLOAT_BASELINE_REF} by > 0.5"
)


# ── Per-block: capture LN2 output (c_fc input) over calib via forward pre-hook
def capture_ln2_outputs(model, layer_idx, calib_ids_cpu):
    chunks = []
    def pre_hook(_mod, inputs):
        # inputs[0] is the input to c_fc, i.e. ln_2(x); shape (B, T, 768)
        chunks.append(inputs[0].detach().float().cpu())
    h = model.transformer.h[layer_idx].mlp.c_fc.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        for ids in calib_ids_cpu:
            _ = model(ids.to(DEVICE))
    h.remove()
    x = torch.cat([c.reshape(-1, 768) for c in chunks], dim=0)  # (N, 768)
    return x


per_block = {}

for l in BLOCKS:
    print(f"\n=== Block {l} ===", flush=True)
    blk = model.transformer.h[l].mlp
    W_fc = blk.c_fc.weight.detach().float().cpu()       # (768, 3072) HF Conv1D
    b_fc = blk.c_fc.bias.detach().float().cpu()         # (3072,)
    W_proj = blk.c_proj.weight.detach().float().cpu()   # (3072, 768) HF Conv1D
    assert W_fc.shape == (768, 3072), f"W_fc {W_fc.shape}"
    assert W_proj.shape == (3072, 768), f"W_proj {W_proj.shape}"

    t0 = time.time()
    x_l = capture_ln2_outputs(model, l, calib_ids_cpu)  # (N, 768) on cpu
    N = x_l.shape[0]
    assert N == N_CALIB * SEQ_LEN, f"{N} != {N_CALIB * SEQ_LEN}"
    print(f"  ln2 captured: {tuple(x_l.shape)}  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    preact = x_l @ W_fc + b_fc                          # (N, 3072)
    delta = F.gelu(preact, approximate="tanh") - F.relu(preact)  # (N, 3072)
    # Free preact & x_l now that delta is computed
    del preact, x_l
    Dy = delta @ W_proj                                  # (N, 768)
    del delta
    print(f"  Dy computed: {tuple(Dy.shape)}  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    U_, S_, Vh = torch.linalg.svd(W_proj, full_matrices=False)
    print(f"  SVD W_proj done in {time.time()-t0:.1f}s; "
          f"U {tuple(U_.shape)} S {tuple(S_.shape)} Vh {tuple(Vh.shape)} "
          f"top5 S={S_[:5].tolist()}", flush=True)
    Vh_r = Vh[:RANK, :].contiguous()                    # (2, 768)
    # Verify orthonormal rows
    II = Vh_r @ Vh_r.T
    off_diag = (II - torch.eye(RANK)).abs().max().item()
    assert off_diag < 1e-4, f"Vh_r rows not orthonormal: max off-diag {off_diag}"

    # Energy fractions
    Dy_norm_sq = float((Dy * Dy).sum().item())
    Dy_kept_top1 = Dy @ Vh[0:1, :].T                     # (N, 1)
    Dy_kept_top2 = Dy @ Vh_r.T                           # (N, 2)
    e_top1 = float((Dy_kept_top1 * Dy_kept_top1).sum().item()) / Dy_norm_sq
    e_top2 = float((Dy_kept_top2 * Dy_kept_top2).sum().item()) / Dy_norm_sq
    print(f"  ||Dy||_F^2 = {Dy_norm_sq:.4e}  E_top1 = {e_top1:.6f}  E_top2 = {e_top2:.6f}", flush=True)

    # Per-token absolute cosine of Dy_t with span(Vh_r)
    Dy_norms = torch.linalg.norm(Dy, dim=1)              # (N,)
    Dy_kept_norms = torch.linalg.norm(Dy_kept_top2, dim=1)
    cos_t = Dy_kept_norms / (Dy_norms + 1e-12)
    # Clip to [0, 1] for safety (numerical noise)
    cos_t = torch.clamp(cos_t, 0.0, 1.0)
    cos_q = torch.quantile(cos_t, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    cos_mean = float(cos_t.mean())
    cos_std = float(cos_t.std())
    cos_hist, cos_hist_edges = torch.histogram(cos_t, bins=50, range=(0.0, 1.0))
    print(f"  cos_t: mean={cos_mean:.4f} std={cos_std:.4f} "
          f"q05={cos_q[0]:.4f} median={cos_q[2]:.4f} q95={cos_q[4]:.4f}", flush=True)

    # Random null: 100 random 2D orthonormal subspaces in R^768
    g_null = torch.Generator().manual_seed(NULL_SEED + l)
    null_es = []
    for _ in range(N_RANDOM):
        R = torch.randn(768, RANK, generator=g_null)
        Q_null, _ = torch.linalg.qr(R, mode="reduced")    # (768, 2) orthonormal
        DyQ = Dy @ Q_null                                  # (N, 2)
        e_null = float((DyQ * DyQ).sum().item()) / Dy_norm_sq
        null_es.append(e_null)
    null_t = torch.tensor(null_es)
    null_mean = float(null_t.mean())
    null_std = float(null_t.std())
    null_min = float(null_t.min())
    null_max = float(null_t.max())
    print(f"  null E (100 rand 2D): mean={null_mean:.6f} std={null_std:.6f} "
          f"min={null_min:.6f} max={null_max:.6f}", flush=True)

    per_block[str(l)] = {
        "block": l,
        "W_proj_shape": list(W_proj.shape),
        "S_top5": S_[:5].tolist(),
        "Dy_frob_sq": Dy_norm_sq,
        "Dy_frob": math.sqrt(Dy_norm_sq),
        "E_kept_top1": e_top1,
        "E_kept_top2": e_top2,
        "Vh_r_orthonormality_max_off_diag": off_diag,
        "cos_per_token": {
            "mean": cos_mean,
            "std": cos_std,
            "q05": float(cos_q[0]),
            "q25": float(cos_q[1]),
            "median": float(cos_q[2]),
            "q75": float(cos_q[3]),
            "q95": float(cos_q[4]),
            "min": float(cos_t.min()),
            "max": float(cos_t.max()),
            "hist_bins": cos_hist.tolist(),
            "hist_edges": cos_hist_edges.tolist(),
        },
        "null_random_2d": {
            "n_draws": N_RANDOM,
            "mean": null_mean,
            "std": null_std,
            "min": null_min,
            "max": null_max,
            "isotropic_reference": 2.0 / 768.0,
            "all_E": null_es,
        },
    }

    del Dy, Dy_kept_top1, Dy_kept_top2, Dy_norms, Dy_kept_norms, cos_t

# ── Verdict (item 311 verbatim) ───────────────────────────────────────────────
b1 = per_block["1"]
e1 = b1["E_kept_top2"]
if e1 <= 0.05:
    verdict = "CONFIRMED"
    verdict_reason = (
        f"E_kept(block 1) = {e1:.4f} <= 0.05 -- (GELU, ReLU) curvature residual "
        f"lives almost entirely in the 766-dim subspace cycle 280 discards."
    )
elif e1 >= 0.30:
    verdict = "REFUTED"
    verdict_reason = (
        f"E_kept(block 1) = {e1:.4f} >= 0.30 -- curvature residual concentrates "
        f"inside the rank-2 kept subspace; LASER preserves rather than denoises it."
    )
else:
    verdict = "INCONCLUSIVE"
    verdict_reason = (
        f"E_kept(block 1) = {e1:.4f} is in (0.05, 0.30); neither CONFIRMED nor REFUTED."
    )

print(f"\n=== VERDICT ===\n{verdict}\n{verdict_reason}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
out = {
    "cycle": 301,
    "item": 311,
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "config": {
        "BLOCKS": BLOCKS,
        "RANK": RANK,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "SEED": SEED,
        "N_RANDOM": N_RANDOM,
        "NULL_SEED": NULL_SEED,
        "DEVICE": DEVICE,
    },
    "float_ppl": float_ppl,
    "per_block": per_block,
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {OUTPUT_JSON}", flush=True)


# ── Plot 4-panel histogram of per-token |cos(Dy_t, col(A_2))| ─────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, l in zip(axes, BLOCKS):
    pb = per_block[str(l)]
    edges = torch.tensor(pb["cos_per_token"]["hist_edges"])
    bins = torch.tensor(pb["cos_per_token"]["hist_bins"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(
        centers.tolist(),
        bins.tolist(),
        width=(edges[1] - edges[0]).item(),
        align="center",
        edgecolor="none",
        color="tab:blue",
    )
    ax.set_title(
        f"block {l}\nE_top2={pb['E_kept_top2']:.4f}\n"
        f"null mean={pb['null_random_2d']['mean']:.4f}"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$|\cos(\Delta y_t,\, \mathrm{col}(V_2))|$")
axes[0].set_ylabel("token count")
fig.suptitle(
    f"Cycle 301 — Curvature residual cosine vs cycle 280 rank-2 col(A) "
    f"per block ({verdict})"
)
fig.tight_layout()
fig.savefig(OUTPUT_HIST_PLOT, dpi=120)
print(f"Wrote {OUTPUT_HIST_PLOT}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append("# Cycle 301 — Item 311: Curvature residual decomposition vs cycle 280 col(A)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(verdict_reason)
lines.append("")
lines.append("## Setup")
lines.append(
    f"- Calib: {N_CALIB} wikitext-2 val seqs × {SEQ_LEN} tokens (seed {SEED}); "
    f"N_TOKENS={N_CALIB * SEQ_LEN}."
)
lines.append(
    f"- Float baseline PPL on {N_EVAL}-seq eval slice = {float_ppl:.4f} "
    f"(ref {FLOAT_BASELINE_REF})."
)
lines.append("- Blocks measured: " + ", ".join(map(str, BLOCKS)) + ".")
lines.append(
    "- For each block: capture LN2(x) via forward pre-hook on c_fc; "
    "compute z = LN2(x) @ W_fc + b_fc; delta = GELU(z) - ReLU(z); "
    "Dy = delta @ W_proj  (W_proj is HF Conv1D (3072, 768))."
)
lines.append(
    "- SVD W_proj = U diag(S) Vh; kept-2D output subspace = span(Vh[:2, :]) ⊂ R^768."
)
lines.append(
    "- E_kept = ||Dy @ Vh[:2,:]^T||_F^2 / ||Dy||_F^2."
)
lines.append(
    "- Random null: 100 random 2D orthonormal subspaces of R^768 (per-block "
    f"seed {NULL_SEED}+l). Trainable params: 0."
)
lines.append("")
lines.append("## Per-block results")
lines.append("")
lines.append("| block | ||Dy||_F | E_top1 | **E_top2** | null mean | null std | cos median | cos q95 |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
for l in BLOCKS:
    pb = per_block[str(l)]
    cp = pb["cos_per_token"]
    nl = pb["null_random_2d"]
    lines.append(
        f"| {l} | {pb['Dy_frob']:.3e} | {pb['E_kept_top1']:.4f} | **{pb['E_kept_top2']:.4f}** | "
        f"{nl['mean']:.4f} | {nl['std']:.5f} | {cp['median']:.4f} | {cp['q95']:.4f} |"
    )
lines.append("")
lines.append("Isotropic reference (random 2D in R^768): 2/768 = 0.00260.")
lines.append("")
lines.append("## Verdict logic (item 311 verbatim)")
lines.append("- CONFIRMED iff E_kept(block 1) ≤ 0.05.")
lines.append("- REFUTED iff E_kept(block 1) ≥ 0.30.")
lines.append("- INCONCLUSIVE otherwise.")
lines.append("")
lines.append(f"E_kept(block 1) = **{e1:.4f}** → **{verdict}**.")
lines.append("")
lines.append("## Artifacts")
for p in [
    "experiments/curvature_residual_A_proj_cycle301.py",
    OUTPUT_JSON,
    OUTPUT_HIST_PLOT,
    SUMMARY_PATH,
]:
    lines.append(f"- {p}")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
print("\nDone.", flush=True)
