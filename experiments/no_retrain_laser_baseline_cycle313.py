"""
Cycle 313 — Item 323: Pure no-retrain LASER baseline reproduction across all 12 blocks.

For block l in {0..11} and rank r in {1, 2, 4, 8, 16, 32, 64}:
  (i)  Uncentered: SVD(c_proj_l.weight); W_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :].
  (ii) Calibrated (SVD-LLM weighted SVD): Σ_l = E[x x^T] from 200 calib seqs;
       A_l = Σ_l^{1/2} (symmetric matrix square root via eigh);
       SVD(A_l @ W) = U_w S_w Vh_w; W_r = pinv(A_l) @ (U_w[:, :r] @ diag(S_w[:r]) @ Vh_w[:r, :]).

Verdict (item 323 verbatim):
  CONFIRMED iff |{l : (calibrated, r=2) PPL ≤ retrain_anchor_l - 2}| ≥ 3.
  REFUTED   iff |calibrated_PPL - uncentered_PPL| < 0.5 for ALL (l, r).
  INCONCLUSIVE otherwise.

84 cells × 2 variants = 168 PPL evals.
"""
import json
import math
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS = list(range(12))
RANKS = [1, 2, 4, 8, 16, 32, 64]
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

OUTPUT_PATH = "autoresearch/results/cycle_313_no_retrain_laser.json"
SUMMARY_PATH = "autoresearch/results/cycle_313_summary.md"

FLOAT_BASELINE_REF = 53.895  # cycle 270/280/284/305/310 reference

# Cycle 280 published init PPLs for cross-check (±0.1)
CYCLE_280_B1_R2_INIT = 57.664
CYCLE_270_B11_R22_INIT = 59.683

# Trained Adam best per (block, rank) — used for the CONFIRMED comparison.
# Anchors are the best published trained-Adam PPL at the closest rank used in cycle 280-class
# experiments. For r=2 specifically (the verdict rank): cycle 280 b1=49.03, cycle 285 b2=49.33,
# cycle 288 b3 r=24=50.59 (closest published, no r=2), cycle 289 b10 r=5=51.47, cycle 270 b11
# r=22=54.4687. Other blocks: no published r=2 retrain anchor → use float as a soft anchor.
RETRAIN_ANCHOR_R2 = {
    0: FLOAT_BASELINE_REF,  # no published anchor; use float as soft anchor
    1: 49.03,
    2: 49.33,
    3: 50.59,
    4: FLOAT_BASELINE_REF,
    5: FLOAT_BASELINE_REF,
    6: FLOAT_BASELINE_REF,
    7: FLOAT_BASELINE_REF,
    8: FLOAT_BASELINE_REF,
    9: FLOAT_BASELINE_REF,
    10: 51.47,
    11: 54.4687,
}

torch.manual_seed(SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load model + data ─────────────────────────────────────────────────────────
print("Loading GPT-2 and data...", flush=True)
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
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
print(f"Calibration: {len(calib_ids_cpu)} seqs, Eval: {len(eval_ids)} seqs", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def capture_cproj_input(model, block_idx, calib_ids):
    """Capture inputs to model.transformer.h[block_idx].mlp.c_proj across calib seqs.

    Returns X ∈ R^{N_tokens × 3072} on CPU float32.
    """
    captured = []

    def hook(module, inputs):
        # inputs is a tuple; first element is x of shape (..., 3072)
        x = inputs[0]
        captured.append(x.detach().reshape(-1, x.shape[-1]).float().cpu())
        return None

    handle = model.transformer.h[block_idx].mlp.c_proj.register_forward_pre_hook(hook)
    model.eval()
    with torch.no_grad():
        for ids in calib_ids:
            model(ids)
    handle.remove()
    X = torch.cat(captured, dim=0)
    return X


def compute_sigma_and_sqrt(X):
    """Σ = X^T X / N; A = Σ^{1/2} via symmetric eigendecomposition.

    X: (N, 3072) cpu float32. Returns (Σ, A, A_pinv) all (3072, 3072) cpu float32.
    """
    N = X.shape[0]
    Sigma = (X.T @ X) / N  # (3072, 3072)
    # Symmetrise to remove float drift
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = torch.linalg.eigh(Sigma)  # ascending
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    sqrt_eig = torch.sqrt(eigvals_clipped)
    A = (eigvecs * sqrt_eig.unsqueeze(0)) @ eigvecs.T
    A = 0.5 * (A + A.T)
    # Pseudo-inverse via eigendecomposition: A^+ = V diag(1/sqrt(eig)) V^T (clip small)
    eps = 1e-6 * float(eigvals_clipped.max().item()) if eigvals_clipped.max() > 0 else 1e-12
    inv_sqrt = torch.where(eigvals_clipped > eps, 1.0 / sqrt_eig.clamp(min=eps**0.5), torch.zeros_like(sqrt_eig))
    A_pinv = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T
    A_pinv = 0.5 * (A_pinv + A_pinv.T)
    return Sigma, A, A_pinv


def low_rank_truncate(W_full, r):
    """Standard SVD truncation: W_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]."""
    U, S, Vh = torch.linalg.svd(W_full, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r, :], U, S, Vh


def weighted_svd_truncate(W, A, A_pinv, r):
    """SVD-LLM weighted SVD truncation.

    W: (3072, 768) original weight (cpu float32).
    A: (3072, 3072) symmetric sqrt of Σ.
    A_pinv: (3072, 3072) pseudo-inverse of A.
    Returns W_r (3072, 768).
    """
    AW = A @ W  # (3072, 768)
    U_w, S_w, Vh_w = torch.linalg.svd(AW, full_matrices=False)
    truncated_AW = (U_w[:, :r] * S_w[:r]) @ Vh_w[:r, :]
    W_r = A_pinv @ truncated_AW
    return W_r


# ── Float baseline ────────────────────────────────────────────────────────────
print("Float baseline PPL...", flush=True)
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
t0 = time.time()
float_ppl = eval_ppl(model, eval_ids)
print(f"Float PPL: {float_ppl:.3f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from cycle reference {FLOAT_BASELINE_REF} by > 0.5"
)

# Snapshot all c_proj weights
orig_weights = {}
orig_dtypes = {}
orig_devices = {}
for l in BLOCKS:
    w = model.transformer.h[l].mlp.c_proj.weight.data
    orig_weights[l] = w.detach().clone()
    orig_dtypes[l] = w.dtype
    orig_devices[l] = w.device

print(f"Snapshotted c_proj weights for {len(orig_weights)} blocks. dtype={orig_dtypes[0]}", flush=True)


# ── Uncentered variant ────────────────────────────────────────────────────────
print("\n=== Uncentered SVD variant ===", flush=True)
uncentered_results = {}  # {block: {rank: ppl}}
sanity_uncentered = {}

t_uncentered_start = time.time()
for l in BLOCKS:
    print(f"\n-- block {l} --", flush=True)
    W_full = orig_weights[l].float().cpu()
    assert W_full.shape == (3072, 768), f"block {l} weight shape {W_full.shape} != (3072, 768)"
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W_full, full_matrices=False)
    print(f"  SVD done ({time.time()-t0:.1f}s); top-3 sv = [{S[0]:.4f}, {S[1]:.4f}, {S[2]:.4f}]", flush=True)
    uncentered_results[l] = {}
    for r in RANKS:
        W_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
        W_r_dev = W_r.to(device=orig_devices[l], dtype=orig_dtypes[l])
        with torch.no_grad():
            model.transformer.h[l].mlp.c_proj.weight.data.copy_(W_r_dev)
        ppl = eval_ppl(model, eval_ids)
        uncentered_results[l][r] = float(ppl)
        print(f"    r={r:3d}: PPL={ppl:.4f}", flush=True)
    # Restore
    with torch.no_grad():
        model.transformer.h[l].mlp.c_proj.weight.data.copy_(orig_weights[l])
    ppl_restore = eval_ppl(model, eval_ids)
    sanity_uncentered[l] = {"restored_ppl": float(ppl_restore), "restore_delta": float(abs(ppl_restore - float_ppl))}
    assert abs(ppl_restore - float_ppl) < 1e-3, f"restore failed at block {l}: {ppl_restore} vs {float_ppl}"

t_uncentered = time.time() - t_uncentered_start
print(f"\nUncentered variant done in {t_uncentered/60:.1f} min", flush=True)

# Cross-check vs cycle 280 published initial PPLs
b1_r2_match = abs(uncentered_results[1][2] - CYCLE_280_B1_R2_INIT)
print(f"\nSanity check: b1 r=2 uncentered = {uncentered_results[1][2]:.4f}, "
      f"cycle 280 published init = {CYCLE_280_B1_R2_INIT}, |delta|={b1_r2_match:.4f}", flush=True)
assert b1_r2_match < 0.1, f"b1 r=2 uncentered {uncentered_results[1][2]} != cycle 280 published {CYCLE_280_B1_R2_INIT} (delta {b1_r2_match})"


# ── Calibrated variant ────────────────────────────────────────────────────────
print("\n=== Calibrated (SVD-LLM weighted SVD) variant ===", flush=True)
calibrated_results = {}
sanity_calibrated = {}

t_calibrated_start = time.time()
for l in BLOCKS:
    print(f"\n-- block {l} (calibrated) --", flush=True)
    # Capture c_proj inputs across calib set (need original weights in place; they are)
    t0 = time.time()
    X = capture_cproj_input(model, l, calib_ids_dev)
    print(f"  captured X shape={tuple(X.shape)} ({time.time()-t0:.1f}s)", flush=True)
    # Compute Σ and its sqrt
    t0 = time.time()
    Sigma, A, A_pinv = compute_sigma_and_sqrt(X)
    sigma_sym = float((Sigma - Sigma.T).abs().max().item())
    aa_err = float((A @ A - Sigma).norm().item() / Sigma.norm().item())
    pinv_err = float((A @ A_pinv @ A - A).norm().item() / A.norm().item())
    print(f"  Σ symmetric err={sigma_sym:.2e}, ||A^2 - Σ||/||Σ||={aa_err:.2e}, "
          f"||A A^+ A - A||/||A||={pinv_err:.2e} ({time.time()-t0:.1f}s)", flush=True)
    # Free X for memory
    del X

    W_full = orig_weights[l].float().cpu()
    calibrated_results[l] = {}
    for r in RANKS:
        W_r = weighted_svd_truncate(W_full, A, A_pinv, r)
        W_r_dev = W_r.to(device=orig_devices[l], dtype=orig_dtypes[l])
        with torch.no_grad():
            model.transformer.h[l].mlp.c_proj.weight.data.copy_(W_r_dev)
        ppl = eval_ppl(model, eval_ids)
        calibrated_results[l][r] = float(ppl)
        print(f"    r={r:3d}: PPL={ppl:.4f}", flush=True)
    # Restore
    with torch.no_grad():
        model.transformer.h[l].mlp.c_proj.weight.data.copy_(orig_weights[l])
    ppl_restore = eval_ppl(model, eval_ids)
    sanity_calibrated[l] = {
        "restored_ppl": float(ppl_restore),
        "restore_delta": float(abs(ppl_restore - float_ppl)),
        "sigma_sym_err": sigma_sym,
        "AA_err": aa_err,
        "pinv_err": pinv_err,
    }
    assert abs(ppl_restore - float_ppl) < 1e-3, f"restore failed at block {l}: {ppl_restore} vs {float_ppl}"
    del Sigma, A, A_pinv

t_calibrated = time.time() - t_calibrated_start
print(f"\nCalibrated variant done in {t_calibrated/60:.1f} min", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
# CONFIRMED: |{l : (calibrated, r=2) PPL_l ≤ retrain_anchor_l - 2}| ≥ 3
# REFUTED: |calibrated_PPL - uncentered_PPL| < 0.5 for ALL (l, r)
# INCONCLUSIVE otherwise

confirmed_blocks = []
for l in BLOCKS:
    cal_r2 = calibrated_results[l][2]
    anchor = RETRAIN_ANCHOR_R2[l]
    if cal_r2 <= anchor - 2.0:
        confirmed_blocks.append(l)
n_confirmed = len(confirmed_blocks)

# REFUTED check
all_match = True
max_abs_delta = 0.0
max_delta_cell = None
for l in BLOCKS:
    for r in RANKS:
        d = abs(calibrated_results[l][r] - uncentered_results[l][r])
        if d > max_abs_delta:
            max_abs_delta = d
            max_delta_cell = (l, r)
        if d >= 0.5:
            all_match = False

if n_confirmed >= 3:
    verdict = "CONFIRMED"
elif all_match:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== Verdict ===", flush=True)
print(f"  CONFIRMED count (calibrated r=2 ≤ retrain_anchor - 2): {n_confirmed} / 12  -> {confirmed_blocks}", flush=True)
print(f"  REFUTED check (all |cal - unc| < 0.5): {all_match}; max |delta| = {max_abs_delta:.4f} at (block={max_delta_cell[0]}, r={max_delta_cell[1]})", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 313,
    "item": 323,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "uncentered_ppl": {str(l): {str(r): uncentered_results[l][r] for r in RANKS} for l in BLOCKS},
    "calibrated_ppl": {str(l): {str(r): calibrated_results[l][r] for r in RANKS} for l in BLOCKS},
    "retrain_anchor_r2": {str(l): RETRAIN_ANCHOR_R2[l] for l in BLOCKS},
    "verdict_logic": {
        "n_confirmed_blocks": n_confirmed,
        "confirmed_blocks": confirmed_blocks,
        "all_cal_unc_match_within_0.5": bool(all_match),
        "max_abs_delta_cal_unc": max_abs_delta,
        "max_delta_cell": [int(max_delta_cell[0]), int(max_delta_cell[1])],
        "threshold_confirmed_n_blocks_ge": 3,
        "threshold_confirmed_delta_le": -2.0,
        "threshold_refuted_all_match_lt": 0.5,
        "verdict": verdict,
    },
    "sanity_uncentered": sanity_uncentered,
    "sanity_calibrated": sanity_calibrated,
    "config": {
        "BLOCKS": BLOCKS,
        "RANKS": RANKS,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "SEED": SEED,
    },
    "wall_seconds": {
        "uncentered": t_uncentered,
        "calibrated": t_calibrated,
        "total": t_uncentered + t_calibrated,
    },
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "cycle_280_b1_r2_init": CYCLE_280_B1_R2_INIT,
        "cycle_270_b11_r22_init": CYCLE_270_B11_R22_INIT,
        "b1_r2_uncentered_match_delta": float(b1_r2_match),
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append(f"# Cycle 313 — Item 323: Pure no-retrain LASER baseline reproduction across all 12 blocks")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Blocks: {BLOCKS}, ranks: {RANKS}, total cells: {len(BLOCKS) * len(RANKS)} × 2 variants = {2 * len(BLOCKS) * len(RANKS)} PPL evals")
lines.append(f"- Wall: uncentered {t_uncentered/60:.1f} min, calibrated {t_calibrated/60:.1f} min")
lines.append("")
lines.append("## Uncentered SVD truncation PPL[block, rank]")
lines.append("")
header = "| block | " + " | ".join(f"r={r}" for r in RANKS) + " |"
sep = "|---|" + "|".join("---" for _ in RANKS) + "|"
lines.append(header)
lines.append(sep)
for l in BLOCKS:
    row = f"| {l} | " + " | ".join(fmt(uncentered_results[l][r], 3) for r in RANKS) + " |"
    lines.append(row)
lines.append("")
lines.append("## Calibrated (SVD-LLM weighted) PPL[block, rank]")
lines.append("")
lines.append(header)
lines.append(sep)
for l in BLOCKS:
    row = f"| {l} | " + " | ".join(fmt(calibrated_results[l][r], 3) for r in RANKS) + " |"
    lines.append(row)
lines.append("")
lines.append("## Calibrated − Uncentered delta")
lines.append("")
lines.append(header)
lines.append(sep)
for l in BLOCKS:
    deltas = [calibrated_results[l][r] - uncentered_results[l][r] for r in RANKS]
    row = f"| {l} | " + " | ".join(f"{d:+.3f}" for d in deltas) + " |"
    lines.append(row)
lines.append("")
lines.append(f"max |delta| = {max_abs_delta:.4f} at (block={max_delta_cell[0]}, r={max_delta_cell[1]})")
lines.append("")
lines.append("## Verdict logic (item 323 verbatim)")
lines.append("")
lines.append(f"- CONFIRMED iff |{{l : calibrated r=2 PPL_l ≤ retrain_anchor_l − 2}}| ≥ 3")
lines.append(f"- REFUTED iff |calibrated_PPL_{{l, r}} − uncentered_PPL_{{l, r}}| < 0.5 for ALL (l, r)")
lines.append("")
lines.append("| block | calibrated_r2 | retrain_anchor_r2 | delta (cal − anchor) | confirmed? |")
lines.append("|---|---|---|---|---|")
for l in BLOCKS:
    cal = calibrated_results[l][2]
    anc = RETRAIN_ANCHOR_R2[l]
    delta = cal - anc
    confirmed = "YES" if delta <= -2.0 else "no"
    lines.append(f"| {l} | {fmt(cal, 3)} | {fmt(anc, 3)} | {delta:+.3f} | {confirmed} |")
lines.append("")
lines.append(f"- n_confirmed_blocks = {n_confirmed} / 12")
lines.append(f"- all (cal, unc) within 0.5? {all_match} (max |delta| = {max_abs_delta:.4f})")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append("## Sanity")
lines.append("")
lines.append(f"- Float PPL {fmt(float_ppl)} matches cycle ref {FLOAT_BASELINE_REF} within 0.5")
lines.append(f"- b1 r=2 uncentered = {fmt(uncentered_results[1][2], 4)} matches cycle 280 published init {CYCLE_280_B1_R2_INIT} within 0.1 (|delta|={b1_r2_match:.4f})")
lines.append(f"- All 24 in-place restores returned PPL to float within 1e-3")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
