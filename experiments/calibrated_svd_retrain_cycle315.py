"""
Cycle 315 — Item 325: Calibrated SVD-LLM init + Adam retrain at block 1.

Tests whether cycle 313's calibrated SVD-LLM init (56.70 PPL at b1 r=2) improves
the cycle 280 retrain floor (49.03 at r=2). Two variants, 3 seeds each:

  (i)  Uncentered (cycle 280 recipe):
       SVD(W) = U S Vh; A_unc = U[:, :2] * S[:2] (3072, 2) frozen;
       B_unc = Vh[:2, :] (2, 768) trainable; bias trainable from W_proj.bias.

  (ii) Calibrated SVD-LLM (cycle 313 recipe + SVD split for A/B):
       Σ = X^T X / N_tokens on 200 calib seqs; A_l = Σ^(1/2) via eigh;
       SVD(A_l @ W) = U_w S_w Vh_w; W_r = A_l_pinv @ (U_w[:, :2] @ diag(S_w[:2]) @ Vh_w[:2, :]);
       SVD(W_r) = U_r S_r Vh_r; A_cal = U_r[:, :2] * S_r[:2] frozen;
       B_cal = Vh_r[:2, :] trainable; bias trainable from W_proj.bias.

Both variants: freeze everything else, Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100,
best-state restore, SEEDS=[42, 137, 2024]. N_CALIB=200, N_EVAL=100, SEQ_LEN=128, MPS.

Verdict (item 325 verbatim):
  CONFIRMED iff delta_retrain = mean(calibrated) - mean(uncentered) <= -1 PPL.
  REFUTED   iff |delta_retrain| <= 0.3 PPL.
  INCONCLUSIVE otherwise.
"""
import json
import math
import random
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_BLOCK = 1
RETRAIN_RANK = 2
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 4000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42
OUTPUT_PATH = "autoresearch/results/cycle_315_calibrated_svd_retrain.json"
SUMMARY_PATH = "autoresearch/results/cycle_315_summary.md"

# Reference baselines (item 325 cross-check literals)
FLOAT_BASELINE_REF = 53.895         # cycles 266-314 reference
CYCLE_280_B1_R2_INIT = 57.664       # cycle 280 uncentered init
CYCLE_313_B1_R2_CAL_INIT = 56.70    # cycle 313 calibrated r=2 init (rounded)
CYCLE_280_B1_R2_RETRAIN = 49.03     # cycle 280 retrain best_mean

# Verdict thresholds (item 325 verbatim)
CONFIRMED_DELTA_LE = -1.0
REFUTED_DELTA_ABS_LE = 0.3

torch.manual_seed(LOAD_SEED)
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


class LowRankCProj(nn.Module):
    """Replacement for HF Conv1D c_proj at rank r (cycle 280/310/314 template).

    Frozen A (3072, r) as buffer; trainable B (r, 768) and bias (768) as
    nn.Parameter. Forward: x @ (A @ B) + bias.
    """

    def __init__(self, A_frozen, B_init, b_init, nf=768):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())
        self.bias = nn.Parameter(b_init.clone().contiguous())
        self.nf = nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W = self.A @ self.B
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        x = x.view(size_out)
        return x


def capture_cproj_input(model, block_idx, calib_ids):
    """Hook c_proj inputs on calib_ids; return X ∈ R^{N_tokens × 3072} cpu float32."""
    captured = []

    def hook(module, inputs):
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
    """Σ = X^T X / N; A = Σ^(1/2) via symmetric eigh. Returns (Σ, A, A_pinv)."""
    N = X.shape[0]
    Sigma = (X.T @ X) / N
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    eigvals_clipped = torch.clamp(eigvals, min=0.0)
    sqrt_eig = torch.sqrt(eigvals_clipped)
    A = (eigvecs * sqrt_eig.unsqueeze(0)) @ eigvecs.T
    A = 0.5 * (A + A.T)
    max_eig = float(eigvals_clipped.max().item()) if eigvals_clipped.max() > 0 else 1e-12
    eps = 1e-6 * max_eig
    inv_sqrt = torch.where(
        eigvals_clipped > eps, 1.0 / sqrt_eig.clamp(min=eps ** 0.5), torch.zeros_like(sqrt_eig)
    )
    A_pinv = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T
    A_pinv = 0.5 * (A_pinv + A_pinv.T)
    return Sigma, A, A_pinv


def weighted_svd_truncate(W, A, A_pinv, r):
    """SVD-LLM weighted SVD truncation (cycle 313 helper)."""
    AW = A @ W
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
print(f"Float PPL: {float_ppl:.4f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from cycle reference {FLOAT_BASELINE_REF} by > 0.5"
)

# Snapshot b1 c_proj
c_proj_orig_weight = model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.detach().clone()
c_proj_orig_bias = model.transformer.h[TARGET_BLOCK].mlp.c_proj.bias.data.detach().clone()
assert c_proj_orig_weight.shape == (3072, 768), (
    f"c_proj.weight shape {c_proj_orig_weight.shape} != (3072, 768)"
)
orig_dtype = c_proj_orig_weight.dtype
orig_device = c_proj_orig_weight.device


# ── Uncentered SVD factors (cycle 280 recipe) ─────────────────────────────────
print(f"\nUncentered SVD(W) at block {TARGET_BLOCK}...", flush=True)
W_full = c_proj_orig_weight.float().cpu()
t0 = time.time()
U, S, Vh = torch.linalg.svd(W_full, full_matrices=False)
print(f"  SVD {time.time()-t0:.1f}s; top-5 sv = {S[:5].tolist()}", flush=True)

A_unc = (U[:, :RETRAIN_RANK] * S[:RETRAIN_RANK]).contiguous()  # (3072, 2)
B_unc = Vh[:RETRAIN_RANK, :].contiguous()                       # (2, 768)
W_r_unc = (A_unc @ B_unc)
print(f"  A_unc {tuple(A_unc.shape)}, B_unc {tuple(B_unc.shape)}", flush=True)

# Sanity: uncentered init PPL matches cycle 280 published
with torch.no_grad():
    model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(
        W_r_unc.to(device=orig_device, dtype=orig_dtype)
    )
uncentered_init_ppl = eval_ppl(model, eval_ids)
unc_delta = abs(uncentered_init_ppl - CYCLE_280_B1_R2_INIT)
print(f"  Uncentered init PPL = {uncentered_init_ppl:.4f} (cycle 280 ref {CYCLE_280_B1_R2_INIT}, |delta|={unc_delta:.4f})", flush=True)
assert unc_delta < 0.01, f"uncentered init PPL {uncentered_init_ppl} != cycle 280 {CYCLE_280_B1_R2_INIT}"
# Restore
with torch.no_grad():
    model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(c_proj_orig_weight)
ppl_restore = eval_ppl(model, eval_ids)
assert abs(ppl_restore - float_ppl) < 1e-3, f"restore failed: {ppl_restore} vs {float_ppl}"


# ── Calibrated SVD-LLM factors (cycle 313 recipe + SVD split) ─────────────────
print(f"\nCalibrated SVD-LLM factors at block {TARGET_BLOCK}...", flush=True)
t0 = time.time()
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
X = capture_cproj_input(model, TARGET_BLOCK, calib_ids_dev)
print(f"  captured X {tuple(X.shape)} ({time.time()-t0:.1f}s)", flush=True)

t0 = time.time()
Sigma, A_lam, A_lam_pinv = compute_sigma_and_sqrt(X)
sigma_sym = float((Sigma - Sigma.T).abs().max().item())
aa_err = float((A_lam @ A_lam - Sigma).norm().item() / Sigma.norm().item())
pinv_err = float((A_lam @ A_lam_pinv @ A_lam - A_lam).norm().item() / A_lam.norm().item())
print(f"  Σ sym={sigma_sym:.2e}, ||A²-Σ||/||Σ||={aa_err:.2e}, pinv_err={pinv_err:.2e} ({time.time()-t0:.1f}s)", flush=True)
del X

W_r_cal = weighted_svd_truncate(W_full, A_lam, A_lam_pinv, RETRAIN_RANK)  # (3072, 768)
# SVD the rank-2 calibrated W_r to get a clean A/B split for the retrain module
U_r, S_r, Vh_r = torch.linalg.svd(W_r_cal, full_matrices=False)
# Numerical rank of W_r should be RETRAIN_RANK; use top-k columns
A_cal = (U_r[:, :RETRAIN_RANK] * S_r[:RETRAIN_RANK]).contiguous()  # (3072, 2)
B_cal = Vh_r[:RETRAIN_RANK, :].contiguous()                         # (2, 768)
print(f"  SVD(W_r_cal) top-5 sv = {S_r[:5].tolist()}  (ranks beyond {RETRAIN_RANK} should be ~0)", flush=True)
print(f"  A_cal {tuple(A_cal.shape)}, B_cal {tuple(B_cal.shape)}", flush=True)

# Reconstruction check: (A_cal @ B_cal) == W_r_cal within SVD precision
recon_err = float((A_cal @ B_cal - W_r_cal).norm().item() / W_r_cal.norm().item())
print(f"  ||A_cal·B_cal - W_r_cal||/||W_r_cal|| = {recon_err:.2e}", flush=True)
assert recon_err < 1e-5, f"SVD split reconstruction error {recon_err} too large"

# Sanity: calibrated init PPL matches cycle 313 published
with torch.no_grad():
    model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(
        W_r_cal.to(device=orig_device, dtype=orig_dtype)
    )
calibrated_init_ppl_from_Wr = eval_ppl(model, eval_ids)
with torch.no_grad():
    model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(
        (A_cal @ B_cal).to(device=orig_device, dtype=orig_dtype)
    )
calibrated_init_ppl_from_split = eval_ppl(model, eval_ids)
cal_delta = abs(calibrated_init_ppl_from_Wr - CYCLE_313_B1_R2_CAL_INIT)
print(f"  Calibrated init PPL (from W_r)     = {calibrated_init_ppl_from_Wr:.4f} (cycle 313 ref {CYCLE_313_B1_R2_CAL_INIT}, |delta|={cal_delta:.4f})", flush=True)
print(f"  Calibrated init PPL (from A_cal@B) = {calibrated_init_ppl_from_split:.4f} (should match W_r within 1e-3)", flush=True)
assert abs(calibrated_init_ppl_from_split - calibrated_init_ppl_from_Wr) < 1e-3, (
    f"SVD split init drift: {calibrated_init_ppl_from_split} vs {calibrated_init_ppl_from_Wr}"
)
assert cal_delta < 0.05, f"calibrated init PPL {calibrated_init_ppl_from_Wr} != cycle 313 ref {CYCLE_313_B1_R2_CAL_INIT}"
# Restore
with torch.no_grad():
    model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(c_proj_orig_weight)
ppl_restore = eval_ppl(model, eval_ids)
assert abs(ppl_restore - float_ppl) < 1e-3, f"restore failed: {ppl_restore} vs {float_ppl}"

del model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Retrain both variants × 3 seeds ───────────────────────────────────────────
def retrain_variant(variant_name, A_factor, B_factor):
    print(f"\n=== Retrain variant: {variant_name} ({len(SEEDS)} seeds) ===", flush=True)
    seeds_out = []
    expected_trainable = RETRAIN_RANK * 768 + 768  # 2304

    for seed in SEEDS:
        print(f"\n--- {variant_name} seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        random.seed(seed)

        model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model_s.eval()

        lowrank = LowRankCProj(
            A_frozen=A_factor.to(device=DEVICE, dtype=torch.float32),
            B_init=B_factor.to(device=DEVICE, dtype=torch.float32),
            b_init=c_proj_orig_bias.to(device=DEVICE, dtype=torch.float32),
            nf=768,
        ).to(DEVICE)
        model_s.transformer.h[TARGET_BLOCK].mlp.c_proj = lowrank

        init_ppl = eval_ppl(model_s, eval_ids)
        print(f"  init PPL ({variant_name} rank-{RETRAIN_RANK}): {init_ppl:.4f}", flush=True)

        for p in model_s.parameters():
            p.requires_grad_(False)
        lowrank.B.requires_grad_(True)
        lowrank.bias.requires_grad_(True)
        trainable = [lowrank.B, lowrank.bias]
        n_trainable = sum(p.numel() for p in trainable)
        assert n_trainable == expected_trainable, f"trainable count {n_trainable} != {expected_trainable}"

        optimizer = torch.optim.Adam(trainable, lr=LR)

        calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids)

        best_ppl = init_ppl
        best_step = 0
        best_state = {
            "B": lowrank.B.detach().clone(),
            "bias": lowrank.bias.detach().clone(),
        }
        history = [{"step": 0, "eval_ppl": float(init_ppl)}]

        step = 0
        t_train = time.time()
        while step < N_STEPS:
            for ids in calib_ids:
                if step >= N_STEPS:
                    break
                model_s.train()
                optimizer.zero_grad()
                out = model_s(ids, labels=ids)
                loss = out.loss
                loss.backward()
                optimizer.step()
                step += 1

                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model_s, eval_ids)
                    history.append({"step": step, "eval_ppl": float(ppl_e), "calib_loss": float(loss.item())})
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        best_state = {
                            "B": lowrank.B.detach().clone(),
                            "bias": lowrank.bias.detach().clone(),
                        }
                    if step % 500 == 0:
                        print(f"    step {step:4d}: eval_ppl={ppl_e:.4f}  best={best_ppl:.4f}@{best_step}", flush=True)

        final_ppl = eval_ppl(model_s, eval_ids)
        with torch.no_grad():
            lowrank.B.data.copy_(best_state["B"])
            lowrank.bias.data.copy_(best_state["bias"])
        restored_ppl = eval_ppl(model_s, eval_ids)
        assert abs(restored_ppl - best_ppl) < 1e-3, (
            f"{variant_name} seed {seed}: restored {restored_ppl} != best {best_ppl}"
        )
        dt = time.time() - t_train
        print(f"  {variant_name} seed {seed}: init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} final@{N_STEPS}={final_ppl:.4f} restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

        seeds_out.append({
            "seed": seed,
            "init_ppl": float(init_ppl),
            "best_ppl": float(best_ppl),
            "best_step": int(best_step),
            "final_ppl_at_n_steps": float(final_ppl),
            "restored_best_ppl": float(restored_ppl),
            "history": history,
        })

        del model_s, lowrank, optimizer
        if DEVICE == "mps":
            torch.mps.empty_cache()

    best_ppls = [s["best_ppl"] for s in seeds_out]
    best_mean = float(sum(best_ppls) / len(best_ppls))
    best_var = float(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
    best_std = math.sqrt(best_var)
    print(f"\n{variant_name} best_mean = {best_mean:.4f} ± {best_std:.4f}  (n={len(SEEDS)})", flush=True)
    return {
        "seeds": seeds_out,
        "best_mean": best_mean,
        "best_std": best_std,
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
    }


t_retrain_start = time.time()
uncentered = retrain_variant("uncentered", A_unc, B_unc)
calibrated = retrain_variant("calibrated", A_cal, B_cal)
t_retrain = time.time() - t_retrain_start
print(f"\nTotal retrain wall {t_retrain/60:.1f} min", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
delta_retrain = calibrated["best_mean"] - uncentered["best_mean"]
if delta_retrain <= CONFIRMED_DELTA_LE:
    verdict = "CONFIRMED"
elif abs(delta_retrain) <= REFUTED_DELTA_ABS_LE:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

# Cross-check vs cycle 280 anchor
unc_vs_cycle280_delta = uncentered["best_mean"] - CYCLE_280_B1_R2_RETRAIN

print(f"\n=== Verdict ===", flush=True)
print(f"  uncentered best_mean = {uncentered['best_mean']:.4f} ± {uncentered['best_std']:.4f}", flush=True)
print(f"  calibrated best_mean = {calibrated['best_mean']:.4f} ± {calibrated['best_std']:.4f}", flush=True)
print(f"  delta_retrain (cal - unc) = {delta_retrain:+.4f}", flush=True)
print(f"  CONFIRMED if ≤ {CONFIRMED_DELTA_LE}, REFUTED if |Δ| ≤ {REFUTED_DELTA_ABS_LE}", flush=True)
print(f"  uncentered vs cycle 280 anchor ({CYCLE_280_B1_R2_RETRAIN}): {unc_vs_cycle280_delta:+.4f}", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 315,
    "item": 325,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "delta_retrain": float(delta_retrain),
    "uncentered_init_ppl": float(uncentered_init_ppl),
    "calibrated_init_ppl": float(calibrated_init_ppl_from_Wr),
    "calibrated_init_ppl_from_split": float(calibrated_init_ppl_from_split),
    "uncentered": uncentered,
    "calibrated": calibrated,
    "verdict_logic": {
        "delta_retrain": float(delta_retrain),
        "threshold_confirmed_delta_le": CONFIRMED_DELTA_LE,
        "threshold_refuted_abs_delta_le": REFUTED_DELTA_ABS_LE,
        "verdict": verdict,
    },
    "cross_checks": {
        "float_ppl": float(float_ppl),
        "float_ref": FLOAT_BASELINE_REF,
        "float_delta": float(abs(float_ppl - FLOAT_BASELINE_REF)),
        "uncentered_init_vs_cycle280": {
            "measured": float(uncentered_init_ppl),
            "ref": CYCLE_280_B1_R2_INIT,
            "delta": float(unc_delta),
            "tolerance": 0.01,
        },
        "calibrated_init_vs_cycle313": {
            "measured": float(calibrated_init_ppl_from_Wr),
            "ref": CYCLE_313_B1_R2_CAL_INIT,
            "delta": float(cal_delta),
            "tolerance": 0.05,
        },
        "calibrated_split_reconstruction_rel_err": recon_err,
        "sigma_sym_err": sigma_sym,
        "AA_minus_Sigma_rel_err": aa_err,
        "pinv_rel_err": pinv_err,
        "uncentered_retrain_vs_cycle280": {
            "measured": float(uncentered["best_mean"]),
            "ref": CYCLE_280_B1_R2_RETRAIN,
            "delta": float(unc_vs_cycle280_delta),
            "tolerance_ref": 0.1,
        },
    },
    "config": {
        "TARGET_BLOCK": TARGET_BLOCK,
        "RETRAIN_RANK": RETRAIN_RANK,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS": N_STEPS,
        "LR": LR,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
        "trainable_params_per_fit": RETRAIN_RANK * 768 + 768,
    },
    "wall_seconds": {
        "retrain_total": t_retrain,
    },
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "cycle_280_b1_r2_init": CYCLE_280_B1_R2_INIT,
        "cycle_280_b1_r2_retrain": CYCLE_280_B1_R2_RETRAIN,
        "cycle_313_b1_r2_cal_init": CYCLE_313_B1_R2_CAL_INIT,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append(f"# Cycle 315 — Item 325: Calibrated SVD-LLM init + Adam retrain at block 1")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Target: block {TARGET_BLOCK} c_proj, rank {RETRAIN_RANK}, HF Conv1D (3072, 768)")
lines.append(f"- Trainable per fit: 2,304 = {RETRAIN_RANK}×768 (B) + 768 (bias); 2 variants × 3 seeds = 6 fits")
lines.append(f"- Wall: {t_retrain/60:.1f} min retrain")
lines.append("")
lines.append("## Init PPLs (no retrain)")
lines.append("")
lines.append(f"- Uncentered: {fmt(uncentered_init_ppl)}  (cycle 280 ref {CYCLE_280_B1_R2_INIT}, |delta|={unc_delta:.4f})")
lines.append(f"- Calibrated: {fmt(calibrated_init_ppl_from_Wr)}  (cycle 313 ref {CYCLE_313_B1_R2_CAL_INIT}, |delta|={cal_delta:.4f})")
lines.append(f"- Init delta (cal − unc): {calibrated_init_ppl_from_Wr - uncentered_init_ppl:+.4f}")
lines.append("")
lines.append(f"## Retrain (Adam LR={LR}, {N_STEPS} steps, 3 seeds, best-state restore)")
lines.append("")
lines.append("### Uncentered (cycle 280 recipe)")
lines.append("")
lines.append("| seed | init | best | best_step | final@N | restored |")
lines.append("|---|---|---|---|---|---|")
for s in uncentered["seeds"]:
    lines.append(f"| {s['seed']} | {fmt(s['init_ppl'])} | {fmt(s['best_ppl'])} | {s['best_step']} | {fmt(s['final_ppl_at_n_steps'])} | {fmt(s['restored_best_ppl'])} |")
lines.append(f"| **mean** | — | **{fmt(uncentered['best_mean'])}** | — | — | — |")
lines.append(f"| **std**  | — | {fmt(uncentered['best_std'])} | — | — | — |")
lines.append("")
lines.append("### Calibrated (SVD-LLM)")
lines.append("")
lines.append("| seed | init | best | best_step | final@N | restored |")
lines.append("|---|---|---|---|---|---|")
for s in calibrated["seeds"]:
    lines.append(f"| {s['seed']} | {fmt(s['init_ppl'])} | {fmt(s['best_ppl'])} | {s['best_step']} | {fmt(s['final_ppl_at_n_steps'])} | {fmt(s['restored_best_ppl'])} |")
lines.append(f"| **mean** | — | **{fmt(calibrated['best_mean'])}** | — | — | — |")
lines.append(f"| **std**  | — | {fmt(calibrated['best_std'])} | — | — | — |")
lines.append("")
lines.append("## Verdict logic (item 325 verbatim)")
lines.append("")
lines.append(f"- delta_retrain = mean(calibrated) − mean(uncentered) = **{delta_retrain:+.4f} PPL**")
lines.append(f"- CONFIRMED iff ≤ {CONFIRMED_DELTA_LE}")
lines.append(f"- REFUTED iff |delta| ≤ {REFUTED_DELTA_ABS_LE}")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append("## Comparisons")
lines.append("")
lines.append(f"- Uncentered best_mean vs cycle 280 retrain anchor ({CYCLE_280_B1_R2_RETRAIN}): {unc_vs_cycle280_delta:+.4f}")
lines.append(f"- Calibrated best_mean vs cycle 280 retrain anchor ({CYCLE_280_B1_R2_RETRAIN}): {calibrated['best_mean'] - CYCLE_280_B1_R2_RETRAIN:+.4f}")
lines.append(f"- Init gap closed by retrain (uncentered): {uncentered_init_ppl - uncentered['best_mean']:+.4f}")
lines.append(f"- Init gap closed by retrain (calibrated): {calibrated_init_ppl_from_Wr - calibrated['best_mean']:+.4f}")
lines.append("")
lines.append("## Sanity")
lines.append("")
lines.append(f"- Float PPL {fmt(float_ppl)} matches cycle ref {FLOAT_BASELINE_REF} within 0.5")
lines.append(f"- Uncentered init PPL matches cycle 280 published {CYCLE_280_B1_R2_INIT} within 0.01 (|delta|={unc_delta:.4f})")
lines.append(f"- Calibrated init PPL matches cycle 313 published {CYCLE_313_B1_R2_CAL_INIT} within 0.05 (|delta|={cal_delta:.4f})")
lines.append(f"- SVD(W_r_cal) reconstruction ||A·B − W_r||/||W_r|| = {recon_err:.2e}")
lines.append(f"- Σ symmetric err = {sigma_sym:.2e}; ||A²−Σ||/||Σ|| = {aa_err:.2e}; pinv rel err = {pinv_err:.2e}")
lines.append(f"- All 6 fits: trainable count == 2,304 asserted; |restored − best| < 1e-3 asserted")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
