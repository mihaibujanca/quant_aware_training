"""
Cycle 299 — Item 309: Cycle 280 retrain decomposition.

Re-run cycle 280's block-1 rank-2 Adam retrain (seed 42 only) with extra
logging: save B(t) at every 100 steps, compute drift metrics relative to the
SVD init, plot trajectory vs eval PPL.

Verdict per item 309:
  CONFIRMED drift   iff rowspace_cos(B(4000), B(0)) <= 0.6 AND PPL tracks drift (corr <= -0.5)
  CONFIRMED stay-put iff rowspace_cos(B(t), B(0)) >= 0.95 for all snapshots
  INCONCLUSIVE otherwise
"""
import json
import math
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
SEED = 42

FLOAT_BASELINE_REF = 53.895   # cycle 266/268/269/270
CYCLE_280_INIT_REF = 57.664   # cycle 280 noretrain r=2 (seed-independent)
CYCLE_280_BEST_REF = 49.032   # cycle 280 retrain best_mean
CONFIRMED_DRIFT_COS_LE = 0.6
CONFIRMED_STAY_COS_GE = 0.95
PPL_TRACKING_CORR_LE = -0.5

OUTPUT_PATH = "autoresearch/results/cycle_299_decomp.json"
SUMMARY_PATH = "autoresearch/results/cycle_299_summary.md"
PLOT_PATH = "autoresearch/results/cycle_299_drift.png"

torch.manual_seed(SEED)
random.seed(SEED)
print(f"Device: {DEVICE}", flush=True)


# ── Load model + data ────────────────────────────────────────────────────────
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
    """Mirrors cycle 280's replacement for HF Conv1D c_proj at rank r."""

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


def orthonormal_row_basis(M):
    """Return an orthonormal basis of row(M) as (r, d). M shape (r, d)."""
    # Use SVD of M^T: M^T = U Σ V^T → rows of V^T (= columns of V) are right SVs
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    # Vh is (r, d); rows are orthonormal right singular vectors, which span row(M)
    return Vh  # (r, d)


def rowspace_cos(Bt, B0):
    """Principal-angle-based similarity between row(Bt) and row(B0).

    Returns ||Q0 @ Qt.T||_F / sqrt(r) where Q0, Qt are orthonormal row bases
    (the normalized sum of squared cosines of the principal angles; 1.0 iff
    subspaces coincide).
    """
    Q0 = orthonormal_row_basis(B0)  # (r, d)
    Qt = orthonormal_row_basis(Bt)  # (r, d)
    M = Q0 @ Qt.T  # (r, r)
    # sum of squared singular values = ||M||_F^2; for equal subspaces this equals r
    return float(torch.linalg.norm(M).item() / math.sqrt(Q0.shape[0]))


def top2_right_sv_cosines(W, ref_Vh2):
    """Compute cosines of W's top-2 right SVs with ref_Vh2[:2, :].

    W shape (3072, 768). Returns (s0, s1, cos0, cos1).
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    v_t0 = Vh[0]
    v_t1 = Vh[1]
    r0 = ref_Vh2[0]
    r1 = ref_Vh2[1]
    # Top-r right SVs are defined up to sign → use absolute cosine
    cos0 = float(abs((v_t0 @ r0).item()))
    cos1 = float(abs((v_t1 @ r1).item()))
    return float(S[0].item()), float(S[1].item()), cos0, cos1


def pearson_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# ── Float baseline + SVD ─────────────────────────────────────────────────────
t0_total = time.time()
print("Float baseline PPL...", flush=True)
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
t0 = time.time()
float_ppl = eval_ppl(model, eval_ids)
print(f"Float PPL: {float_ppl:.3f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from cycle reference {FLOAT_BASELINE_REF} by > 0.5"
)

print(f"\nSVD of block {TARGET_BLOCK} c_proj.weight...", flush=True)
c_proj_orig_weight = model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.detach().clone()
c_proj_orig_bias = model.transformer.h[TARGET_BLOCK].mlp.c_proj.bias.data.detach().clone()
assert c_proj_orig_weight.shape == (3072, 768)
W_orig = c_proj_orig_weight.float().cpu()
t0 = time.time()
U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
print(f"  SVD done in {time.time()-t0:.1f}s; S[:5]={S[:5].tolist()}", flush=True)

A2 = (U[:, :RETRAIN_RANK] * S[:RETRAIN_RANK]).contiguous()  # (3072, 2)
B2_init = Vh[:RETRAIN_RANK, :].contiguous()  # (2, 768)

# Reference: orthonormal row basis of the SVD init
Vh_ref = Vh[:RETRAIN_RANK, :].contiguous()  # rows already orthonormal

# Sanity: top-2 right SVs of W_orig are exactly Vh[:2, :]
assert torch.allclose(Vh[:RETRAIN_RANK], Vh_ref, atol=1e-6)


# ── Build rank-2 model ────────────────────────────────────────────────────────
print(f"\n=== Retrain at r={RETRAIN_RANK} (seed {SEED}) with B(t) trajectory ===", flush=True)
torch.manual_seed(SEED)
random.seed(SEED)

model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_s.eval()

lowrank = LowRankCProj(
    A_frozen=A2.to(device=DEVICE, dtype=torch.float32),
    B_init=B2_init.to(device=DEVICE, dtype=torch.float32),
    b_init=c_proj_orig_bias.to(device=DEVICE, dtype=torch.float32),
    nf=768,
).to(DEVICE)
model_s.transformer.h[TARGET_BLOCK].mlp.c_proj = lowrank

init_ppl = eval_ppl(model_s, eval_ids)
print(f"  init PPL (rank-{RETRAIN_RANK} no retrain): {init_ppl:.4f} (cycle 280 ref {CYCLE_280_INIT_REF})", flush=True)
assert abs(init_ppl - CYCLE_280_INIT_REF) < 0.05, (
    f"init PPL {init_ppl} != cycle 280 ref {CYCLE_280_INIT_REF} within 0.05"
)

for p in model_s.parameters():
    p.requires_grad_(False)
lowrank.B.requires_grad_(True)
lowrank.bias.requires_grad_(True)
trainable = [lowrank.B, lowrank.bias]
n_trainable = sum(p.numel() for p in trainable)
expected_trainable = RETRAIN_RANK * 768 + 768
print(f"  Trainable params: {n_trainable:,} (expect {expected_trainable:,})", flush=True)
assert n_trainable == expected_trainable

optimizer = torch.optim.Adam(trainable, lr=LR)

calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
random.shuffle(calib_ids)

# ── Trajectory bookkeeping ────────────────────────────────────────────────────
B0_cpu = lowrank.B.detach().float().cpu().clone()
bias0_cpu = lowrank.bias.detach().float().cpu().clone()
B0_norm = float(torch.linalg.norm(B0_cpu).item())
Vh_ref_cpu = Vh_ref.cpu().float()

# Double-check: row-space cos of B(0) with itself should be 1 (within float)
rc0 = rowspace_cos(B0_cpu, B0_cpu)
assert abs(rc0 - 1.0) < 1e-5, f"rowspace_cos(B0, B0) = {rc0}"

def snapshot(step, eval_ppl_val):
    B_t = lowrank.B.detach().float().cpu().clone()
    bias_t = lowrank.bias.detach().float().cpu().clone()
    fro_abs = float(torch.linalg.norm(B_t - B0_cpu).item())
    fro_rel = fro_abs / B0_norm
    cos_b_b0 = rowspace_cos(B_t, B0_cpu)
    cos_b_vh = rowspace_cos(B_t, Vh_ref_cpu)  # sanity: should equal cos_b_b0 (same subspace)
    # Composite operator W(t) = A @ B(t)
    W_t = A2.cpu().float() @ B_t  # (3072, 768)
    sv0, sv1, cos0, cos1 = top2_right_sv_cosines(W_t, Vh_ref_cpu)
    bias_delta = float(torch.linalg.norm(bias_t - bias0_cpu).item())
    return {
        "step": int(step),
        "eval_ppl": float(eval_ppl_val),
        "fro_abs": fro_abs,
        "fro_rel": fro_rel,
        "rowspace_cos_B_B0": cos_b_b0,
        "rowspace_cos_B_Vh": cos_b_vh,
        "W_sv0": sv0,
        "W_sv1": sv1,
        "W_top1_cos_vs_orig_Vh": cos0,
        "W_top2_cos_vs_orig_Vh": cos1,
        "bias_delta_norm": bias_delta,
    }

trajectory = [snapshot(0, init_ppl)]
print(f"  step    0: ppl={init_ppl:.4f}  cos(B,B0)={trajectory[0]['rowspace_cos_B_B0']:.4f}  fro_rel={trajectory[0]['fro_rel']:.4e}", flush=True)

# ── Training loop with snapshots every 100 steps ──────────────────────────────
best_ppl = init_ppl
best_step = 0
best_state = {
    "B": lowrank.B.detach().clone(),
    "bias": lowrank.bias.detach().clone(),
}

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
            snap = snapshot(step, ppl_e)
            trajectory.append(snap)
            if ppl_e < best_ppl:
                best_ppl = ppl_e
                best_step = step
                best_state = {
                    "B": lowrank.B.detach().clone(),
                    "bias": lowrank.bias.detach().clone(),
                }
            if step % 500 == 0:
                print(
                    f"  step {step:4d}: ppl={ppl_e:.4f}  best={best_ppl:.4f}@{best_step}  "
                    f"cos(B,B0)={snap['rowspace_cos_B_B0']:.4f}  fro_rel={snap['fro_rel']:.4e}  "
                    f"bias_Δ={snap['bias_delta_norm']:.4e}",
                    flush=True,
                )

final_ppl = eval_ppl(model_s, eval_ids)
# Restore best
with torch.no_grad():
    lowrank.B.data.copy_(best_state["B"])
    lowrank.bias.data.copy_(best_state["bias"])
restored_ppl = eval_ppl(model_s, eval_ids)
assert abs(restored_ppl - best_ppl) < 1e-3, f"restored {restored_ppl} != best {best_ppl}"
assert best_ppl < init_ppl, f"training failed: best {best_ppl} >= init {init_ppl}"
dt_train = time.time() - t_train
print(
    f"\nFinal: init={init_ppl:.4f}  best={best_ppl:.4f}@{best_step}  "
    f"final@{N_STEPS}={final_ppl:.4f}  restored={restored_ppl:.4f}  [{dt_train/60:.1f} min]",
    flush=True,
)

# ── Drift summary metrics ─────────────────────────────────────────────────────
final_snap = trajectory[-1]
cos_final = final_snap["rowspace_cos_B_B0"]
fro_rel_final = final_snap["fro_rel"]
bias_delta_final = final_snap["bias_delta_norm"]

cos_series = [s["rowspace_cos_B_B0"] for s in trajectory]
ppl_series = [s["eval_ppl"] for s in trajectory]
cos_min = min(cos_series)
cos_max = max(cos_series)
always_stay_put = all(c >= CONFIRMED_STAY_COS_GE for c in cos_series)

# Correlation between cos drop and PPL drop (exclude step 0)
# Flip sign: we expect cos to decrease while ppl decreases → positive corr between
# cos AND ppl (both falling). For "PPL tracks drift" we want negative corr between
# cos and ppl? No: if cos falls and ppl falls, then corr(cos, ppl) is POSITIVE.
# We want to test whether the drift MOVES with the ppl improvement, so we report
# corr(ΔB_from_init, -Δppl_from_init) — equivalent to corr(fro_rel, init_ppl - ppl).
fro_series = [s["fro_rel"] for s in trajectory]
ppl_improvement = [init_ppl - p for p in ppl_series]
corr_fro_improvement = pearson_corr(fro_series, ppl_improvement)
# Item 309 phrased it as "held-out PPL improvement tracks the drift": we want
# high correlation between fro_rel and ppl_improvement. For the CONFIRMED drift
# clause we use corr >= 0.5 as the tracking criterion (equivalently, cos and ppl
# are strongly positively correlated because both fall together).
corr_cos_ppl = pearson_corr(cos_series, ppl_series)

print(f"\n=== Drift summary ===", flush=True)
print(f"  rowspace_cos(B(t), B(0)): init={cos_series[0]:.4f}  final={cos_final:.4f}  min={cos_min:.4f}  max={cos_max:.4f}", flush=True)
print(f"  fro_rel final: {fro_rel_final:.4e}", flush=True)
print(f"  bias delta norm final: {bias_delta_final:.4e}", flush=True)
print(f"  corr(fro_rel, ppl_improvement) = {corr_fro_improvement:.4f}  (expect >= +0.5 if tracking)", flush=True)
print(f"  corr(cos, ppl) = {corr_cos_ppl:.4f}  (expect >= +0.5 since both fall together if tracking)", flush=True)

# ── Verdict per item 309 ──────────────────────────────────────────────────────
drift_cos_ok = cos_final <= CONFIRMED_DRIFT_COS_LE
tracking_ok = corr_fro_improvement >= 0.5  # strong positive = drift tracks improvement
stay_put_ok = always_stay_put

if drift_cos_ok and tracking_ok:
    verdict = "CONFIRMED_drift"
elif stay_put_ok:
    verdict = "CONFIRMED_stay_put"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== Verdict ===", flush=True)
print(f"  cos_final = {cos_final:.4f}  (CONFIRMED_drift iff <= {CONFIRMED_DRIFT_COS_LE})", flush=True)
print(f"  all cos >= {CONFIRMED_STAY_COS_GE}? {stay_put_ok}  (CONFIRMED_stay_put iff True)", flush=True)
print(f"  tracking (corr(fro_rel, ppl_improve) >= 0.5)? {tracking_ok}  (val={corr_fro_improvement:.4f})", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 299,
    "item": 309,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "cycle_280_init_ref": CYCLE_280_INIT_REF,
    "cycle_280_best_ref": CYCLE_280_BEST_REF,
    "init_ppl": float(init_ppl),
    "best_ppl": float(best_ppl),
    "best_step": int(best_step),
    "final_ppl_at_n_steps": float(final_ppl),
    "restored_best_ppl": float(restored_ppl),
    "adam_recovery": float(init_ppl - best_ppl),
    "trajectory": trajectory,
    "drift_summary": {
        "cos_initial": cos_series[0],
        "cos_final": cos_final,
        "cos_min": cos_min,
        "cos_max": cos_max,
        "fro_rel_final": fro_rel_final,
        "bias_delta_final": bias_delta_final,
        "corr_fro_ppl_improvement": corr_fro_improvement,
        "corr_cos_ppl": corr_cos_ppl,
        "always_stay_put": bool(stay_put_ok),
        "drift_cos_ok": bool(drift_cos_ok),
        "tracking_ok": bool(tracking_ok),
    },
    "verdict_logic": {
        "threshold_drift_cos_le": CONFIRMED_DRIFT_COS_LE,
        "threshold_stay_cos_ge": CONFIRMED_STAY_COS_GE,
        "threshold_tracking_corr_ge": 0.5,
        "cos_final": cos_final,
        "corr_fro_ppl_improvement": corr_fro_improvement,
        "verdict": verdict,
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
        "SEED": SEED,
        "trainable_params": RETRAIN_RANK * 768 + 768,
        "n_snapshots": len(trajectory),
    },
    "svd_summary": {
        "top_5_sv": S[:5].tolist(),
        "S_min": float(S[-1]),
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
steps = [s["step"] for s in trajectory]

axes[0].plot(steps, ppl_series, "b-o", markersize=3)
axes[0].axhline(FLOAT_BASELINE_REF, color="k", linestyle="--", label=f"float {FLOAT_BASELINE_REF:.2f}")
axes[0].axhline(init_ppl, color="r", linestyle=":", label=f"init {init_ppl:.2f}")
axes[0].set_xlabel("step")
axes[0].set_ylabel("eval PPL")
axes[0].set_title(f"Eval PPL (best {best_ppl:.3f} @ {best_step})")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(steps, cos_series, "g-o", markersize=3)
axes[1].axhline(CONFIRMED_DRIFT_COS_LE, color="r", linestyle="--", label=f"drift ≤ {CONFIRMED_DRIFT_COS_LE}")
axes[1].axhline(CONFIRMED_STAY_COS_GE, color="b", linestyle="--", label=f"stay ≥ {CONFIRMED_STAY_COS_GE}")
axes[1].set_xlabel("step")
axes[1].set_ylabel("rowspace cos(B(t), B(0))")
axes[1].set_title(f"Row-space drift (final {cos_final:.4f})")
axes[1].set_ylim(0, 1.05)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].plot(steps, fro_series, "m-o", markersize=3)
axes[2].set_xlabel("step")
axes[2].set_ylabel("||B(t) - B(0)||_F / ||B(0)||_F")
axes[2].set_title(f"Relative Frobenius drift (final {fro_rel_final:.3e})")
axes[2].grid(True, alpha=0.3)

fig.suptitle(
    f"Cycle 299 — item 309: B(t) drift during cycle 280 retrain (seed {SEED})",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {PLOT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"

lines = []
lines.append(f"# Cycle 299 — Item 309: Cycle 280 retrain decomposition (B drift)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Block {TARGET_BLOCK}, rank {RETRAIN_RANK} LowRankCProj, seed {SEED}")
lines.append(f"- Init PPL: {fmt(init_ppl)} (cycle 280 ref {CYCLE_280_INIT_REF})")
lines.append(f"- Best PPL: {fmt(best_ppl)} @ step {best_step}  (cycle 280 ref {CYCLE_280_BEST_REF})")
lines.append(f"- Adam recovery: {init_ppl - best_ppl:+.3f} PPL over {N_STEPS} steps")
lines.append(f"- Final rowspace cos(B(t), B(0)) = **{fmt(cos_final)}**")
lines.append(f"- Final fro_rel = {fmt(fro_rel_final, 4)}")
lines.append(f"- Final bias delta norm = {fmt(bias_delta_final, 4)}")
lines.append(f"- corr(fro_rel, ppl_improvement) = {fmt(corr_fro_improvement)}")
lines.append("")
lines.append("## Trajectory (every 100 steps)")
lines.append("")
lines.append("| step | eval_ppl | cos(B,B0) | fro_rel | W_sv0 | W_sv1 | top1_cos_Vh | top2_cos_Vh | bias_Δ |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for s in trajectory[::5]:  # every 500 steps for brevity
    lines.append(
        f"| {s['step']} | {fmt(s['eval_ppl'],3)} | {fmt(s['rowspace_cos_B_B0'])} | "
        f"{s['fro_rel']:.3e} | {fmt(s['W_sv0'],3)} | {fmt(s['W_sv1'],3)} | "
        f"{fmt(s['W_top1_cos_vs_orig_Vh'])} | {fmt(s['W_top2_cos_vs_orig_Vh'])} | "
        f"{s['bias_delta_norm']:.3e} |"
    )
lines.append(f"| {trajectory[-1]['step']} | {fmt(trajectory[-1]['eval_ppl'],3)} | "
             f"{fmt(trajectory[-1]['rowspace_cos_B_B0'])} | {trajectory[-1]['fro_rel']:.3e} | "
             f"{fmt(trajectory[-1]['W_sv0'],3)} | {fmt(trajectory[-1]['W_sv1'],3)} | "
             f"{fmt(trajectory[-1]['W_top1_cos_vs_orig_Vh'])} | {fmt(trajectory[-1]['W_top2_cos_vs_orig_Vh'])} | "
             f"{trajectory[-1]['bias_delta_norm']:.3e} |")
lines.append("")
lines.append("## Verdict logic (item 309 verbatim)")
lines.append("")
lines.append(f"- **CONFIRMED drift** iff rowspace_cos(B(4000), B(0)) ≤ {CONFIRMED_DRIFT_COS_LE} AND held-out PPL improvement tracks drift (corr(fro_rel, ppl_improvement) ≥ 0.5).")
lines.append(f"- **CONFIRMED stay-put** iff rowspace_cos(B(t), B(0)) ≥ {CONFIRMED_STAY_COS_GE} for all snapshots.")
lines.append(f"- **INCONCLUSIVE** otherwise.")
lines.append("")
lines.append(f"- cos_final = {fmt(cos_final)}  → drift_cos_ok (≤ {CONFIRMED_DRIFT_COS_LE})? **{drift_cos_ok}**")
lines.append(f"- all cos ≥ {CONFIRMED_STAY_COS_GE}? **{stay_put_ok}**  (min={fmt(cos_min)}, max={fmt(cos_max)})")
lines.append(f"- corr(fro_rel, ppl_improvement) = {fmt(corr_fro_improvement)}  → tracking_ok? **{tracking_ok}**")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append("## Interpretation")
lines.append("")
if verdict == "CONFIRMED_drift":
    lines.append("B(t) has drifted OUT of the SVD row-space. Adam is finding a DIFFERENT rank-2 operator, "
                 "with its column space fixed by the frozen A factor. Implication: eff_rank predicts r but "
                 "not the optimal B; pick r, train B from any reasonable init.")
elif verdict == "CONFIRMED_stay_put":
    lines.append("B(t) has stayed inside the SVD row-space throughout. The rank-2 weight is essentially "
                 "the SVD truncation plus a small calibration. Implication: a pure closed-form SVD-LLM "
                 "rescaling could match cycle 280 without Adam.")
else:
    lines.append("Neither the drift nor the stay-put clause fires cleanly. The substantive reading "
                 "depends on which quantities moved; see the trajectory table above.")
lines.append("")
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)

print(f"\nTotal wall: {time.time() - t0_total:.1f}s", flush=True)
