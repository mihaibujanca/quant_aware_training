"""
Cycle 316 — Item 326: (b10, b11) destructive pair deep dive.

Cycle 310 found (b10, b11) is the only destructive pair at +6.25 PPL above
worst-single. Three candidate mechanisms:
  M1 — residual stream redundancy (b10 and b11 write similar directions)
  M2 — LayerNorm coupling (b10 distorts b11's LN input)
  M3 — joint Adam saddle (sequential optimisation would escape it)

Three tests on the same wikitext-2 100x128 eval slice / SEED=42 / N_STEPS=2000:
  Test M1 — principal angle between Vh_10[:5] and Vh_11[:22] of Δy_l, where
            Δy_l is computed two ways:
              (a) raw     = GELU(z_l) @ W_proj_l       (full FFN residual write)
              (b) curv    = (GELU(z_l)-ReLU(z_l)) @ W_proj_l   (cycle 302/304 form)
            Report mean angle (degrees) for both rank pairings (5x22 and 5x5).
            Threshold: mean angle < 30° => M1 supported.
  Test M2 — sequential Adam: install LowRankCProj at b10 r=5 AND b11 r=22.
            Phase A: freeze everything except {B_10, bias_10}, Adam LR=1e-4,
                     N_STEPS=2000, EVAL_EVERY=100, best-state restore.
            Phase B: copy phase-A best back, freeze {B_10, bias_10}, unfreeze
                     {B_11, bias_11}, FRESH Adam, same budget, best-state restore.
            sequential_best_ppl = min(eval PPL across both phases).
            Threshold: sequential_best_ppl <= 55 => M3 confirmed (joint-Adam saddle).
  Test M3 — one-sided restores:
            Ablation A: b10 float + b11 LowRankCProj r=22, Adam over {B_11, bias_11}
                        N_STEPS=2000.
            Ablation B: b10 LowRankCProj r=5 + b11 float, Adam over {B_10, bias_10}
                        N_STEPS=2000.
            Compare both to cycle 310 joint best 60.7161 and cycle-282-family
            single-block diagonals (b10=51.4659, b11=54.4687).

Verdict (item 326 verbatim):
  sequential_ppl <= 55                                  => CONFIRMED-M3-pathology
  sequential_ppl >= 60 AND mean_angle < 30°             => CONFIRMED-M1-redundancy
  sequential_ppl >= 60 AND mean_angle > 60°             => REFUTED-mechanism-unknown
  else                                                  => INCONCLUSIVE

LowRankCProj template matches cycle 270/282/310 verbatim:
  W_l (3072,768) -> SVD -> U S Vh
  A_r = U[:,:r]*S[:r] (3072,r) frozen buffer
  B_r = Vh[:r,:]      (r,768)  trainable
  bias = c_proj.bias   (768,)  trainable
  forward: x @ (A_r @ B_r) + bias  (HF Conv1D semantics)
"""
import json
import math
import random
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
B10 = 10
B11 = 11
R10 = 5
R11 = 22

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 2000
LR = 1e-4
EVAL_EVERY = 100
SEED = 42

OUTPUT_PATH = "autoresearch/results/cycle_316_b10_b11_deep_dive.json"
SUMMARY_PATH = "autoresearch/results/cycle_316_summary.md"

FLOAT_BASELINE_REF = 53.895
CYCLE_270_B11_R22_BEST_MEAN = 54.4687
CYCLE_282_B10_R5_DIAG = 51.4659     # cycle 289 b10 r=5 single-block anchor (used by cycle 310 diag)
CYCLE_310_B10_B11_JOINT_BEST = 60.7161  # cycle 310 (b10,b11) joint Adam best, SEED=42, N_STEPS=2000
CYCLE_310_B10_B11_INIT = 76.6535        # cycle 310 (b10,b11) joint init no-retrain

# Verdict thresholds (item 326 verbatim)
M3_PATHOLOGY_LE = 55.0
DESTRUCTIVE_GE = 60.0
M1_ANGLE_LT = 30.0   # degrees
M1_REFUTE_GT = 60.0  # degrees

print(f"Device: {DEVICE}", flush=True)
torch.manual_seed(SEED)
random.seed(SEED)


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
    """HF Conv1D c_proj rank-r replacement (cycle 270/282/310 template)."""

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


def capture_cfc_inputs(model, blocks, calib_ids_dev):
    """Forward calib once; capture ln_2(x) at each hooked c_fc input.
    Returns dict[l] -> (N_tokens, 768) CPU float32.
    """
    stash = {l: [] for l in blocks}

    def make_prehook(l):
        def fn(_mod, args):
            stash[l].append(args[0].detach().float().cpu())
        return fn

    hooks = []
    for l in blocks:
        h = model.transformer.h[l].mlp.c_fc.register_forward_pre_hook(make_prehook(l))
        hooks.append(h)
    model.eval()
    with torch.no_grad():
        for ids in calib_ids_dev:
            _ = model(ids)
    for h in hooks:
        h.remove()
    out = {}
    for l in blocks:
        t = torch.cat(stash[l], dim=0).reshape(-1, 768)
        out[l] = t
        stash[l] = None
    return out


def principal_angles_deg(V_a, V_b):
    """Principal angles in degrees between row spaces of V_a (k_a, d) and V_b (k_b, d).
    Both inputs should have orthonormal rows. Returns (cosines tensor, angles_deg tensor).
    """
    M = V_a @ V_b.T  # (k_a, k_b)
    cos = torch.linalg.svdvals(M)  # min(k_a, k_b) values in [0, 1]
    cos = cos.clamp(min=0.0, max=1.0)
    angles = torch.rad2deg(torch.arccos(cos))
    return cos, angles


# ── Float baseline + SVD cache ────────────────────────────────────────────────
print("\nFloat baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
t0 = time.time()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f}  ({time.time() - t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5

svd_cache = {}
for blk, r in [(B10, R10), (B11, R11)]:
    print(f"\nSVD of block {blk} c_proj.weight (target rank {r})...", flush=True)
    W_orig = base_model.transformer.h[blk].mlp.c_proj.weight.data.detach().clone()
    b_orig = base_model.transformer.h[blk].mlp.c_proj.bias.data.detach().clone()
    assert W_orig.shape == (3072, 768)
    W_cpu = W_orig.float().cpu()
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    print(f"  SVD done in {time.time() - t0:.1f}s; U {tuple(U.shape)} S {tuple(S.shape)} Vh {tuple(Vh.shape)}", flush=True)
    A_r = (U[:, :r] * S[:r]).contiguous()
    B_r = Vh[:r, :].contiguous()
    svd_cache[blk] = {
        "A": A_r,
        "B_init": B_r,
        "b_init": b_orig,
        "S_top5": [float(s) for s in S[:5].tolist()],
        "S_min": float(S[-1]),
        "rank": r,
    }


# ── Test M1 — Δy SVD principal angle ────────────────────────────────────────
print("\n=== M1 test: principal angle between Δy_10 and Δy_11 ===", flush=True)
calib_ids_dev = [t.to(DEVICE) for t in calib_ids_cpu]
mlp_inputs = capture_cfc_inputs(base_model, [B10, B11], calib_ids_dev)
print(f"  captured ln_2(x) at b10: {tuple(mlp_inputs[B10].shape)}", flush=True)
print(f"  captured ln_2(x) at b11: {tuple(mlp_inputs[B11].shape)}", flush=True)

m1_results = {}
for blk in [B10, B11]:
    block = base_model.transformer.h[blk]
    W_fc = block.mlp.c_fc.weight.detach().float().cpu()      # (768, 3072)
    b_fc = block.mlp.c_fc.bias.detach().float().cpu()        # (3072,)
    W_proj = block.mlp.c_proj.weight.detach().float().cpu()  # (3072, 768)
    x = mlp_inputs[blk]
    z = x @ W_fc + b_fc                            # (N, 3072)
    Dy_raw = F.gelu(z, approximate="tanh") @ W_proj  # full FFN residual write
    Dy_curv = (F.gelu(z, approximate="tanh") - F.relu(z)) @ W_proj  # cycle 302 form
    m1_results[blk] = {
        "Dy_raw": Dy_raw,
        "Dy_curv": Dy_curv,
        "Dy_raw_frob": float(torch.linalg.norm(Dy_raw).item()),
        "Dy_curv_frob": float(torch.linalg.norm(Dy_curv).item()),
    }
    print(f"  block {blk}: Dy_raw frob={m1_results[blk]['Dy_raw_frob']:.3f} | Dy_curv frob={m1_results[blk]['Dy_curv_frob']:.3f}", flush=True)

# SVD top-r right singular vectors of Dy
m1_svd = {"raw": {}, "curv": {}}
for variant, key in [("raw", "Dy_raw"), ("curv", "Dy_curv")]:
    for blk in [B10, B11]:
        Dy = m1_results[blk][key]
        U_, S_, Vh_ = torch.linalg.svd(Dy, full_matrices=False)
        m1_svd[variant][blk] = {
            "Vh": Vh_.contiguous(),
            "S_top5": [float(s) for s in S_[:5].tolist()],
        }
        # orthonormality sanity on top-22
        max_r = max(R10, R11)
        VhR = Vh_[:max_r, :]
        II = VhR @ VhR.T
        off = float((II - torch.eye(max_r)).abs().max().item())
        assert off < 1e-4, f"Vh ortho fail block {blk} variant {variant}: off={off}"

# Principal angles between Vh_10[:R10] and Vh_11[:R11] (5 cosines, since k=min(5,22)=5)
# AND between Vh_10[:R10] and Vh_11[:R10] (5x5).
m1_angles = {}
for variant in ["raw", "curv"]:
    Vh10 = m1_svd[variant][B10]["Vh"]
    Vh11 = m1_svd[variant][B11]["Vh"]

    # rank pair (5, 22): use the first R10 SVs of b10 and the first R11 of b11
    cos_5_22, ang_5_22 = principal_angles_deg(Vh10[:R10], Vh11[:R11])
    # rank pair (5, 5): symmetric matched
    cos_5_5, ang_5_5 = principal_angles_deg(Vh10[:R10], Vh11[:R10])
    # rank pair (22, 22): full b11 budget on both sides (use top-22 of both)
    cos_22_22, ang_22_22 = principal_angles_deg(Vh10[:R11], Vh11[:R11])

    m1_angles[variant] = {
        "cos_5_22": [float(c) for c in cos_5_22.tolist()],
        "ang_5_22_deg": [float(a) for a in ang_5_22.tolist()],
        "ang_5_22_mean_deg": float(ang_5_22.mean().item()),
        "ang_5_22_min_deg": float(ang_5_22.min().item()),
        "cos_5_5": [float(c) for c in cos_5_5.tolist()],
        "ang_5_5_deg": [float(a) for a in ang_5_5.tolist()],
        "ang_5_5_mean_deg": float(ang_5_5.mean().item()),
        "ang_5_5_min_deg": float(ang_5_5.min().item()),
        "cos_22_22": [float(c) for c in cos_22_22.tolist()],
        "ang_22_22_mean_deg": float(ang_22_22.mean().item()),
        "ang_22_22_min_deg": float(ang_22_22.min().item()),
    }
    print(f"  variant={variant}", flush=True)
    print(f"    cos(5x22): {[f'{c:.3f}' for c in m1_angles[variant]['cos_5_22']]}", flush=True)
    print(f"    ang(5x22) mean={m1_angles[variant]['ang_5_22_mean_deg']:.2f}°  min={m1_angles[variant]['ang_5_22_min_deg']:.2f}°", flush=True)
    print(f"    ang(5x5)  mean={m1_angles[variant]['ang_5_5_mean_deg']:.2f}°  min={m1_angles[variant]['ang_5_5_min_deg']:.2f}°", flush=True)
    print(f"    ang(22x22) mean={m1_angles[variant]['ang_22_22_mean_deg']:.2f}°", flush=True)

# Use the smaller mean angle across (raw, curv) and across (5x22, 5x5) — this is the
# strongest evidence of redundancy any reasonable rank pairing supports.
m1_min_mean_angle = min(
    m1_angles["raw"]["ang_5_22_mean_deg"],
    m1_angles["raw"]["ang_5_5_mean_deg"],
    m1_angles["curv"]["ang_5_22_mean_deg"],
    m1_angles["curv"]["ang_5_5_mean_deg"],
)
print(f"  ==> M1 min mean angle (across raw/curv x 5x22/5x5): {m1_min_mean_angle:.2f}°", flush=True)

# Free large tensors before retrains
del mlp_inputs, m1_results, m1_svd
del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Adam fit helper ───────────────────────────────────────────────────────────
def adam_fit(model, trainable_params, n_steps, lr, eval_every, label):
    """Run Adam over `trainable_params` for n_steps; track best eval PPL.

    Returns (best_ppl, best_step, history, best_state_dict).
    best_state_dict copies the trainable parameter tensors at best_ppl time.
    """
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
    random.shuffle(calib_ids)

    init_ppl = eval_ppl(model, eval_ids)
    print(f"  [{label}] init PPL: {init_ppl:.4f}", flush=True)
    best_ppl = init_ppl
    best_step = 0
    best_state = [p.detach().clone() for p in trainable_params]
    history = [{"step": 0, "eval_ppl": float(init_ppl)}]

    step = 0
    t0 = time.time()
    while step < n_steps:
        for ids in calib_ids:
            if step >= n_steps:
                break
            model.train()
            optimizer.zero_grad()
            out = model(ids, labels=ids)
            loss = out.loss
            loss.backward()
            optimizer.step()
            step += 1

            if step % eval_every == 0:
                ppl_e = eval_ppl(model, eval_ids)
                history.append({
                    "step": step,
                    "eval_ppl": float(ppl_e),
                    "calib_loss": float(loss.item()),
                })
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    best_state = [p.detach().clone() for p in trainable_params]
                if step % 500 == 0:
                    print(f"    [{label}] step {step:4d}: eval_ppl={ppl_e:.4f}  best={best_ppl:.4f}@{best_step}", flush=True)

    final_ppl = eval_ppl(model, eval_ids)

    # Restore best
    with torch.no_grad():
        for p, s in zip(trainable_params, best_state):
            p.data.copy_(s)
    restored_ppl = eval_ppl(model, eval_ids)
    assert abs(restored_ppl - best_ppl) < 1e-3, (
        f"[{label}] restored {restored_ppl} != best {best_ppl}"
    )
    dt = time.time() - t0
    print(f"  [{label}] init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} "
          f"final@{n_steps}={final_ppl:.4f} restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

    return {
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_n_steps": float(final_ppl),
        "restored_best_ppl": float(restored_ppl),
        "history": history,
    }


def install_lowrank(model, blk, svd_cache):
    cache = svd_cache[blk]
    lr_mod = LowRankCProj(
        A_frozen=cache["A"].to(device=DEVICE, dtype=torch.float32),
        B_init=cache["B_init"].to(device=DEVICE, dtype=torch.float32),
        b_init=cache["b_init"].to(device=DEVICE, dtype=torch.float32),
        nf=768,
    ).to(DEVICE)
    model.transformer.h[blk].mlp.c_proj = lr_mod
    return lr_mod


# ── Test M2 — sequential Adam (b10 first, then b11) ────────────────────────
print("\n=== M2 test: sequential Adam (phase A b10 -> phase B b11) ===", flush=True)
torch.manual_seed(SEED)
random.seed(SEED)
model_seq = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_seq.eval()

lr10 = install_lowrank(model_seq, B10, svd_cache)
lr11 = install_lowrank(model_seq, B11, svd_cache)

init_seq = eval_ppl(model_seq, eval_ids)
print(f"  install init (both LowRankCProj installed, no fit): {init_seq:.4f}", flush=True)
assert abs(init_seq - CYCLE_310_B10_B11_INIT) < 0.5, (
    f"sequential init {init_seq} differs from cycle 310 init {CYCLE_310_B10_B11_INIT} by > 0.5"
)

# Phase A: only b10 trains
for p in model_seq.parameters():
    p.requires_grad_(False)
lr10.B.requires_grad_(True)
lr10.bias.requires_grad_(True)
trainable_A = [lr10.B, lr10.bias]
n_train_A = sum(p.numel() for p in trainable_A)
expected_A = R10 * 768 + 768
assert n_train_A == expected_A, (n_train_A, expected_A)
print(f"  phase A trainable: {n_train_A:,} (b10 r={R10})", flush=True)
phase_A = adam_fit(model_seq, trainable_A, N_STEPS, LR, EVAL_EVERY, label="phaseA-b10")

# Phase B: freeze b10 (already at phase-A best after restore), unfreeze b11, fresh Adam
for p in model_seq.parameters():
    p.requires_grad_(False)
lr11.B.requires_grad_(True)
lr11.bias.requires_grad_(True)
trainable_B = [lr11.B, lr11.bias]
n_train_B = sum(p.numel() for p in trainable_B)
expected_B = R11 * 768 + 768
assert n_train_B == expected_B, (n_train_B, expected_B)
print(f"  phase B trainable: {n_train_B:,} (b11 r={R11})", flush=True)
phase_B = adam_fit(model_seq, trainable_B, N_STEPS, LR, EVAL_EVERY, label="phaseB-b11")

sequential_best_ppl = min(phase_A["best_ppl"], phase_B["best_ppl"])
print(f"\n  ==> M2 sequential_best_ppl = {sequential_best_ppl:.4f}  "
      f"(phaseA best {phase_A['best_ppl']:.4f}, phaseB best {phase_B['best_ppl']:.4f})", flush=True)
print(f"      cycle 310 joint best (1 seed)  = {CYCLE_310_B10_B11_JOINT_BEST:.4f}", flush=True)
print(f"      delta vs joint                 = {sequential_best_ppl - CYCLE_310_B10_B11_JOINT_BEST:+.4f} PPL", flush=True)

del model_seq, lr10, lr11
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Test M3 — one-sided restores ────────────────────────────────────────────
print("\n=== M3 test: one-sided restores ===", flush=True)

# Ablation A: b10 float, b11 r=22 LowRankCProj fit
print("\n--- Ablation A: b10 float + b11 r=22 LowRankCProj ---", flush=True)
torch.manual_seed(SEED)
random.seed(SEED)
model_A = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_A.eval()
lr11_A = install_lowrank(model_A, B11, svd_cache)
for p in model_A.parameters():
    p.requires_grad_(False)
lr11_A.B.requires_grad_(True)
lr11_A.bias.requires_grad_(True)
trainable_M3A = [lr11_A.B, lr11_A.bias]
n_train_M3A = sum(p.numel() for p in trainable_M3A)
assert n_train_M3A == R11 * 768 + 768
ablation_A = adam_fit(model_M := model_A, trainable_M3A, N_STEPS, LR, EVAL_EVERY, label="M3A-b11only")
del model_A, lr11_A
if DEVICE == "mps":
    torch.mps.empty_cache()

# Ablation B: b10 r=5 LowRankCProj fit, b11 float
print("\n--- Ablation B: b10 r=5 LowRankCProj + b11 float ---", flush=True)
torch.manual_seed(SEED)
random.seed(SEED)
model_B = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_B.eval()
lr10_B = install_lowrank(model_B, B10, svd_cache)
for p in model_B.parameters():
    p.requires_grad_(False)
lr10_B.B.requires_grad_(True)
lr10_B.bias.requires_grad_(True)
trainable_M3B = [lr10_B.B, lr10_B.bias]
n_train_M3B = sum(p.numel() for p in trainable_M3B)
assert n_train_M3B == R10 * 768 + 768
ablation_B = adam_fit(model_B, trainable_M3B, N_STEPS, LR, EVAL_EVERY, label="M3B-b10only")
del model_B, lr10_B
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Verdict ───────────────────────────────────────────────────────────────────
mean_angle_for_verdict = m1_min_mean_angle  # smallest mean across raw/curv x 5x22/5x5
if sequential_best_ppl <= M3_PATHOLOGY_LE:
    verdict = "CONFIRMED-M3-optimisation-pathology"
elif sequential_best_ppl >= DESTRUCTIVE_GE and mean_angle_for_verdict < M1_ANGLE_LT:
    verdict = "CONFIRMED-M1-residual-redundancy"
elif sequential_best_ppl >= DESTRUCTIVE_GE and mean_angle_for_verdict > M1_REFUTE_GT:
    verdict = "REFUTED-mechanism-unknown"
else:
    verdict = "INCONCLUSIVE"

print("\n=== Verdict ===", flush=True)
print(f"  M1 min mean angle:        {mean_angle_for_verdict:.2f}°  (M1 if <30°, REFUTE-unknown if >60°)", flush=True)
print(f"  M2 sequential_best_ppl:   {sequential_best_ppl:.4f}     (M3 if <=55, destructive if >=60)", flush=True)
print(f"  M3 ablation A (b11 only): {ablation_A['best_ppl']:.4f}    (cycle 270 ref {CYCLE_270_B11_R22_BEST_MEAN})", flush=True)
print(f"  M3 ablation B (b10 only): {ablation_B['best_ppl']:.4f}    (cycle 282 b10 r=5 diag {CYCLE_282_B10_R5_DIAG})", flush=True)
print(f"  cycle 310 joint best:     {CYCLE_310_B10_B11_JOINT_BEST}", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 316,
    "item": 326,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "config": {
        "B10": B10, "B11": B11, "R10": R10, "R11": R11,
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE, "N_STEPS": N_STEPS, "LR": LR,
        "EVAL_EVERY": EVAL_EVERY, "SEED": SEED,
    },
    "svd_cache_summary": {
        str(blk): {"S_top5": svd_cache[blk]["S_top5"], "S_min": svd_cache[blk]["S_min"], "rank": svd_cache[blk]["rank"]}
        for blk in [B10, B11]
    },
    "m1_test": {
        "description": "principal angles between top-r right SVs of Δy_l for l in {10,11}",
        "raw_variant": m1_angles["raw"],
        "curv_variant": m1_angles["curv"],
        "min_mean_angle_deg": mean_angle_for_verdict,
        "threshold_M1_lt_deg": M1_ANGLE_LT,
        "threshold_REFUTE_gt_deg": M1_REFUTE_GT,
    },
    "m2_test": {
        "description": "sequential Adam: phase A fits b10 (b11 frozen at SVD init), phase B freezes b10 at A best and fits b11",
        "install_init_ppl": float(init_seq),
        "phase_A_b10": phase_A,
        "phase_B_b11": phase_B,
        "sequential_best_ppl": float(sequential_best_ppl),
        "joint_best_cycle_310_ref": CYCLE_310_B10_B11_JOINT_BEST,
        "delta_vs_joint": float(sequential_best_ppl - CYCLE_310_B10_B11_JOINT_BEST),
        "threshold_M3_le": M3_PATHOLOGY_LE,
        "threshold_destructive_ge": DESTRUCTIVE_GE,
    },
    "m3_test": {
        "description": "one-sided restores: ablation A keeps b10 float and fits b11 only; ablation B keeps b11 float and fits b10 only",
        "ablation_A_b10float_b11r22": ablation_A,
        "ablation_B_b10r5_b11float": ablation_B,
        "cycle_270_b11_r22_ref": CYCLE_270_B11_R22_BEST_MEAN,
        "cycle_282_b10_r5_diag_ref": CYCLE_282_B10_R5_DIAG,
    },
    "verdict_logic": {
        "sequential_best_ppl": float(sequential_best_ppl),
        "min_mean_angle_deg": mean_angle_for_verdict,
        "threshold_M3_pathology_le": M3_PATHOLOGY_LE,
        "threshold_destructive_ge": DESTRUCTIVE_GE,
        "threshold_M1_angle_lt": M1_ANGLE_LT,
        "threshold_REFUTE_angle_gt": M1_REFUTE_GT,
        "verdict": verdict,
    },
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "cycle_270_b11_r22_best_mean": CYCLE_270_B11_R22_BEST_MEAN,
        "cycle_282_b10_r5_diag": CYCLE_282_B10_R5_DIAG,
        "cycle_310_b10_b11_joint_best": CYCLE_310_B10_B11_JOINT_BEST,
        "cycle_310_b10_b11_init": CYCLE_310_B10_B11_INIT,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append("# Cycle 316 — Item 326: (b10, b11) destructive pair deep dive")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Blocks: b{B10} r={R10}, b{B11} r={R11}; SEED={SEED}; N_STEPS={N_STEPS} per phase; LR={LR}")
lines.append(f"- Cycle 310 joint best (1 seed): {CYCLE_310_B10_B11_JOINT_BEST}")
lines.append("")
lines.append("## M1 — principal angle between Δy_10 and Δy_11 right singular subspaces")
lines.append("")
lines.append("Two Δy variants:")
lines.append("- raw  : Δy_l = GELU(z_l) @ W_proj_l (full FFN residual write)")
lines.append("- curv : Δy_l = (GELU(z_l) − ReLU(z_l)) @ W_proj_l (cycle 302 form)")
lines.append("")
lines.append("| variant | (5×22) mean | (5×22) min | (5×5) mean | (5×5) min | (22×22) mean |")
lines.append("|---|---|---|---|---|---|")
for variant in ["raw", "curv"]:
    a = m1_angles[variant]
    lines.append(
        f"| {variant} | {a['ang_5_22_mean_deg']:.2f}° | {a['ang_5_22_min_deg']:.2f}° | "
        f"{a['ang_5_5_mean_deg']:.2f}° | {a['ang_5_5_min_deg']:.2f}° | {a['ang_22_22_mean_deg']:.2f}° |"
    )
lines.append("")
lines.append(f"- M1 min mean angle (smallest across variants × rank-pairings): **{mean_angle_for_verdict:.2f}°**")
lines.append(f"- Threshold: M1 supported iff <{M1_ANGLE_LT}°; REFUTE-unknown iff >{M1_REFUTE_GT}°")
lines.append("")
lines.append("## M2 — sequential Adam (phase A: b10 only, phase B: b11 only)")
lines.append("")
lines.append(f"- install init PPL (both LowRankCProj, no Adam): {fmt(init_seq)} (cycle 310 init {CYCLE_310_B10_B11_INIT})")
lines.append("")
lines.append("| phase | trainable | init | best | best_step | final | restored |")
lines.append("|---|---|---|---|---|---|---|")
lines.append(f"| A (b10 r=5) | {R10*768+768:,} | {fmt(phase_A['init_ppl'])} | {fmt(phase_A['best_ppl'])} | {phase_A['best_step']} | {fmt(phase_A['final_ppl_at_n_steps'])} | {fmt(phase_A['restored_best_ppl'])} |")
lines.append(f"| B (b11 r=22) | {R11*768+768:,} | {fmt(phase_B['init_ppl'])} | {fmt(phase_B['best_ppl'])} | {phase_B['best_step']} | {fmt(phase_B['final_ppl_at_n_steps'])} | {fmt(phase_B['restored_best_ppl'])} |")
lines.append("")
lines.append(f"- sequential_best_ppl = min(A,B) = **{fmt(sequential_best_ppl)}**")
lines.append(f"- joint best (cycle 310) = {CYCLE_310_B10_B11_JOINT_BEST}")
lines.append(f"- delta vs joint = {sequential_best_ppl - CYCLE_310_B10_B11_JOINT_BEST:+.4f}")
lines.append(f"- Threshold: M3 (joint-Adam saddle) confirmed iff sequential_best_ppl <={M3_PATHOLOGY_LE}; destructive band starts at {DESTRUCTIVE_GE}")
lines.append("")
lines.append("## M3 — one-sided restores")
lines.append("")
lines.append("| ablation | configuration | init | best | best_step | reference |")
lines.append("|---|---|---|---|---|---|")
lines.append(f"| A | b10 float + b11 r=22 fit | {fmt(ablation_A['init_ppl'])} | {fmt(ablation_A['best_ppl'])} | {ablation_A['best_step']} | cycle 270 b11 r=22 ref {CYCLE_270_B11_R22_BEST_MEAN} |")
lines.append(f"| B | b10 r=5 fit + b11 float | {fmt(ablation_B['init_ppl'])} | {fmt(ablation_B['best_ppl'])} | {ablation_B['best_step']} | cycle 282 b10 r=5 diag {CYCLE_282_B10_R5_DIAG} |")
lines.append("")
lines.append("## Verdict logic (item 326 verbatim)")
lines.append("")
lines.append(f"- sequential_best_ppl ≤ {M3_PATHOLOGY_LE}                                  → CONFIRMED-M3-pathology")
lines.append(f"- sequential_best_ppl ≥ {DESTRUCTIVE_GE} AND min mean angle < {M1_ANGLE_LT}°          → CONFIRMED-M1-redundancy")
lines.append(f"- sequential_best_ppl ≥ {DESTRUCTIVE_GE} AND min mean angle > {M1_REFUTE_GT}°          → REFUTED-mechanism-unknown")
lines.append(f"- else                                                                          → INCONCLUSIVE")
lines.append("")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)

print("\nDone.", flush=True)
