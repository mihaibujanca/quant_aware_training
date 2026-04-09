"""
Cycle 334 — Item 328: Extended budget rank-1 retrain at b11 (8000 steps).

Extends cycle 311's b11 rank-1 LASER retrain from 4000 to 8000 steps with LR decay,
to settle whether b11's borderline 59.71 PPL is a plateau or still descending.

LR schedule: Adam LR=1e-4 for steps 0-4000, then 5e-5 for steps 4001-8000.
Seeds: {42, 137, 2024}. Best-state restore.

Verdict:
  best_mean_ppl <= 58.0  → CONFIRMED-robust-r1
  best_mean_ppl >= 60.0  → REFUTED-borderline-artifact
  otherwise              → INCONCLUSIVE
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
BLOCK = 11
RETRAIN_RANK = 1
N_STEPS = 8000
LR_PHASE1 = 1e-4
LR_PHASE2 = 5e-5
LR_SWITCH_STEP = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LOAD_SEED = 42

OUTPUT_PATH = "autoresearch/results/cycle_334_b11_extended.json"
SUMMARY_PATH = "autoresearch/results/cycle_334_summary.md"
MANIFEST_PATH = "autoresearch/results/cycle_334_manifest.txt"

# Cross-validation references from cycle 311
CYCLE_311_B11_INIT_PPL = 60.6103
CYCLE_311_B11_BEST_MEAN = 59.7074
CYCLE_311_B11_PER_SEED_BEST = {42: 59.7154, 137: 59.6968, 2024: 59.7099}
FLOAT_BASELINE_REF = 53.8949

# Verdict thresholds
CONFIRMED_LE = 58.0
REFUTED_GE = 60.0

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)
print(f"Config: block={BLOCK}, rank={RETRAIN_RANK}, N_STEPS={N_STEPS}", flush=True)
print(f"  LR_PHASE1={LR_PHASE1} (steps 1-{LR_SWITCH_STEP}), LR_PHASE2={LR_PHASE2} (steps {LR_SWITCH_STEP+1}-{N_STEPS})", flush=True)

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
    """Replacement for HF Conv1D c_proj at rank r.

    Stores frozen A = U[:, :r] * Sigma[:r] (3072, r) as buffer, trainable B (r, 768)
    as nn.Parameter (initialised from Vh[:r, :]), trainable bias (768,) initialised
    from c_proj.bias. Forward: x @ (A @ B) + b. Matches HF Conv1D semantics
    (input @ weight + bias).
    """

    def __init__(self, A_frozen, B_init, b_init, nf=768):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())
        self.bias = nn.Parameter(b_init.clone().contiguous())
        self.nf = nf

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        W = self.A @ self.B  # (3072, 768) recomputed each forward
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        x = x.view(size_out)
        return x


# ── Float baseline ────────────────────────────────────────────────────────────
print("\nFloat baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
t0 = time.time()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── SVD of b11 c_proj weight ──────────────────────────────────────────────────
print(f"\nComputing SVD of block {BLOCK} c_proj...", flush=True)
W_orig = base_model.transformer.h[BLOCK].mlp.c_proj.weight.data.detach().clone()
b_orig = base_model.transformer.h[BLOCK].mlp.c_proj.bias.data.detach().clone()
assert W_orig.shape == (3072, 768), f"block {BLOCK} c_proj.weight shape {W_orig.shape} != (3072, 768)"
W_cpu = W_orig.float().cpu()
t0 = time.time()
U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
print(f"  SVD in {time.time()-t0:.1f}s, S[0]={float(S[0]):.4f} S[1]={float(S[1]):.4f}", flush=True)
A1 = (U[:, :RETRAIN_RANK] * S[:RETRAIN_RANK]).contiguous()  # (3072, 1)
B1_init = Vh[:RETRAIN_RANK, :].contiguous()                  # (1, 768)
b_init_cpu = b_orig.float().cpu()
S2 = S * S
energy_r1 = float(S2[:RETRAIN_RANK].sum() / S2.sum())
print(f"  energy_r1={energy_r1:.4f}, trainable params per seed: {RETRAIN_RANK*768 + 768}", flush=True)

# ── No-retrain r=1 PPL (sanity) ───────────────────────────────────────────────
print(f"\nNo-retrain r=1 PPL for block {BLOCK}...", flush=True)
W_r1 = (A1 @ B1_init).to(device=W_orig.device, dtype=W_orig.dtype)
with torch.no_grad():
    base_model.transformer.h[BLOCK].mlp.c_proj.weight.data.copy_(W_r1)
t0 = time.time()
init_ppl_noretrain = eval_ppl(base_model, eval_ids)
print(f"  No-retrain r=1 PPL: {init_ppl_noretrain:.4f}  ({time.time()-t0:.1f}s)", flush=True)
delta_init = abs(init_ppl_noretrain - CYCLE_311_B11_INIT_PPL)
print(f"  Cross-check vs cycle 311 ref {CYCLE_311_B11_INIT_PPL}: |Δ|={delta_init:.4f}", flush=True)
if delta_init >= 0.01:
    print(f"  WARNING: init PPL cross-check failed (|Δ|={delta_init:.4f} >= 0.01). Continuing.", flush=True)

# Restore
with torch.no_grad():
    base_model.transformer.h[BLOCK].mlp.c_proj.weight.data.copy_(W_orig)
ppl_after_restore = eval_ppl(base_model, eval_ids)
assert abs(ppl_after_restore - float_ppl) < 1e-3, f"restore failed: {ppl_after_restore} vs {float_ppl}"
print(f"  Restore OK; float PPL = {ppl_after_restore:.4f}", flush=True)

del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()

# ── Retrain at r=1 (3 seeds, 8000 steps, LR decay at 4000) ───────────────────
print(f"\n=== Retrain block {BLOCK} at r={RETRAIN_RANK}, {N_STEPS} steps, LR decay at {LR_SWITCH_STEP} ===", flush=True)
seed_results = []

for seed in SEEDS:
    print(f"\n--- seed {seed} ---", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_s.eval()

    lowrank = LowRankCProj(
        A_frozen=A1.to(device=DEVICE, dtype=torch.float32),
        B_init=B1_init.to(device=DEVICE, dtype=torch.float32),
        b_init=b_init_cpu.to(device=DEVICE, dtype=torch.float32),
        nf=768,
    ).to(DEVICE)
    model_s.transformer.h[BLOCK].mlp.c_proj = lowrank

    init_ppl = eval_ppl(model_s, eval_ids)
    print(f"  init PPL (rank-{RETRAIN_RANK} no retrain): {init_ppl:.4f}", flush=True)
    assert abs(init_ppl - init_ppl_noretrain) < 1e-2, (
        f"init PPL {init_ppl} != noretrain {init_ppl_noretrain}"
    )

    for p in model_s.parameters():
        p.requires_grad_(False)
    lowrank.B.requires_grad_(True)
    lowrank.bias.requires_grad_(True)
    trainable = [lowrank.B, lowrank.bias]
    n_trainable = sum(p.numel() for p in trainable)
    expected_trainable = RETRAIN_RANK * 768 + 768
    assert n_trainable == expected_trainable, f"trainable {n_trainable} != {expected_trainable}"
    print(f"  Trainable params: {n_trainable}", flush=True)

    optimizer = torch.optim.Adam(trainable, lr=LR_PHASE1)

    calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
    random.shuffle(calib_ids)

    best_ppl = init_ppl
    best_step = 0
    best_state = {
        "B": lowrank.B.detach().clone(),
        "bias": lowrank.bias.detach().clone(),
    }
    history = [{"step": 0, "eval_ppl": float(init_ppl)}]

    # Track phase-1 best for cross-validation
    best_ppl_at_4000 = init_ppl
    best_step_at_4000 = 0
    lr_decayed = False

    step = 0
    t_train = time.time()
    while step < N_STEPS:
        for ids in calib_ids:
            if step >= N_STEPS:
                break
            step += 1

            # LR decay at midpoint
            if step == LR_SWITCH_STEP + 1 and not lr_decayed:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR_PHASE2
                lr_decayed = True
                print(f"  LR decayed to {LR_PHASE2} at step {step}", flush=True)

            model_s.train()
            optimizer.zero_grad()
            out = model_s(ids, labels=ids)
            loss = out.loss
            loss.backward()
            optimizer.step()

            if step % EVAL_EVERY == 0:
                ppl_e = eval_ppl(model_s, eval_ids)
                history.append({"step": step, "eval_ppl": float(ppl_e), "calib_loss": float(loss.item())})
                # Track overall best
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    best_state = {
                        "B": lowrank.B.detach().clone(),
                        "bias": lowrank.bias.detach().clone(),
                    }
                # Track phase-1 best
                if step <= LR_SWITCH_STEP and ppl_e < best_ppl_at_4000:
                    best_ppl_at_4000 = ppl_e
                    best_step_at_4000 = step

                if step % 500 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f"    step {step:5d}: eval_ppl={ppl_e:.4f}  best={best_ppl:.4f}@{best_step}  lr={current_lr:.2e}", flush=True)

    # Cross-validation: compare phase-1 best to cycle 311 reference
    cycle311_ref = CYCLE_311_B11_PER_SEED_BEST.get(seed)
    if cycle311_ref is not None:
        delta_xval = abs(best_ppl_at_4000 - cycle311_ref)
        print(f"  Cross-val seed {seed}: best_ppl@4000={best_ppl_at_4000:.4f}, cycle311_ref={cycle311_ref:.4f}, |Δ|={delta_xval:.4f}", flush=True)
        if delta_xval >= 0.05:
            print(f"  WARNING: cross-val mismatch (|Δ|={delta_xval:.4f} >= 0.05). Continuing.", flush=True)

    final_ppl = eval_ppl(model_s, eval_ids)
    with torch.no_grad():
        lowrank.B.data.copy_(best_state["B"])
        lowrank.bias.data.copy_(best_state["bias"])
    restored_ppl = eval_ppl(model_s, eval_ids)
    assert abs(restored_ppl - best_ppl) < 1e-3, (
        f"seed {seed}: restored {restored_ppl} != best {best_ppl}"
    )
    dt = time.time() - t_train
    print(f"  seed {seed}: init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} final@{N_STEPS}={final_ppl:.4f} restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

    # Trajectory last 10 eval points
    last_10 = history[-10:] if len(history) >= 10 else history

    seed_results.append({
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_n_steps": float(final_ppl),
        "restored_best_ppl": float(restored_ppl),
        "best_ppl_at_4000": float(best_ppl_at_4000),
        "best_step_at_4000": int(best_step_at_4000),
        "cycle311_ref": float(cycle311_ref) if cycle311_ref is not None else None,
        "delta_vs_cycle311": float(abs(best_ppl_at_4000 - cycle311_ref)) if cycle311_ref is not None else None,
        "history": history,
        "trajectory_last_10": last_10,
    })

    del model_s, lowrank, optimizer
    if DEVICE == "mps":
        torch.mps.empty_cache()


# ── Aggregate + verdict ───────────────────────────────────────────────────────
best_ppls = [s["best_ppl"] for s in seed_results]
best_mean_ppl = float(sum(best_ppls) / len(best_ppls))
best_std_ppl = math.sqrt(sum((x - best_mean_ppl) ** 2 for x in best_ppls) / len(best_ppls))

best_ppls_at_4000 = [s["best_ppl_at_4000"] for s in seed_results]
best_mean_ppl_at_4000 = float(sum(best_ppls_at_4000) / len(best_ppls_at_4000))

if best_mean_ppl <= CONFIRMED_LE:
    verdict = "CONFIRMED-robust-r1"
elif best_mean_ppl >= REFUTED_GE:
    verdict = "REFUTED-borderline-artifact"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== Verdict ===", flush=True)
for s in seed_results:
    print(f"  seed {s['seed']}: best={s['best_ppl']:.4f}@step{s['best_step']}  best@4000={s['best_ppl_at_4000']:.4f}@step{s['best_step_at_4000']}", flush=True)
print(f"  best_mean_ppl (full 8000): {best_mean_ppl:.4f} ± {best_std_ppl:.4f}", flush=True)
print(f"  best_mean_ppl_at_4000: {best_mean_ppl_at_4000:.4f} (cycle 311 ref: {CYCLE_311_B11_BEST_MEAN})", flush=True)
print(f"  VERDICT: {verdict}", flush=True)
print(f"  (CONFIRMED ≤ {CONFIRMED_LE}, REFUTED ≥ {REFUTED_GE})", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 334,
    "item": 328,
    "verdict": verdict,
    "best_mean_ppl": best_mean_ppl,
    "best_std_ppl": best_std_ppl,
    "best_mean_ppl_at_4000": best_mean_ppl_at_4000,
    "float_ppl": float(float_ppl),
    "init_ppl_noretrain": float(init_ppl_noretrain),
    "per_seed": {str(s["seed"]): s for s in seed_results},
    "cross_validation": {
        str(s["seed"]): {
            "cycle311_ref": s["cycle311_ref"],
            "measured_at_4000": s["best_ppl_at_4000"],
            "delta": s["delta_vs_cycle311"],
        }
        for s in seed_results
    },
    "verdict_logic": {
        "confirmed_le": CONFIRMED_LE,
        "refuted_ge": REFUTED_GE,
        "best_mean_ppl": best_mean_ppl,
        "verdict": verdict,
    },
    "config": {
        "BLOCK": BLOCK,
        "RETRAIN_RANK": RETRAIN_RANK,
        "N_STEPS": N_STEPS,
        "LR_PHASE1": LR_PHASE1,
        "LR_PHASE2": LR_PHASE2,
        "LR_SWITCH_STEP": LR_SWITCH_STEP,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
        "trainable_params_per_fit": RETRAIN_RANK * 768 + 768,
    },
    "references": {
        "cycle_311_b11_init_ppl": CYCLE_311_B11_INIT_PPL,
        "cycle_311_b11_best_mean": CYCLE_311_B11_BEST_MEAN,
        "cycle_311_b11_per_seed_best": CYCLE_311_B11_PER_SEED_BEST,
        "float_baseline_ref": FLOAT_BASELINE_REF,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
lines = []
lines.append(f"# Cycle 334 — Item 328: Extended budget rank-1 retrain at b11 (8000 steps)")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Block: {BLOCK}, rank: {RETRAIN_RANK}, steps: {N_STEPS}")
lines.append(f"- LR schedule: {LR_PHASE1} (steps 1-{LR_SWITCH_STEP}) → {LR_PHASE2} (steps {LR_SWITCH_STEP+1}-{N_STEPS})")
lines.append(f"- Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Init PPL (no-retrain r=1): {init_ppl_noretrain:.4f} (cycle 311 ref: {CYCLE_311_B11_INIT_PPL})")
lines.append(f"- Trainable: {RETRAIN_RANK*768 + 768} params (B={RETRAIN_RANK}×768 + bias)")
lines.append(f"- Seeds: {SEEDS}, best-state restore")
lines.append("")
lines.append("## Per-seed results")
lines.append("")
lines.append("| seed | init | best | best_step | best@4000 | Δ vs c311 | final@8000 |")
lines.append("|---|---|---|---|---|---|---|")
for s in seed_results:
    delta_str = f"{s['delta_vs_cycle311']:+.4f}" if s["delta_vs_cycle311"] is not None else "N/A"
    lines.append(f"| {s['seed']} | {s['init_ppl']:.4f} | {s['best_ppl']:.4f} | {s['best_step']} | {s['best_ppl_at_4000']:.4f} | {delta_str} | {s['final_ppl_at_n_steps']:.4f} |")
lines.append("")
lines.append("## Aggregate")
lines.append("")
lines.append(f"- best_mean_ppl (8000 steps): **{best_mean_ppl:.4f}** ± {best_std_ppl:.4f}")
lines.append(f"- best_mean_ppl_at_4000: {best_mean_ppl_at_4000:.4f} (cycle 311 ref: {CYCLE_311_B11_BEST_MEAN})")
lines.append("")
lines.append("## Verdict logic")
lines.append("")
lines.append(f"- CONFIRMED-robust-r1 iff best_mean_ppl ≤ {CONFIRMED_LE}")
lines.append(f"- REFUTED-borderline-artifact iff best_mean_ppl ≥ {REFUTED_GE}")
lines.append(f"- best_mean_ppl = {best_mean_ppl:.4f} → **{verdict}**")
lines.append("")
lines.append("## Last 10 eval points per seed (plateau analysis)")
lines.append("")
for s in seed_results:
    lines.append(f"### seed {s['seed']}")
    lines.append("| step | eval_ppl |")
    lines.append("|---|---|")
    for pt in s["trajectory_last_10"]:
        lines.append(f"| {pt['step']} | {pt['eval_ppl']:.4f} |")
    lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)


# ── Manifest ──────────────────────────────────────────────────────────────────
manifest_lines = [
    f"cycle: 334",
    f"item: 328",
    f"verdict: {verdict}",
    f"best_mean_ppl: {best_mean_ppl:.4f}",
    f"best_mean_ppl_at_4000: {best_mean_ppl_at_4000:.4f}",
    f"cycle_311_b11_best_mean_ref: {CYCLE_311_B11_BEST_MEAN}",
    f"float_ppl: {float_ppl:.4f}",
    f"init_ppl_noretrain_r1: {init_ppl_noretrain:.4f}",
    f"seeds: {SEEDS}",
    f"per_seed_best: " + ", ".join(f"s{s['seed']}={s['best_ppl']:.4f}@{s['best_step']}" for s in seed_results),
    f"per_seed_best_at_4000: " + ", ".join(f"s{s['seed']}={s['best_ppl_at_4000']:.4f}@{s['best_step_at_4000']}" for s in seed_results),
    f"outputs: {OUTPUT_PATH}, {SUMMARY_PATH}, {MANIFEST_PATH}",
    f"script: experiments/b11_extended_budget_cycle334.py",
]
with open(MANIFEST_PATH, "w") as f:
    f.write("\n".join(manifest_lines) + "\n")
print(f"Wrote {MANIFEST_PATH}", flush=True)
print(f"\nDone. Verdict: {verdict}  best_mean_ppl={best_mean_ppl:.4f}", flush=True)
