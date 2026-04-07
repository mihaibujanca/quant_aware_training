"""
Cycle 292 — Item 302: Block 1 c_proj rank-1 weight-SVD sub-eff_rank stress test.

Adapts cycle 280 (block 1 rank-2) verbatim but at rank-1 (r < eff_rank=1.94).
Re-verify no-retrain r=1 (cycle 280 reported 59.75), then retrain r=1 joint Adam
over B (1, 768) + bias (768) = 1,536 trainable params, 3 seeds, best-state restore.

HF Conv1D layout: c_proj.weight is (3072, 768); applied as x @ weight + bias.
SVD: U (3072, 768), Σ (768,), Vh (768, 768). Rank-r approx:
W_r = (U[:, :r] * Σ[:r]) @ Vh[:r, :].

Verdict (item 302 verbatim):
  CONFIRMED iff rank-1 retrained best_mean ≤ 51.0 (still beats float by ≥2.9 PPL
              at 1,536 params — new lowest-param Pareto anchor).
  REFUTED   iff rank-1 retrained best_mean ≥ 53.5 (rank-1 doesn't beat float
              meaningfully; r=2 is the floor for block 1).
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
RANKS_NORETRAIN = [1]
RETRAIN_RANK = 1
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 4000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42  # for the initial fp baseline / SVD reproducibility
OUTPUT_PATH = "autoresearch/results/cycle_292_block1_rank1.json"
SUMMARY_PATH = "autoresearch/results/cycle_292_summary.md"

# Verdict thresholds (item 302 verbatim)
CONFIRMED_RETRAIN_LE = 51.0       # r=1 retrain best_mean
REFUTED_RETRAIN_GE = 53.5         # r=1 retrain best_mean

FLOAT_BASELINE_REF = 53.895  # cycle 266/268/269/270/280/281/282/285/288/289 reference
CYCLE_278_EFF_RANK_B1 = 1.9448      # cycle 278: block 1 c_proj OUTPUT eff_rank
CYCLE_278_TOP1_ENERGY_B1 = 0.9294   # cycle 278: block 1 top-1 energy
CYCLE_280_B1_R1_NORETRAIN = 59.7513 # cycle 280 reported r=1 no-retrain PPL
CYCLE_280_B1_R2_BEST_MEAN = 49.0324 # cycle 280: block 1 r=2 retrain best_mean (2,304 params)
CYCLE_289_B10_R5_BEST_MEAN = 51.4659 # cycle 289: block 10 r=5 retrain best_mean (4,608 params)
CYCLE_285_B2_R2_BEST_MEAN = 49.3290  # cycle 285: block 2 r=2 retrain best_mean (2,304 params)
CYCLE_282_B10_R17_BEST_MEAN = 50.0445 # cycle 282: block 10 r=17 retrain best_mean (13,824 params)
CYCLE_270_B11_R22_BEST_MEAN = 54.4687 # cycle 270: block 11 r=22 retrain best_mean (17,664 params)
CYCLE_288_B3_R24_BEST_MEAN = 50.5935  # cycle 288: block 3 r=24 retrain best_mean (19,200 params)

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
    """Replacement for HF Conv1D c_proj at rank r.

    Stores frozen A = U[:, :r] * Σ[:r] (3072, r) as buffer, trainable B (r, 768)
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
        W = self.A @ self.B  # (3072, 768) — recomputed each forward
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), W)
        x = x.view(size_out)
        return x


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

# ── SVD of c_proj_1.weight ────────────────────────────────────────────────────
print(f"\nSVD of block {TARGET_BLOCK} c_proj.weight...", flush=True)
c_proj_orig_weight = model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.detach().clone()
c_proj_orig_bias = model.transformer.h[TARGET_BLOCK].mlp.c_proj.bias.data.detach().clone()
assert c_proj_orig_weight.shape == (3072, 768), (
    f"c_proj.weight shape {c_proj_orig_weight.shape} != (3072, 768)"
)
W = c_proj_orig_weight.float().cpu()
t0 = time.time()
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
print(f"  SVD done in {time.time()-t0:.1f}s; U {tuple(U.shape)}, S {tuple(S.shape)}, Vh {tuple(Vh.shape)}", flush=True)
print(f"  Top-5 singular values: {S[:5].tolist()}", flush=True)
print(f"  S[0]: {float(S[0]):.4f}, S[1]: {float(S[1]):.4f}, S[4]: {float(S[4]):.4f}", flush=True)
print(f"  S[-1]: {float(S[-1]):.6f}", flush=True)

# Energy fractions for context
S2 = S * S
total_energy = float(S2.sum())
energy_at = {}
for r in RANKS_NORETRAIN:
    energy_at[r] = float(S2[:r].sum() / total_energy)
print(f"  Energy fractions: " + ", ".join(f"r={r}: {energy_at[r]:.4f}" for r in RANKS_NORETRAIN), flush=True)


# ── No-retrain rank-r sweep ───────────────────────────────────────────────────
print("\n=== No-retrain rank-r sweep ===", flush=True)
device_dtype = c_proj_orig_weight.dtype
device_target = c_proj_orig_weight.device

noretrain_results = {}
for r in RANKS_NORETRAIN:
    A_r = (U[:, :r] * S[:r]).contiguous()  # (3072, r)
    B_r = Vh[:r, :].contiguous()  # (r, 768)
    W_r = (A_r @ B_r).to(device=device_target, dtype=device_dtype)
    assert W_r.shape == c_proj_orig_weight.shape, f"shape mismatch at r={r}"
    with torch.no_grad():
        model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(W_r)
    t0 = time.time()
    ppl_r = eval_ppl(model, eval_ids)
    dt = time.time() - t0
    delta_float = ppl_r - float_ppl
    pct = (ppl_r / float_ppl - 1) * 100
    print(f"  r={r:3d}: PPL={ppl_r:.3f}  (Δfloat={delta_float:+.3f}, {pct:+.2f}%)  [{dt:.1f}s, energy={energy_at[r]:.4f}]", flush=True)
    noretrain_results[r] = {
        "ppl": float(ppl_r),
        "delta_float": float(delta_float),
        "pct_above_float": float(pct),
        "energy_fraction": float(energy_at[r]),
    }

with torch.no_grad():
    model.transformer.h[TARGET_BLOCK].mlp.c_proj.weight.data.copy_(c_proj_orig_weight)
ppl_restored = eval_ppl(model, eval_ids)
print(f"  Restored original c_proj.weight, PPL={ppl_restored:.3f} (should match float {float_ppl:.3f})", flush=True)
assert abs(ppl_restored - float_ppl) < 1e-3, (
    f"restore failed: {ppl_restored} vs {float_ppl}"
)

# Cross-check against cycle 280 reference for r=1 no-retrain
cycle_280_delta = abs(noretrain_results[1]["ppl"] - CYCLE_280_B1_R1_NORETRAIN)
print(f"  Cross-check cycle 280 r=1 no-retrain {CYCLE_280_B1_R1_NORETRAIN}: |Δ|={cycle_280_delta:.4f}", flush=True)
assert cycle_280_delta < 0.01, (
    f"noretrain r=1 cross-check vs cycle 280 failed: {noretrain_results[1]['ppl']} vs {CYCLE_280_B1_R1_NORETRAIN}"
)


# ── Retrain at r=1 (3 seeds) ──────────────────────────────────────────────────
print(f"\n=== Retrain at r={RETRAIN_RANK} ({len(SEEDS)} seeds) ===", flush=True)
A1 = (U[:, :RETRAIN_RANK] * S[:RETRAIN_RANK]).contiguous()  # (3072, 1)
B1_init = Vh[:RETRAIN_RANK, :].contiguous()  # (1, 768)

retrain_seeds = []
for seed in SEEDS:
    print(f"\n--- seed {seed} ---", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_s.eval()

    lowrank = LowRankCProj(
        A_frozen=A1.to(device=DEVICE, dtype=torch.float32),
        B_init=B1_init.to(device=DEVICE, dtype=torch.float32),
        b_init=c_proj_orig_bias.to(device=DEVICE, dtype=torch.float32),
        nf=768,
    ).to(DEVICE)
    model_s.transformer.h[TARGET_BLOCK].mlp.c_proj = lowrank

    init_ppl = eval_ppl(model_s, eval_ids)
    print(f"  init PPL (rank-{RETRAIN_RANK} no retrain): {init_ppl:.3f}", flush=True)
    assert abs(init_ppl - noretrain_results[RETRAIN_RANK]["ppl"]) < 1e-2, (
        f"init PPL {init_ppl} != noretrain rank-{RETRAIN_RANK} {noretrain_results[RETRAIN_RANK]['ppl']}"
    )

    for p in model_s.parameters():
        p.requires_grad_(False)
    lowrank.B.requires_grad_(True)
    lowrank.bias.requires_grad_(True)
    trainable = [lowrank.B, lowrank.bias]
    n_trainable = sum(p.numel() for p in trainable)
    expected_trainable = RETRAIN_RANK * 768 + 768
    print(f"  Trainable params: {n_trainable:,} (expect {expected_trainable:,} = {RETRAIN_RANK}*768 + 768)", flush=True)
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
        f"seed {seed}: restored {restored_ppl} != best {best_ppl}"
    )
    dt = time.time() - t_train
    print(f"  seed {seed}: init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} final@{N_STEPS}={final_ppl:.4f} restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

    retrain_seeds.append({
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

# Aggregate retrain
best_ppls = [s["best_ppl"] for s in retrain_seeds]
best_mean = float(sum(best_ppls) / len(best_ppls))
best_var = float(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
best_std = math.sqrt(best_var)
print(f"\nRetrain at r={RETRAIN_RANK}: best_mean={best_mean:.4f} ± {best_std:.4f}  (n={len(SEEDS)})", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
if best_mean <= CONFIRMED_RETRAIN_LE:
    verdict = "CONFIRMED"
elif best_mean >= REFUTED_RETRAIN_GE:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== Verdict ===", flush=True)
print(f"  noretrain PPL: " + ", ".join(f"r={r}: {noretrain_results[r]['ppl']:.3f}" for r in RANKS_NORETRAIN), flush=True)
print(f"  r={RETRAIN_RANK} retrain best_mean = {best_mean:.4f}  (CONFIRMED ≤ {CONFIRMED_RETRAIN_LE}, REFUTED ≥ {REFUTED_RETRAIN_GE})", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 292,
    "item": 302,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "noretrain": {str(r): noretrain_results[r] for r in RANKS_NORETRAIN},
    "retrain_rank": RETRAIN_RANK,
    "retrain_seeds": retrain_seeds,
    "retrain_aggregate": {
        "best_mean": best_mean,
        "best_std": best_std,
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
        "n_seeds": len(SEEDS),
    },
    "verdict_logic": {
        "noretrain_ppls": {str(r): float(noretrain_results[r]["ppl"]) for r in RANKS_NORETRAIN},
        "retrain_best_mean": best_mean,
        "threshold_confirmed_retrain_le": CONFIRMED_RETRAIN_LE,
        "threshold_refuted_retrain_ge": REFUTED_RETRAIN_GE,
        "verdict": verdict,
    },
    "config": {
        "TARGET_BLOCK": TARGET_BLOCK,
        "RANKS_NORETRAIN": RANKS_NORETRAIN,
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
    "svd_summary": {
        "top_5_sv": S[:5].tolist(),
        "S_at_0": float(S[0]),
        "S_at_1": float(S[1]),
        "S_at_4": float(S[4]),
        "S_min": float(S[-1]),
        "energy_fractions": {str(r): energy_at[r] for r in RANKS_NORETRAIN},
        "cycle_278_eff_rank_b1": CYCLE_278_EFF_RANK_B1,
        "cycle_278_top1_energy_b1": CYCLE_278_TOP1_ENERGY_B1,
    },
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "cycle_278_eff_rank_b1": CYCLE_278_EFF_RANK_B1,
        "cycle_278_top1_energy_b1": CYCLE_278_TOP1_ENERGY_B1,
        "cycle_280_b1_r1_noretrain_ref": CYCLE_280_B1_R1_NORETRAIN,
        "cycle_280_b1_r2_retrain_best_mean": CYCLE_280_B1_R2_BEST_MEAN,
        "cycle_289_b10_r5_retrain_best_mean": CYCLE_289_B10_R5_BEST_MEAN,
        "cycle_285_b2_r2_retrain_best_mean": CYCLE_285_B2_R2_BEST_MEAN,
        "cycle_282_b10_r17_retrain_best_mean": CYCLE_282_B10_R17_BEST_MEAN,
        "cycle_270_b11_r22_retrain_best_mean": CYCLE_270_B11_R22_BEST_MEAN,
        "cycle_288_b3_r24_retrain_best_mean": CYCLE_288_B3_R24_BEST_MEAN,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append(f"# Cycle 292 — Item 302: Block 1 rank-1 c_proj sub-eff_rank stress test")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Block: {TARGET_BLOCK}, c_proj.weight shape (3072, 768), HF Conv1D")
lines.append(f"- Cycle 278 reference: eff_rank(c_proj_{TARGET_BLOCK} OUTPUT) = {CYCLE_278_EFF_RANK_B1} (top-1 energy {CYCLE_278_TOP1_ENERGY_B1})")
lines.append(f"- Sub-eff_rank test: r={RETRAIN_RANK} < eff_rank {CYCLE_278_EFF_RANK_B1} (r/eff_rank = {RETRAIN_RANK/CYCLE_278_EFF_RANK_B1:.3f})")
lines.append("")
lines.append("## No-retrain rank-r sweep")
lines.append("")
lines.append("| r | PPL | Δfloat | %above | top-r energy |")
lines.append("|---|---|---|---|---|")
for r in RANKS_NORETRAIN:
    nr = noretrain_results[r]
    lines.append(f"| {r} | {fmt(nr['ppl'])} | {nr['delta_float']:+.3f} | {nr['pct_above_float']:+.2f}% | {fmt(nr['energy_fraction'])} |")
lines.append(f"- Cycle 280 reported r=1 no-retrain = {CYCLE_280_B1_R1_NORETRAIN}, measured here = {noretrain_results[1]['ppl']:.4f}, |Δ| = {cycle_280_delta:.4f}")
lines.append("")
lines.append(f"## Retrain at r={RETRAIN_RANK} (3 seeds, Adam LR={LR}, {N_STEPS} steps)")
lines.append("")
lines.append(f"Trainable params: {RETRAIN_RANK * 768 + 768:,} = {RETRAIN_RANK}×768 (B) + 768 (bias). A=(U[:, :{RETRAIN_RANK}]·Σ[:{RETRAIN_RANK}]) frozen.")
lines.append("")
lines.append("| seed | init | best | best_step | final@N | restored |")
lines.append("|---|---|---|---|---|---|")
for s in retrain_seeds:
    lines.append(f"| {s['seed']} | {fmt(s['init_ppl'])} | {fmt(s['best_ppl'])} | {s['best_step']} | {fmt(s['final_ppl_at_n_steps'])} | {fmt(s['restored_best_ppl'])} |")
lines.append(f"| **mean** | — | **{fmt(best_mean)}** | — | — | — |")
lines.append(f"| **std**  | — | {fmt(best_std)} | — | — | — |")
lines.append("")
lines.append("## Verdict logic (item 302 verbatim)")
lines.append("")
lines.append(f"- r={RETRAIN_RANK} retrain best_mean = {fmt(best_mean)}")
lines.append(f"- CONFIRMED iff ≤ {CONFIRMED_RETRAIN_LE}")
lines.append(f"- REFUTED iff ≥ {REFUTED_RETRAIN_GE}")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append("## Comparisons")
lines.append("")
lines.append(f"- Δ vs float: noretrain r={RETRAIN_RANK} = {noretrain_results[RETRAIN_RANK]['delta_float']:+.3f}; retrain best_mean = {best_mean - float_ppl:+.3f}")
lines.append(f"- Δ retrain vs noretrain at r={RETRAIN_RANK}: {best_mean - noretrain_results[RETRAIN_RANK]['ppl']:+.3f} (Adam recovery)")
lines.append(f"- Δ vs cycle 280 block 1 r=2 ({CYCLE_280_B1_R2_BEST_MEAN}, 2,304 params): {best_mean - CYCLE_280_B1_R2_BEST_MEAN:+.3f} PPL (2× more params)")
lines.append(f"- Δ vs cycle 289 block 10 r=5 ({CYCLE_289_B10_R5_BEST_MEAN}, 4,608 params): {best_mean - CYCLE_289_B10_R5_BEST_MEAN:+.3f} PPL (3× more params)")
lines.append(f"- Δ vs cycle 285 block 2 r=2 ({CYCLE_285_B2_R2_BEST_MEAN}, 2,304 params): {best_mean - CYCLE_285_B2_R2_BEST_MEAN:+.3f} PPL")
lines.append(f"- Δ vs cycle 282 block 10 r=17 ({CYCLE_282_B10_R17_BEST_MEAN}, 13,824 params): {best_mean - CYCLE_282_B10_R17_BEST_MEAN:+.3f} PPL")
lines.append(f"- Δ vs cycle 270 block 11 r=22 ({CYCLE_270_B11_R22_BEST_MEAN}, 17,664 params): {best_mean - CYCLE_270_B11_R22_BEST_MEAN:+.3f} PPL")
lines.append(f"- Δ vs cycle 288 block 3 r=24 ({CYCLE_288_B3_R24_BEST_MEAN}, 19,200 params): {best_mean - CYCLE_288_B3_R24_BEST_MEAN:+.3f} PPL")
lines.append(f"- Trainable param ratio vs cycle 280 (2,304): {(RETRAIN_RANK*768+768) / 2304:.4f} (cycle 292 = {RETRAIN_RANK*768+768:,})")
lines.append(f"- Trainable param ratio vs cycle 289 (4,608): {(RETRAIN_RANK*768+768) / 4608:.4f}")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
