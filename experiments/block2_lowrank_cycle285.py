"""
Cycle 285 — Item 295: Targeted block 2 low-rank c_proj compression.

For r in {1, 2, 5, 10}: replace c_proj_2.weight with rank-r SVD approx,
measure PPL on the standard 100-doc eval slice. Then for r=2: rerun with
joint Adam retrain of B (2, 768) + bias (768) = 2,304 trainable params,
3 seeds, best-state restore.

HF Conv1D layout: c_proj.weight is (3072, 768); applied as x @ weight + bias.
SVD: U (3072, 768), Σ (768,), Vh (768, 768). Rank-r approx:
W_r = (U[:, :r] * Σ[:r]) @ Vh[:r, :].

Verdict (item 295 verbatim):
  CONFIRMED iff (r=2 retrain best_mean ≤ 50.0) OR (r=1 retrain best_mean ≤ 51.0).
  REFUTED   iff r=2 retrain best_mean ≥ 53.0.
  INCONCLUSIVE otherwise.

Block 2 has the LOWEST eff_rank in the model (cycle 278: 1.0503, top-1 energy 0.9961).
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
TARGET_BLOCK = 2
RANKS_NORETRAIN = [1, 2, 5, 10]
RETRAIN_RANK = 2
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 4000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42  # for the initial fp baseline / SVD reproducibility
OUTPUT_PATH = "autoresearch/results/cycle_285_block2_lowrank.json"
SUMMARY_PATH = "autoresearch/results/cycle_285_summary.md"

# Optional second retrain pass at r=1, only if no-retrain r=1 is in striking distance.
R1_GATE_PPL = 55.0  # if noretrain[r=1].ppl <= 55.0, also retrain at r=1

# Verdict thresholds (item 295 verbatim)
CONFIRMED_R2_RETRAIN_LE = 50.0      # r=2 retrain best_mean
CONFIRMED_R1_RETRAIN_LE = 51.0      # r=1 retrain best_mean (only if gate fires)
REFUTED_R2_RETRAIN_GE = 53.0        # r=2 retrain best_mean

FLOAT_BASELINE_REF = 53.895  # cycle 266/268/269/270 reference
CYCLE_278_EFF_RANK_B2 = 1.0503  # cycle 278: block 2 c_proj OUTPUT eff_rank (model min)
CYCLE_278_TOP1_ENERGY_B2 = 0.9961  # cycle 278: top-1 energy fraction at block 2
CYCLE_280_B1_R2_BEST_MEAN = 49.0324  # cycle 280 retrain best_mean (2,304 params)
CYCLE_270_B11_R22_BEST_MEAN = 54.4687  # cycle 270 retrain best_mean (17,664 params)
CYCLE_282_B10_R17_BEST_MEAN = 50.0445  # cycle 282 retrain best_mean (13,824 params)
CYCLE_283_B2_K16_BEST_MEAN = 49.4621  # cycle 283 [2] K=16 retrain (13,056 params)

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

# ── SVD of c_proj_2.weight ────────────────────────────────────────────────────
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
print(f"  S[0], S[1], S[4], S[9]: {float(S[0]):.4f}, {float(S[1]):.4f}, {float(S[4]):.4f}, {float(S[9]):.4f}", flush=True)
print(f"  S[-1]: {float(S[-1]):.6f}", flush=True)

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


# ── Retrain helper ────────────────────────────────────────────────────────────
def retrain_at_rank(rank_r):
    """Retrain LowRankCProj at given rank with 3 seeds. Returns list of seed dicts."""
    print(f"\n=== Retrain at r={rank_r} ({len(SEEDS)} seeds) ===", flush=True)
    A_r = (U[:, :rank_r] * S[:rank_r]).contiguous()  # (3072, r)
    B_r_init = Vh[:rank_r, :].contiguous()  # (r, 768)

    seeds_out = []
    for seed in SEEDS:
        print(f"\n--- seed {seed} (r={rank_r}) ---", flush=True)
        torch.manual_seed(seed)
        random.seed(seed)

        model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model_s.eval()

        lowrank = LowRankCProj(
            A_frozen=A_r.to(device=DEVICE, dtype=torch.float32),
            B_init=B_r_init.to(device=DEVICE, dtype=torch.float32),
            b_init=c_proj_orig_bias.to(device=DEVICE, dtype=torch.float32),
            nf=768,
        ).to(DEVICE)
        model_s.transformer.h[TARGET_BLOCK].mlp.c_proj = lowrank

        init_ppl = eval_ppl(model_s, eval_ids)
        print(f"  init PPL (rank-{rank_r} no retrain): {init_ppl:.3f}", flush=True)
        assert abs(init_ppl - noretrain_results[rank_r]["ppl"]) < 1e-2, (
            f"init PPL {init_ppl} != noretrain rank-{rank_r} {noretrain_results[rank_r]['ppl']}"
        )

        for p in model_s.parameters():
            p.requires_grad_(False)
        lowrank.B.requires_grad_(True)
        lowrank.bias.requires_grad_(True)
        trainable = [lowrank.B, lowrank.bias]
        n_trainable = sum(p.numel() for p in trainable)
        expected_trainable = rank_r * 768 + 768
        print(f"  Trainable params: {n_trainable:,} (expect {expected_trainable:,} = {rank_r}*768 + 768)", flush=True)
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
        print(f"  seed {seed} (r={rank_r}): init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} final@{N_STEPS}={final_ppl:.4f} restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

        seeds_out.append({
            "seed": seed,
            "rank": rank_r,
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
    return seeds_out


def aggregate(seeds_out):
    bps = [s["best_ppl"] for s in seeds_out]
    mean = float(sum(bps) / len(bps))
    var = float(sum((x - mean) ** 2 for x in bps) / len(bps))
    return {
        "best_mean": mean,
        "best_std": math.sqrt(var),
        "best_min": float(min(bps)),
        "best_max": float(max(bps)),
        "n_seeds": len(seeds_out),
    }


# ── Retrain at r=2 (always) ───────────────────────────────────────────────────
retrain_r2_seeds = retrain_at_rank(RETRAIN_RANK)
agg_r2 = aggregate(retrain_r2_seeds)
print(f"\nRetrain at r={RETRAIN_RANK}: best_mean={agg_r2['best_mean']:.4f} ± {agg_r2['best_std']:.4f}  (n={len(SEEDS)})", flush=True)

# ── Optional retrain at r=1 (gated) ───────────────────────────────────────────
r1_gate_fired = noretrain_results[1]["ppl"] <= R1_GATE_PPL
print(f"\nr=1 gate: noretrain r=1 PPL={noretrain_results[1]['ppl']:.3f} <= {R1_GATE_PPL}? {r1_gate_fired}", flush=True)

retrain_r1_seeds = None
agg_r1 = None
if r1_gate_fired:
    retrain_r1_seeds = retrain_at_rank(1)
    agg_r1 = aggregate(retrain_r1_seeds)
    print(f"\nRetrain at r=1: best_mean={agg_r1['best_mean']:.4f} ± {agg_r1['best_std']:.4f}  (n={len(SEEDS)})", flush=True)
else:
    print("  Skipping r=1 retrain (gate did not fire).", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
r2_confirmed = agg_r2["best_mean"] <= CONFIRMED_R2_RETRAIN_LE
r1_confirmed = (agg_r1 is not None) and (agg_r1["best_mean"] <= CONFIRMED_R1_RETRAIN_LE)
r2_refuted = agg_r2["best_mean"] >= REFUTED_R2_RETRAIN_GE

if r2_confirmed or r1_confirmed:
    verdict = "CONFIRMED"
elif r2_refuted:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== Verdict ===", flush=True)
print(f"  noretrain PPLs: " + ", ".join(f"r={r}: {noretrain_results[r]['ppl']:.3f}" for r in RANKS_NORETRAIN), flush=True)
print(f"  r={RETRAIN_RANK} retrain best_mean = {agg_r2['best_mean']:.4f}  (CONFIRMED ≤ {CONFIRMED_R2_RETRAIN_LE}, REFUTED ≥ {REFUTED_R2_RETRAIN_GE})", flush=True)
if agg_r1 is not None:
    print(f"  r=1 retrain best_mean = {agg_r1['best_mean']:.4f}  (CONFIRMED ≤ {CONFIRMED_R1_RETRAIN_LE})", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 285,
    "item": 295,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "noretrain": {str(r): noretrain_results[r] for r in RANKS_NORETRAIN},
    "retrain_rank": RETRAIN_RANK,
    "retrain_r2_seeds": retrain_r2_seeds,
    "retrain_r2_aggregate": agg_r2,
    "r1_gate_fired": bool(r1_gate_fired),
    "r1_gate_threshold": R1_GATE_PPL,
    "retrain_r1_seeds": retrain_r1_seeds,
    "retrain_r1_aggregate": agg_r1,
    "verdict_logic": {
        "noretrain_ppls": {str(r): float(noretrain_results[r]["ppl"]) for r in RANKS_NORETRAIN},
        "retrain_r2_best_mean": agg_r2["best_mean"],
        "retrain_r1_best_mean": (agg_r1["best_mean"] if agg_r1 is not None else None),
        "r2_confirmed_clause": bool(r2_confirmed),
        "r1_confirmed_clause": bool(r1_confirmed),
        "r2_refuted_clause": bool(r2_refuted),
        "threshold_confirmed_r2_retrain_le": CONFIRMED_R2_RETRAIN_LE,
        "threshold_confirmed_r1_retrain_le": CONFIRMED_R1_RETRAIN_LE,
        "threshold_refuted_r2_retrain_ge": REFUTED_R2_RETRAIN_GE,
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
        "trainable_params_per_fit_r2": RETRAIN_RANK * 768 + 768,
        "trainable_params_per_fit_r1": 1 * 768 + 768,
    },
    "svd_summary": {
        "top_5_sv": S[:5].tolist(),
        "S_at_0": float(S[0]),
        "S_at_1": float(S[1]),
        "S_at_4": float(S[4]),
        "S_at_9": float(S[9]),
        "S_min": float(S[-1]),
        "energy_fractions": {str(r): energy_at[r] for r in RANKS_NORETRAIN},
        "cycle_278_eff_rank_b2": CYCLE_278_EFF_RANK_B2,
        "cycle_278_top1_energy_b2": CYCLE_278_TOP1_ENERGY_B2,
    },
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "cycle_280_block1_r2_retrain_best_mean": CYCLE_280_B1_R2_BEST_MEAN,
        "cycle_270_block11_r22_retrain_best_mean": CYCLE_270_B11_R22_BEST_MEAN,
        "cycle_282_block10_r17_retrain_best_mean": CYCLE_282_B10_R17_BEST_MEAN,
        "cycle_283_block2_K16_retrain_best_mean": CYCLE_283_B2_K16_BEST_MEAN,
        "cycle_278_eff_rank_b2": CYCLE_278_EFF_RANK_B2,
        "cycle_278_top1_energy_b2": CYCLE_278_TOP1_ENERGY_B2,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append(f"# Cycle 285 — Item 295: Targeted block 2 low-rank c_proj compression")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Block: {TARGET_BLOCK}, c_proj.weight shape (3072, 768), HF Conv1D")
lines.append(f"- Cycle 278 reference: eff_rank(c_proj_{TARGET_BLOCK} OUTPUT) = {CYCLE_278_EFF_RANK_B2} (top-1 energy {CYCLE_278_TOP1_ENERGY_B2}) — model minimum")
lines.append("")
lines.append("## No-retrain rank-r sweep")
lines.append("")
lines.append("| r | PPL | Δfloat | %above | top-r energy |")
lines.append("|---|---|---|---|---|")
for r in RANKS_NORETRAIN:
    nr = noretrain_results[r]
    lines.append(f"| {r} | {fmt(nr['ppl'])} | {nr['delta_float']:+.3f} | {nr['pct_above_float']:+.2f}% | {fmt(nr['energy_fraction'])} |")
lines.append("")
lines.append(f"## Retrain at r={RETRAIN_RANK} (3 seeds, Adam LR={LR}, {N_STEPS} steps)")
lines.append("")
lines.append(f"Trainable params: {RETRAIN_RANK * 768 + 768:,} = {RETRAIN_RANK}×768 (B) + 768 (bias). A=(U[:, :{RETRAIN_RANK}]·Σ[:{RETRAIN_RANK}]) frozen.")
lines.append("")
lines.append("| seed | init | best | best_step | final@N | restored |")
lines.append("|---|---|---|---|---|---|")
for s in retrain_r2_seeds:
    lines.append(f"| {s['seed']} | {fmt(s['init_ppl'])} | {fmt(s['best_ppl'])} | {s['best_step']} | {fmt(s['final_ppl_at_n_steps'])} | {fmt(s['restored_best_ppl'])} |")
lines.append(f"| **mean** | — | **{fmt(agg_r2['best_mean'])}** | — | — | — |")
lines.append(f"| **std**  | — | {fmt(agg_r2['best_std'])} | — | — | — |")
lines.append("")
if agg_r1 is not None:
    lines.append(f"## Retrain at r=1 (gate fired: noretrain r=1 ≤ {R1_GATE_PPL})")
    lines.append("")
    lines.append(f"Trainable params: {1 * 768 + 768:,} = 1×768 (B) + 768 (bias).")
    lines.append("")
    lines.append("| seed | init | best | best_step | final@N | restored |")
    lines.append("|---|---|---|---|---|---|")
    for s in retrain_r1_seeds:
        lines.append(f"| {s['seed']} | {fmt(s['init_ppl'])} | {fmt(s['best_ppl'])} | {s['best_step']} | {fmt(s['final_ppl_at_n_steps'])} | {fmt(s['restored_best_ppl'])} |")
    lines.append(f"| **mean** | — | **{fmt(agg_r1['best_mean'])}** | — | — | — |")
    lines.append(f"| **std**  | — | {fmt(agg_r1['best_std'])} | — | — | — |")
    lines.append("")
else:
    lines.append(f"## Retrain at r=1 — SKIPPED (noretrain r=1 PPL = {fmt(noretrain_results[1]['ppl'])} > gate {R1_GATE_PPL})")
    lines.append("")
lines.append("## Verdict logic (item 295 verbatim)")
lines.append("")
lines.append(f"- r={RETRAIN_RANK} retrain best_mean = {fmt(agg_r2['best_mean'])}; CONFIRMED iff ≤ {CONFIRMED_R2_RETRAIN_LE}; REFUTED iff ≥ {REFUTED_R2_RETRAIN_GE}")
if agg_r1 is not None:
    lines.append(f"- r=1 retrain best_mean = {fmt(agg_r1['best_mean'])}; CONFIRMED iff ≤ {CONFIRMED_R1_RETRAIN_LE}")
else:
    lines.append(f"- r=1 retrain SKIPPED")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append("## Comparisons")
lines.append("")
lines.append(f"- Δ vs float: noretrain r={RETRAIN_RANK} = {noretrain_results[RETRAIN_RANK]['delta_float']:+.3f}; retrain r={RETRAIN_RANK} best_mean = {agg_r2['best_mean'] - float_ppl:+.3f}")
lines.append(f"- Δ retrain vs noretrain at r={RETRAIN_RANK}: {agg_r2['best_mean'] - noretrain_results[RETRAIN_RANK]['ppl']:+.3f} (Adam recovery)")
lines.append(f"- Δ vs cycle 280 b1 r=2 (2,304 params, {CYCLE_280_B1_R2_BEST_MEAN}): {agg_r2['best_mean'] - CYCLE_280_B1_R2_BEST_MEAN:+.3f}")
lines.append(f"- Δ vs cycle 270 b11 r=22 (17,664 params, {CYCLE_270_B11_R22_BEST_MEAN}): {agg_r2['best_mean'] - CYCLE_270_B11_R22_BEST_MEAN:+.3f}")
lines.append(f"- Δ vs cycle 282 b10 r=17 (13,824 params, {CYCLE_282_B10_R17_BEST_MEAN}): {agg_r2['best_mean'] - CYCLE_282_B10_R17_BEST_MEAN:+.3f}")
lines.append(f"- Δ vs cycle 283 [2] K=16 (13,056 params, {CYCLE_283_B2_K16_BEST_MEAN}): {agg_r2['best_mean'] - CYCLE_283_B2_K16_BEST_MEAN:+.3f}")
if agg_r1 is not None:
    lines.append(f"- r=1 retrain (1,536 params) vs float: {agg_r1['best_mean'] - float_ppl:+.3f}")
    lines.append(f"- r=1 retrain (1,536 params) vs cycle 280 (2,304 params, {CYCLE_280_B1_R2_BEST_MEAN}): {agg_r1['best_mean'] - CYCLE_280_B1_R2_BEST_MEAN:+.3f}")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
