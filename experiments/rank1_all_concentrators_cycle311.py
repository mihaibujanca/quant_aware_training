"""
Cycle 311 — Item 321: Rank-1 LASER for all concentrator blocks {b2, b3, b10, b11}.

Extends cycle 292 (b1 r=1 = 50.08) to the other 4 cycle-282-family blocks.
For each block in {2, 3, 10, 11}: install single-block LowRankCProj at r=1
(cycle 292 template), freeze rest of model, joint Adam LR=1e-4 over
B (1, 768) + bias (768) = 1,536 trainable params, 3 seeds, best-state restore.

HF Conv1D layout: c_proj.weight is (3072, 768); applied as x @ weight + bias.
SVD: U (3072, 768), Σ (768,), Vh (768, 768). Rank-1 approx:
W_1 = (U[:, :1] * Σ[:1]) @ Vh[:1, :].

Verdict (item 321 verbatim):
  n_pass := |{b in BLOCKS : r1_best_mean_ppl[b] <= 60}|
  CONFIRMED iff n_pass >= 3
  REFUTED   iff n_pass <= 1
  INCONCLUSIVE otherwise
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
BLOCKS = [2, 3, 10, 11]
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
OUTPUT_PATH = "autoresearch/results/cycle_311_rank1_all_concentrators.json"
SUMMARY_PATH = "autoresearch/results/cycle_311_summary.md"

# Verdict thresholds (item 321 verbatim)
PASS_THRESHOLD_PPL = 60.0
CONFIRMED_N_PASS_GE = 3
REFUTED_N_PASS_LE = 1

FLOAT_BASELINE_REF = 53.8949  # cycles 268-310 reference
CYCLE_292_B1_R1_BEST_MEAN = 50.0794  # cycle 292 b1 r=1 retrain
CYCLE_280_B1_R2_BEST_MEAN = 49.0324  # cycle 280 b1 r=2 retrain
CYCLE_285_B2_R2_BEST_MEAN = 49.3290  # cycle 285 b2 r=2 retrain
CYCLE_285_B2_R1_NORETRAIN = 72.3808  # cycle 285 b2 r=1 no-retrain
CYCLE_288_B3_R24_BEST_MEAN = 50.5935  # cycle 288 b3 r=24 retrain
CYCLE_289_B10_R5_BEST_MEAN = 51.4659  # cycle 289 b10 r=5 retrain
CYCLE_270_B11_R22_BEST_MEAN = 54.4687  # cycle 270 b11 r=22 retrain

# Cycle 285 b2 r=1 no-retrain anchor for cross-check (only block with prior r=1 ref)
B2_R1_NORETRAIN_TOL = 0.01

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
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
t0 = time.time()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from cycle reference {FLOAT_BASELINE_REF} by > 0.5"
)


# ── Cache SVDs of all 4 c_proj weights once ──────────────────────────────────
print("\nComputing SVDs of c_proj weights for blocks", BLOCKS, "...", flush=True)
svd_cache = {}
for blk in BLOCKS:
    W_orig = base_model.transformer.h[blk].mlp.c_proj.weight.data.detach().clone()
    b_orig = base_model.transformer.h[blk].mlp.c_proj.bias.data.detach().clone()
    assert W_orig.shape == (3072, 768), f"block {blk} c_proj.weight shape {W_orig.shape} != (3072, 768)"
    W_cpu = W_orig.float().cpu()
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    print(f"  block {blk}: SVD in {time.time()-t0:.1f}s, S[0]={float(S[0]):.4f} S[1]={float(S[1]):.4f} S[-1]={float(S[-1]):.6f}", flush=True)
    A_r = (U[:, :RETRAIN_RANK] * S[:RETRAIN_RANK]).contiguous()  # (3072, 1)
    B_init = Vh[:RETRAIN_RANK, :].contiguous()                   # (1, 768)
    S2 = S * S
    energy_r = float(S2[:RETRAIN_RANK].sum() / S2.sum())
    svd_cache[blk] = {
        "A": A_r,
        "B_init": B_init,
        "b_init": b_orig.float().cpu(),
        "S_top5": S[:5].tolist(),
        "S_min": float(S[-1]),
        "energy_r1": energy_r,
        "W_orig": W_orig,  # to restore later for noretrain check across blocks
    }


# ── No-retrain r=1 PPL per block (sanity) ────────────────────────────────────
print("\n=== No-retrain r=1 PPL per block ===", flush=True)
noretrain_ppls = {}
for blk in BLOCKS:
    A_r = svd_cache[blk]["A"]
    B_init = svd_cache[blk]["B_init"]
    W_r = (A_r @ B_init).to(
        device=svd_cache[blk]["W_orig"].device,
        dtype=svd_cache[blk]["W_orig"].dtype,
    )
    with torch.no_grad():
        base_model.transformer.h[blk].mlp.c_proj.weight.data.copy_(W_r)
    t0 = time.time()
    ppl_nr = eval_ppl(base_model, eval_ids)
    dt = time.time() - t0
    delta_float = ppl_nr - float_ppl
    pct = (ppl_nr / float_ppl - 1) * 100
    print(f"  block {blk}: PPL={ppl_nr:.4f}  Δfloat={delta_float:+.3f}  ({pct:+.2f}%)  energy_r1={svd_cache[blk]['energy_r1']:.4f}  [{dt:.1f}s]", flush=True)
    noretrain_ppls[blk] = float(ppl_nr)
    # restore
    with torch.no_grad():
        base_model.transformer.h[blk].mlp.c_proj.weight.data.copy_(svd_cache[blk]["W_orig"])

# Restore-check: full float PPL after restore
ppl_after_restore = eval_ppl(base_model, eval_ids)
assert abs(ppl_after_restore - float_ppl) < 1e-3, f"restore failed: {ppl_after_restore} vs {float_ppl}"
print(f"  All restores OK; float PPL = {ppl_after_restore:.4f}", flush=True)

# Cross-check b2 r=1 noretrain vs cycle 285 reference
delta_b2 = abs(noretrain_ppls[2] - CYCLE_285_B2_R1_NORETRAIN)
print(f"  Cross-check b2 r=1 noretrain vs cycle 285 ref {CYCLE_285_B2_R1_NORETRAIN}: |Δ|={delta_b2:.4f}", flush=True)
assert delta_b2 < B2_R1_NORETRAIN_TOL, (
    f"b2 r=1 noretrain cross-check failed: {noretrain_ppls[2]} vs {CYCLE_285_B2_R1_NORETRAIN}"
)

del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Retrain at r=1 per block (3 seeds each) ──────────────────────────────────
print(f"\n=== Retrain at r={RETRAIN_RANK} for blocks {BLOCKS} (3 seeds each) ===", flush=True)
per_block_results = {}
for blk in BLOCKS:
    print(f"\n========== block {blk} ==========", flush=True)
    A1 = svd_cache[blk]["A"]
    B1_init = svd_cache[blk]["B_init"]
    b_init_cpu = svd_cache[blk]["b_init"]

    seed_results = []
    for seed in SEEDS:
        print(f"\n--- block {blk} / seed {seed} ---", flush=True)
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
        model_s.transformer.h[blk].mlp.c_proj = lowrank

        init_ppl = eval_ppl(model_s, eval_ids)
        print(f"  init PPL (rank-{RETRAIN_RANK} no retrain): {init_ppl:.4f}", flush=True)
        assert abs(init_ppl - noretrain_ppls[blk]) < 1e-2, (
            f"init PPL {init_ppl} != noretrain rank-{RETRAIN_RANK} {noretrain_ppls[blk]}"
        )

        for p in model_s.parameters():
            p.requires_grad_(False)
        lowrank.B.requires_grad_(True)
        lowrank.bias.requires_grad_(True)
        trainable = [lowrank.B, lowrank.bias]
        n_trainable = sum(p.numel() for p in trainable)
        expected_trainable = RETRAIN_RANK * 768 + 768
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
            f"block {blk} seed {seed}: restored {restored_ppl} != best {best_ppl}"
        )
        dt = time.time() - t_train
        print(f"  block {blk} seed {seed}: init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} final@{N_STEPS}={final_ppl:.4f} restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

        seed_results.append({
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

    best_ppls = [s["best_ppl"] for s in seed_results]
    best_mean = float(sum(best_ppls) / len(best_ppls))
    best_var = float(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
    best_std = math.sqrt(best_var)
    print(f"\nblock {blk}: best_mean={best_mean:.4f} ± {best_std:.4f} (n={len(SEEDS)})", flush=True)

    per_block_results[blk] = {
        "block": blk,
        "noretrain_r1_ppl": noretrain_ppls[blk],
        "energy_r1": svd_cache[blk]["energy_r1"],
        "S_top5": svd_cache[blk]["S_top5"],
        "S_min": svd_cache[blk]["S_min"],
        "seeds": seed_results,
        "best_mean": best_mean,
        "best_std": best_std,
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
        "n_seeds": len(SEEDS),
        "passes_threshold": bool(best_mean <= PASS_THRESHOLD_PPL),
    }


# ── Verdict ───────────────────────────────────────────────────────────────────
n_pass = sum(1 for blk in BLOCKS if per_block_results[blk]["passes_threshold"])
if n_pass >= CONFIRMED_N_PASS_GE:
    verdict = "CONFIRMED"
elif n_pass <= REFUTED_N_PASS_LE:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== Verdict ===", flush=True)
for blk in BLOCKS:
    pr = per_block_results[blk]
    flag = "PASS" if pr["passes_threshold"] else "FAIL"
    print(f"  block {blk}: best_mean={pr['best_mean']:.4f} (≤ {PASS_THRESHOLD_PPL}? {flag})", flush=True)
print(f"  n_pass = {n_pass} / {len(BLOCKS)}  (CONFIRMED ≥ {CONFIRMED_N_PASS_GE}, REFUTED ≤ {REFUTED_N_PASS_LE})", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 311,
    "item": 321,
    "verdict": verdict,
    "n_pass": int(n_pass),
    "n_blocks": len(BLOCKS),
    "blocks": BLOCKS,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "noretrain_r1_ppls": {str(b): noretrain_ppls[b] for b in BLOCKS},
    "per_block": {str(b): per_block_results[b] for b in BLOCKS},
    "verdict_logic": {
        "pass_threshold_ppl": PASS_THRESHOLD_PPL,
        "n_pass": int(n_pass),
        "n_blocks": len(BLOCKS),
        "threshold_confirmed_n_pass_ge": CONFIRMED_N_PASS_GE,
        "threshold_refuted_n_pass_le": REFUTED_N_PASS_LE,
        "verdict": verdict,
        "per_block_passes": {str(b): per_block_results[b]["passes_threshold"] for b in BLOCKS},
    },
    "config": {
        "BLOCKS": BLOCKS,
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
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "cycle_292_b1_r1_best_mean": CYCLE_292_B1_R1_BEST_MEAN,
        "cycle_280_b1_r2_best_mean": CYCLE_280_B1_R2_BEST_MEAN,
        "cycle_285_b2_r2_best_mean": CYCLE_285_B2_R2_BEST_MEAN,
        "cycle_285_b2_r1_noretrain": CYCLE_285_B2_R1_NORETRAIN,
        "cycle_288_b3_r24_best_mean": CYCLE_288_B3_R24_BEST_MEAN,
        "cycle_289_b10_r5_best_mean": CYCLE_289_B10_R5_BEST_MEAN,
        "cycle_270_b11_r22_best_mean": CYCLE_270_B11_R22_BEST_MEAN,
    },
    "cross_checks": {
        "b2_r1_noretrain_measured": float(noretrain_ppls[2]),
        "b2_r1_noretrain_ref_cycle285": CYCLE_285_B2_R1_NORETRAIN,
        "b2_r1_noretrain_abs_delta": float(delta_b2),
        "b2_r1_noretrain_tol": B2_R1_NORETRAIN_TOL,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append(f"# Cycle 311 — Item 321: Rank-1 LASER for all concentrator blocks {{b2, b3, b10, b11}}")
lines.append("")
lines.append(f"**Verdict: {verdict}**  (n_pass = {n_pass} / {len(BLOCKS)} at PPL ≤ {PASS_THRESHOLD_PPL})")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Blocks: {BLOCKS}, c_proj.weight shape (3072, 768), HF Conv1D")
lines.append(f"- Trainable per fit: 1,536 = 1·768 (B) + 768 (bias). A=(U[:, :1]·Σ[:1]) frozen.")
lines.append(f"- Adam LR={LR}, N_STEPS={N_STEPS}, EVAL_EVERY={EVAL_EVERY}, SEEDS={SEEDS}, best-state restore.")
lines.append("")
lines.append("## No-retrain r=1 PPL per block (sanity)")
lines.append("")
lines.append("| block | PPL | Δfloat | %above | top-1 energy |")
lines.append("|---|---|---|---|---|")
for blk in BLOCKS:
    ppl = noretrain_ppls[blk]
    e = per_block_results[blk]["energy_r1"]
    lines.append(f"| {blk} | {fmt(ppl)} | {ppl - float_ppl:+.3f} | {(ppl/float_ppl - 1)*100:+.2f}% | {fmt(e)} |")
lines.append("")
lines.append(f"- b2 r=1 noretrain cross-check vs cycle 285 ref {CYCLE_285_B2_R1_NORETRAIN}: |Δ| = {delta_b2:.4f}")
lines.append("")
lines.append(f"## Retrain at r=1 (3 seeds per block, Adam LR={LR}, {N_STEPS} steps)")
lines.append("")
lines.append("| block | seed | init | best | best_step | final@N | restored |")
lines.append("|---|---|---|---|---|---|---|")
for blk in BLOCKS:
    for s in per_block_results[blk]["seeds"]:
        lines.append(f"| {blk} | {s['seed']} | {fmt(s['init_ppl'])} | {fmt(s['best_ppl'])} | {s['best_step']} | {fmt(s['final_ppl_at_n_steps'])} | {fmt(s['restored_best_ppl'])} |")
lines.append("")
lines.append("## Per-block aggregate")
lines.append("")
lines.append("| block | best_mean | best_std | best_min | best_max | passes ≤60 |")
lines.append("|---|---|---|---|---|---|")
for blk in BLOCKS:
    pr = per_block_results[blk]
    flag = "yes" if pr["passes_threshold"] else "no"
    lines.append(f"| {blk} | {fmt(pr['best_mean'])} | {fmt(pr['best_std'])} | {fmt(pr['best_min'])} | {fmt(pr['best_max'])} | {flag} |")
lines.append("")
lines.append("## Verdict logic (item 321 verbatim)")
lines.append("")
lines.append(f"- n_pass := |{{b ∈ {BLOCKS} : best_mean ≤ {PASS_THRESHOLD_PPL}}}| = **{n_pass}**")
lines.append(f"- CONFIRMED iff n_pass ≥ {CONFIRMED_N_PASS_GE}")
lines.append(f"- REFUTED iff n_pass ≤ {REFUTED_N_PASS_LE}")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append("## Comparisons")
lines.append("")
lines.append(f"- Cycle 292 b1 r=1 best_mean: {CYCLE_292_B1_R1_BEST_MEAN}")
lines.append(f"- Cycle 280 b1 r=2 best_mean: {CYCLE_280_B1_R2_BEST_MEAN}")
lines.append(f"- Cycle 285 b2 r=2 best_mean: {CYCLE_285_B2_R2_BEST_MEAN}")
lines.append(f"- Cycle 288 b3 r=24 best_mean: {CYCLE_288_B3_R24_BEST_MEAN}")
lines.append(f"- Cycle 289 b10 r=5 best_mean: {CYCLE_289_B10_R5_BEST_MEAN}")
lines.append(f"- Cycle 270 b11 r=22 best_mean: {CYCLE_270_B11_R22_BEST_MEAN}")
lines.append("")
lines.append("| block | r=1 (this cycle) | multi-rank anchor | rank | Δ (r1 − anchor) |")
lines.append("|---|---|---|---|---|")
multi_rank = {2: (CYCLE_285_B2_R2_BEST_MEAN, 2), 3: (CYCLE_288_B3_R24_BEST_MEAN, 24), 10: (CYCLE_289_B10_R5_BEST_MEAN, 5), 11: (CYCLE_270_B11_R22_BEST_MEAN, 22)}
for blk in BLOCKS:
    bm = per_block_results[blk]["best_mean"]
    anchor, anchor_r = multi_rank[blk]
    lines.append(f"| {blk} | {fmt(bm)} | {fmt(anchor)} | {anchor_r} | {bm - anchor:+.4f} |")
lines.append("")
with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
