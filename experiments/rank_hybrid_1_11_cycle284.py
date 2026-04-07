"""
Cycle 284 — Item 294: Hybrid block 1 rank-2 c_proj + block 11 rank-22 c_proj.

Composes cycle 280 (block 1 rank-2 SVD) with cycle 270 (block 11 rank-22 SVD)
into a single homogeneous low-rank weight-SVD hybrid. Joint Adam retrains
B1, b1, B11, b11 (19,968 params total). All other model parameters frozen.

HF Conv1D layout: c_proj.weight is (3072, 768); applied as x @ weight + bias.
SVD: U (3072, 768), Σ (768,), Vh (768, 768). Rank-r approx:
  W_r = (U[:, :r] * Σ[:r]) @ Vh[:r, :].

Verdict (item 294 verbatim):
  CONFIRMED iff best_mean ≤ 55.0  (matches cycle 270 band → composition non-destructive)
  REFUTED   iff best_mean ≥ 58.0  (composition destructive, same as cycle 275)
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
TARGET_BLOCKS = (1, 11)
RANKS = (2, 22)
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 4000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42
OUTPUT_PATH = "autoresearch/results/cycle_284_rank_hybrid_1_11.json"
SUMMARY_PATH = "autoresearch/results/cycle_284_summary.md"

CONFIRMED_LE = 55.0
REFUTED_GE = 58.0

FLOAT_BASELINE_REF = 53.895
CYCLE_270_B11_R22_BEST_MEAN = 54.4687  # block 11 rank-22 retrain
CYCLE_280_B1_R2_BEST_MEAN = 49.0324    # block 1 rank-2 retrain
CYCLE_282_B10_R17_BEST_MEAN = 50.0445  # block 10 rank-17 retrain
CYCLE_275_HETEROGENEOUS_BEST_MEAN = 48.2004  # [1] K=512 + b11 r=22 (REFUTED ax)

torch.manual_seed(LOAD_SEED)
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


# ── Helpers ──────────────────────────────────────────────────────────────────
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

    Frozen A = U[:, :r] * Σ[:r] (3072, r) as buffer; trainable B (r, 768) as
    nn.Parameter (init Vh[:r, :]); trainable bias (768,) init c_proj.bias.
    Forward: x @ (A @ B) + b. Matches HF Conv1D semantics.
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


# ── Float baseline ────────────────────────────────────────────────────────────
print("Float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
t0 = time.time()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.3f}  ({time.time()-t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from cycle reference {FLOAT_BASELINE_REF} by > 0.5"
)


# ── SVD on each target block c_proj ───────────────────────────────────────────
svd_cache = {}
for blk, r in zip(TARGET_BLOCKS, RANKS):
    print(f"\nSVD of block {blk} c_proj.weight (target rank {r})...", flush=True)
    W_orig = base_model.transformer.h[blk].mlp.c_proj.weight.data.detach().clone()
    b_orig = base_model.transformer.h[blk].mlp.c_proj.bias.data.detach().clone()
    assert W_orig.shape == (3072, 768), (
        f"block {blk} c_proj.weight shape {W_orig.shape} != (3072, 768)"
    )
    W_cpu = W_orig.float().cpu()
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    print(f"  SVD done in {time.time()-t0:.1f}s; "
          f"U {tuple(U.shape)}, S {tuple(S.shape)}, Vh {tuple(Vh.shape)}", flush=True)
    print(f"  Top-5 singular values: {S[:5].tolist()}", flush=True)
    A_r = (U[:, :r] * S[:r]).contiguous()  # (3072, r)
    B_r = Vh[:r, :].contiguous()           # (r, 768)
    svd_cache[blk] = {
        "A": A_r,
        "B_init": B_r,
        "b_init": b_orig,
        "S_top5": S[:5].tolist(),
        "S_min": float(S[-1]),
        "rank": r,
    }

# Free base model — fresh per seed below
del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Joint Adam retrain (3 seeds) ──────────────────────────────────────────────
print(f"\n=== Hybrid retrain b1 r=2 + b11 r=22 ({len(SEEDS)} seeds) ===", flush=True)
expected_trainable = sum(r * 768 + 768 for r in RANKS)
assert expected_trainable == 19968, expected_trainable

retrain_seeds = []
for seed in SEEDS:
    print(f"\n--- seed {seed} ---", flush=True)
    torch.manual_seed(seed)
    random.seed(seed)

    model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model_s.eval()

    # Install both LowRankCProj modules
    lowrank_modules = {}
    for blk in TARGET_BLOCKS:
        cache = svd_cache[blk]
        lr_mod = LowRankCProj(
            A_frozen=cache["A"].to(device=DEVICE, dtype=torch.float32),
            B_init=cache["B_init"].to(device=DEVICE, dtype=torch.float32),
            b_init=cache["b_init"].to(device=DEVICE, dtype=torch.float32),
            nf=768,
        ).to(DEVICE)
        model_s.transformer.h[blk].mlp.c_proj = lr_mod
        lowrank_modules[blk] = lr_mod

    init_ppl = eval_ppl(model_s, eval_ids)
    print(f"  init PPL (hybrid no-retrain): {init_ppl:.3f}", flush=True)

    for p in model_s.parameters():
        p.requires_grad_(False)
    trainable = []
    for blk in TARGET_BLOCKS:
        lowrank_modules[blk].B.requires_grad_(True)
        lowrank_modules[blk].bias.requires_grad_(True)
        trainable.append(lowrank_modules[blk].B)
        trainable.append(lowrank_modules[blk].bias)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_trainable:,} (expect {expected_trainable:,})", flush=True)
    assert n_trainable == expected_trainable, (
        f"trainable count {n_trainable} != {expected_trainable}"
    )

    optimizer = torch.optim.Adam(trainable, lr=LR)

    calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
    random.shuffle(calib_ids)

    best_ppl = init_ppl
    best_step = 0
    best_state = {
        f"B_{blk}": lowrank_modules[blk].B.detach().clone() for blk in TARGET_BLOCKS
    }
    for blk in TARGET_BLOCKS:
        best_state[f"bias_{blk}"] = lowrank_modules[blk].bias.detach().clone()
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
                history.append({
                    "step": step,
                    "eval_ppl": float(ppl_e),
                    "calib_loss": float(loss.item()),
                })
                if ppl_e < best_ppl:
                    best_ppl = ppl_e
                    best_step = step
                    best_state = {
                        f"B_{blk}": lowrank_modules[blk].B.detach().clone()
                        for blk in TARGET_BLOCKS
                    }
                    for blk in TARGET_BLOCKS:
                        best_state[f"bias_{blk}"] = lowrank_modules[blk].bias.detach().clone()
                if step % 500 == 0:
                    print(f"    step {step:4d}: eval_ppl={ppl_e:.4f}  "
                          f"best={best_ppl:.4f}@{best_step}", flush=True)

    final_ppl = eval_ppl(model_s, eval_ids)
    with torch.no_grad():
        for blk in TARGET_BLOCKS:
            lowrank_modules[blk].B.data.copy_(best_state[f"B_{blk}"])
            lowrank_modules[blk].bias.data.copy_(best_state[f"bias_{blk}"])
    restored_ppl = eval_ppl(model_s, eval_ids)
    assert abs(restored_ppl - best_ppl) < 1e-3, (
        f"seed {seed}: restored {restored_ppl} != best {best_ppl}"
    )
    dt = time.time() - t_train
    print(f"  seed {seed}: init={init_ppl:.4f} best={best_ppl:.4f}@{best_step} "
          f"final@{N_STEPS}={final_ppl:.4f} restored={restored_ppl:.4f}  "
          f"[{dt/60:.1f} min]", flush=True)

    retrain_seeds.append({
        "seed": seed,
        "init_ppl": float(init_ppl),
        "best_ppl": float(best_ppl),
        "best_step": int(best_step),
        "final_ppl_at_n_steps": float(final_ppl),
        "restored_best_ppl": float(restored_ppl),
        "history": history,
    })

    del model_s, lowrank_modules, optimizer
    if DEVICE == "mps":
        torch.mps.empty_cache()

# Aggregate
best_ppls = [s["best_ppl"] for s in retrain_seeds]
best_mean = float(sum(best_ppls) / len(best_ppls))
best_var = float(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
best_std = math.sqrt(best_var)
init_ppls = [s["init_ppl"] for s in retrain_seeds]
init_mean = float(sum(init_ppls) / len(init_ppls))
final_ppls = [s["final_ppl_at_n_steps"] for s in retrain_seeds]
final_mean = float(sum(final_ppls) / len(final_ppls))
print(f"\nHybrid b1r2+b11r22: best_mean={best_mean:.4f} ± {best_std:.4f}  "
      f"(n={len(SEEDS)})", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
if best_mean <= CONFIRMED_LE:
    verdict = "CONFIRMED"
elif best_mean >= REFUTED_GE:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== Verdict ===", flush=True)
print(f"  best_mean = {best_mean:.4f}", flush=True)
print(f"  CONFIRMED iff <= {CONFIRMED_LE}; REFUTED iff >= {REFUTED_GE}", flush=True)
print(f"  VERDICT: {verdict}", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 284,
    "item": 294,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "retrain_seeds": retrain_seeds,
    "retrain_aggregate": {
        "init_mean": init_mean,
        "best_mean": best_mean,
        "best_std": best_std,
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
        "final_mean": final_mean,
        "n_seeds": len(SEEDS),
    },
    "verdict_logic": {
        "best_mean": best_mean,
        "threshold_confirmed_le": CONFIRMED_LE,
        "threshold_refuted_ge": REFUTED_GE,
        "verdict": verdict,
    },
    "config": {
        "TARGET_BLOCKS": list(TARGET_BLOCKS),
        "RANKS": list(RANKS),
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS": N_STEPS,
        "LR": LR,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
        "trainable_params_per_fit": expected_trainable,
    },
    "svd_summary": {
        str(blk): {
            "rank": svd_cache[blk]["rank"],
            "S_top5": svd_cache[blk]["S_top5"],
            "S_min": svd_cache[blk]["S_min"],
        }
        for blk in TARGET_BLOCKS
    },
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "cycle_270_block11_r22_retrain_best_mean": CYCLE_270_B11_R22_BEST_MEAN,
        "cycle_280_block1_r2_retrain_best_mean": CYCLE_280_B1_R2_BEST_MEAN,
        "cycle_282_block10_r17_retrain_best_mean": CYCLE_282_B10_R17_BEST_MEAN,
        "cycle_275_heterogeneous_hybrid_best_mean": CYCLE_275_HETEROGENEOUS_BEST_MEAN,
    },
    "deltas": {
        "vs_float": best_mean - float_ppl,
        "vs_cycle_270_b11_r22": best_mean - CYCLE_270_B11_R22_BEST_MEAN,
        "vs_cycle_280_b1_r2": best_mean - CYCLE_280_B1_R2_BEST_MEAN,
        "vs_cycle_282_b10_r17": best_mean - CYCLE_282_B10_R17_BEST_MEAN,
        "vs_cycle_275_heterogeneous": best_mean - CYCLE_275_HETEROGENEOUS_BEST_MEAN,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append(f"# Cycle 284 — Item 294: Hybrid block 1 rank-2 + block 11 rank-22 c_proj")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Blocks: {list(TARGET_BLOCKS)}, ranks: {list(RANKS)}")
lines.append(f"- Trainable params per fit: {expected_trainable:,} = "
             f"(2·768 + 768) + (22·768 + 768) = 2,304 + 17,664")
lines.append("")
lines.append(f"## Joint Adam retrain ({len(SEEDS)} seeds, LR={LR}, {N_STEPS} steps)")
lines.append("")
lines.append("| seed | init | best | best_step | final@N | restored |")
lines.append("|---|---|---|---|---|---|")
for s in retrain_seeds:
    lines.append(f"| {s['seed']} | {fmt(s['init_ppl'])} | {fmt(s['best_ppl'])} | "
                 f"{s['best_step']} | {fmt(s['final_ppl_at_n_steps'])} | "
                 f"{fmt(s['restored_best_ppl'])} |")
lines.append(f"| **mean** | {fmt(init_mean)} | **{fmt(best_mean)}** | — | "
             f"{fmt(final_mean)} | — |")
lines.append(f"| **std**  | — | {fmt(best_std)} | — | — | — |")
lines.append("")
lines.append("## Verdict logic (item 294 verbatim)")
lines.append("")
lines.append(f"- best_mean = {fmt(best_mean)}")
lines.append(f"- CONFIRMED iff best_mean ≤ {CONFIRMED_LE}? {best_mean <= CONFIRMED_LE}")
lines.append(f"- REFUTED iff best_mean ≥ {REFUTED_GE}? {best_mean >= REFUTED_GE}")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append("## Comparisons")
lines.append("")
lines.append(f"- Δ vs float ({FLOAT_BASELINE_REF}): {best_mean - float_ppl:+.3f}")
lines.append(f"- Δ vs cycle 270 b11 r=22 ({CYCLE_270_B11_R22_BEST_MEAN}): "
             f"{best_mean - CYCLE_270_B11_R22_BEST_MEAN:+.3f}")
lines.append(f"- Δ vs cycle 280 b1 r=2 ({CYCLE_280_B1_R2_BEST_MEAN}): "
             f"{best_mean - CYCLE_280_B1_R2_BEST_MEAN:+.3f}")
lines.append(f"- Δ vs cycle 282 b10 r=17 ({CYCLE_282_B10_R17_BEST_MEAN}): "
             f"{best_mean - CYCLE_282_B10_R17_BEST_MEAN:+.3f}")
lines.append(f"- Δ vs cycle 275 heterogeneous hybrid ({CYCLE_275_HETEROGENEOUS_BEST_MEAN}): "
             f"{best_mean - CYCLE_275_HETEROGENEOUS_BEST_MEAN:+.3f} (different scoring axis)")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)
