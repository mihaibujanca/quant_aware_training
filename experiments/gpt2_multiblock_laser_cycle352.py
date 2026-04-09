#!/usr/bin/env python3
"""Cycle 343 / Item 352 — GPT-2 multi-block LASER accumulation curve.

Tests whether LASER-compressed blocks compose when added incrementally,
ordered by output-covariance concentration (most concentrated first).

Accumulation stages:
  Stage 1: b2 (rank 2, eff_rank 1.05)
  Stage 2: b2 + b1 (rank 2, eff_rank 1.94)
  Stage 3: b2 + b1 + b10 (rank 17, eff_rank 16.70)
  Stage 4: b2 + b1 + b10 + b11 (rank 23, eff_rank 22.49)

Each stage: SVD → LowRankCProj → joint Adam retrain (1000 steps) → eval PPL.
Ranks = ceil(output-covariance eff_rank) from cycle 341.

Prior results:
  Float PPL: 53.90
  b1 rank-2 single: 49.03 (cycle 280)
  b2 rank-2 single: 49.33 (cycle 285)
  5-block all-at-once: 73.01 (cycle 305, REFUTED)

Verdict (item 352):
  CONFIRMED if any stage with >=2 blocks has best_mean <= 54.90 (float + 1.0)
  REFUTED if single-block best < float and all >=2-block > single-block best
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
# Accumulation order: most concentrated first (ascending eff_rank)
ACCUM_BLOCKS = [2, 1, 10, 11]
ACCUM_RANKS = [2, 2, 17, 23]  # ceil(output-cov eff_rank)
ACCUM_EFFRANKS = [1.05, 1.94, 16.70, 22.49]

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 1000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
LOAD_SEED = 42

FLOAT_BASELINE_REF = 53.895
CONFIRMED_LE = 54.90  # float + 1.0

OUTPUT_PATH = "autoresearch/results/cycle_343_accumulation.json"

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading GPT-2 tokenizer and wikitext-2 data...", flush=True)
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

calib_ids_cpu = list(all_ids[:N_CALIB])
eval_ids = [t.to(DEVICE) for t in all_ids[N_CALIB : N_CALIB + N_EVAL]]
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

    Frozen A = U[:, :r] * S[:r] (3072, r); trainable B (r, 768) init Vh[:r, :];
    trainable bias (768,) init c_proj.bias.
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
print("Computing float baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f} (ref {FLOAT_BASELINE_REF})", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from reference {FLOAT_BASELINE_REF} by > 0.5"
)

# ── SVD all target blocks ────────────────────────────────────────────────────
svd_cache = {}
for blk, r in zip(ACCUM_BLOCKS, ACCUM_RANKS):
    W_orig = base_model.transformer.h[blk].mlp.c_proj.weight.data.detach().clone()
    b_orig = base_model.transformer.h[blk].mlp.c_proj.bias.data.detach().clone()
    assert W_orig.shape == (3072, 768), f"block {blk} shape {W_orig.shape}"
    W_cpu = W_orig.float().cpu()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    A_r = (U[:, :r] * S[:r]).contiguous()
    B_r = Vh[:r, :].contiguous()
    svd_cache[blk] = {
        "A": A_r, "B_init": B_r, "b_init": b_orig,
        "S_top5": S[:5].tolist(), "S_min": float(S[-1]), "rank": r,
    }
    print(f"SVD block {blk}: rank {r}, top-5 S = {S[:5].tolist()}", flush=True)

del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Accumulation curve ────────────────────────────────────────────────────────
stages = []
for n_blocks in range(1, len(ACCUM_BLOCKS) + 1):
    active_blocks = ACCUM_BLOCKS[:n_blocks]
    active_ranks = ACCUM_RANKS[:n_blocks]
    active_effranks = ACCUM_EFFRANKS[:n_blocks]
    trainable_count = sum(svd_cache[b]["rank"] * 768 + 768 for b in active_blocks)

    print(f"\n{'='*60}", flush=True)
    print(f"Stage {n_blocks}: blocks {active_blocks}, ranks {active_ranks}, "
          f"trainable={trainable_count:,}", flush=True)
    print(f"{'='*60}", flush=True)

    seed_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        random.seed(seed)

        model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model.eval()

        # Install LowRankCProj for active blocks
        lowrank_modules = {}
        for blk in active_blocks:
            cache = svd_cache[blk]
            lr_mod = LowRankCProj(
                A_frozen=cache["A"].to(device=DEVICE, dtype=torch.float32),
                B_init=cache["B_init"].to(device=DEVICE, dtype=torch.float32),
                b_init=cache["b_init"].to(device=DEVICE, dtype=torch.float32),
            ).to(DEVICE)
            model.transformer.h[blk].mlp.c_proj = lr_mod
            lowrank_modules[blk] = lr_mod

        init_ppl = eval_ppl(model, eval_ids)

        # Freeze everything, unfreeze B-factors + biases
        for p in model.parameters():
            p.requires_grad_(False)
        trainable = []
        for blk in active_blocks:
            lowrank_modules[blk].B.requires_grad_(True)
            lowrank_modules[blk].bias.requires_grad_(True)
            trainable.append(lowrank_modules[blk].B)
            trainable.append(lowrank_modules[blk].bias)
        n_train = sum(p.numel() for p in trainable)
        assert n_train == trainable_count, f"{n_train} != {trainable_count}"

        optimizer = torch.optim.Adam(trainable, lr=LR)
        calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids)

        best_ppl = init_ppl
        best_step = 0
        best_state = {}
        for blk in active_blocks:
            best_state[f"B_{blk}"] = lowrank_modules[blk].B.detach().clone()
            best_state[f"bias_{blk}"] = lowrank_modules[blk].bias.detach().clone()
        history = [{"step": 0, "eval_ppl": float(init_ppl)}]

        step = 0
        t0 = time.time()
        while step < N_STEPS:
            for ids in calib_ids:
                if step >= N_STEPS:
                    break
                model.train()
                optimizer.zero_grad()
                out = model(ids, labels=ids)
                loss = out.loss
                loss.backward()
                optimizer.step()
                step += 1

                if step % EVAL_EVERY == 0:
                    ppl_e = eval_ppl(model, eval_ids)
                    history.append({"step": step, "eval_ppl": float(ppl_e)})
                    if ppl_e < best_ppl:
                        best_ppl = ppl_e
                        best_step = step
                        best_state = {}
                        for blk in active_blocks:
                            best_state[f"B_{blk}"] = lowrank_modules[blk].B.detach().clone()
                            best_state[f"bias_{blk}"] = lowrank_modules[blk].bias.detach().clone()
                    if step % 500 == 0:
                        print(f"  seed {seed} step {step}: ppl={ppl_e:.2f} "
                              f"best={best_ppl:.2f}@{best_step}", flush=True)

        # Restore best
        with torch.no_grad():
            for blk in active_blocks:
                lowrank_modules[blk].B.data.copy_(best_state[f"B_{blk}"])
                lowrank_modules[blk].bias.data.copy_(best_state[f"bias_{blk}"])
        restored_ppl = eval_ppl(model, eval_ids)
        dt = time.time() - t0

        print(f"  seed {seed}: init={init_ppl:.2f} best={best_ppl:.2f}@{best_step} "
              f"restored={restored_ppl:.2f} ({dt:.0f}s)", flush=True)

        seed_results.append({
            "seed": seed,
            "init_ppl": float(init_ppl),
            "best_ppl": float(best_ppl),
            "best_step": int(best_step),
            "restored_ppl": float(restored_ppl),
            "history": history,
        })

        del model, lowrank_modules, optimizer
        if DEVICE == "mps":
            torch.mps.empty_cache()

    # Aggregate stage
    best_ppls = [s["best_ppl"] for s in seed_results]
    best_mean = sum(best_ppls) / len(best_ppls)
    best_std = math.sqrt(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
    init_ppls = [s["init_ppl"] for s in seed_results]
    init_mean = sum(init_ppls) / len(init_ppls)

    stage_data = {
        "n_blocks": n_blocks,
        "blocks": active_blocks,
        "ranks": active_ranks,
        "eff_ranks": active_effranks,
        "trainable_params": trainable_count,
        "init_mean": float(init_mean),
        "best_mean": float(best_mean),
        "best_std": float(best_std),
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
        "seeds": seed_results,
    }
    stages.append(stage_data)

    print(f"\nStage {n_blocks} result: best_mean={best_mean:.2f} +/- {best_std:.2f} "
          f"(init {init_mean:.2f})", flush=True)


# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print("ACCUMULATION CURVE", flush=True)
print(f"{'='*60}", flush=True)
print(f"{'Blocks':>20s}  {'Init':>8s}  {'Best':>8s}  {'Std':>6s}  {'Params':>8s}")
for s in stages:
    print(f"{str(s['blocks']):>20s}  {s['init_mean']:8.2f}  {s['best_mean']:8.2f}  "
          f"{s['best_std']:6.2f}  {s['trainable_params']:8d}")

# Check verdict
single_best = stages[0]["best_mean"]
multi_compose = any(s["best_mean"] <= CONFIRMED_LE for s in stages[1:])
multi_all_degrade = all(s["best_mean"] > single_best for s in stages[1:])

if multi_compose:
    verdict = "CONFIRMED"
elif multi_all_degrade:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nFloat PPL: {float_ppl:.4f}", flush=True)
print(f"Single-block best (stage 1): {single_best:.4f}", flush=True)
print(f"CONFIRMED threshold (<=): {CONFIRMED_LE}", flush=True)
print(f"Multi-block compose (any <=2-block <= {CONFIRMED_LE}): {multi_compose}", flush=True)
print(f"Multi-block all degrade vs single: {multi_all_degrade}", flush=True)
print(f"VERDICT: {verdict}", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 343,
    "item": 352,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "confirmed_le": CONFIRMED_LE,
    "accumulation_order": list(zip(ACCUM_BLOCKS, ACCUM_RANKS, ACCUM_EFFRANKS)),
    "stages": [{k: v for k, v in s.items() if k != "seeds"} for s in stages],
    "stage_seeds": {s["n_blocks"]: s["seeds"] for s in stages},
    "config": {
        "N_CALIB": N_CALIB, "N_EVAL": N_EVAL, "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE, "N_STEPS": N_STEPS, "LR": LR,
        "EVAL_EVERY": EVAL_EVERY, "SEEDS": SEEDS,
    },
    "svd_summary": {
        str(blk): {"rank": svd_cache[blk]["rank"], "S_top5": svd_cache[blk]["S_top5"]}
        for blk in ACCUM_BLOCKS
    },
    "verdict_logic": {
        "single_best": single_best,
        "multi_compose": multi_compose,
        "multi_all_degrade": multi_all_degrade,
        "confirmed_le": CONFIRMED_LE,
        "verdict": verdict,
    },
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)
print("Done.", flush=True)
