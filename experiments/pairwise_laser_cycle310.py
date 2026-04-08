"""
Cycle 310 — Item 320: Pairwise LASER composition matrix over
blocks {1, 2, 3, 10, 11} at the cycle-282-family single-block rank anchors.

For each of C(5, 2) = 10 pairs (l_i, l_j) in [(1,2), (1,3), (1,10), (1,11),
(2,3), (2,10), (2,11), (3,10), (3,11), (10,11)]: install LowRankCProj at
both blocks at their anchor ranks (b1 r=2, b2 r=2, b3 r=24, b10 r=5,
b11 r=22), freeze every other model parameter, joint Adam LR=1e-4,
N_STEPS=2000 (item 320 verbatim: half of cycle 305 budget; pairs are
smaller). SEEDS = [42] only (item 320 budget directive — 10 fits at 3
seeds is too much; extend only if a pair lands below worst-single).

For each pair, measure best_ppl, init_ppl, final_ppl, and report
  delta_vs_worst_single = best_ppl - max(anchor_i, anchor_j)
  pair_ppl_matrix[5, 5] (symmetric; diagonal = single-block anchor)

Item 320 verbatim verdict:
  n_destructive := |{pair : delta_vs_worst_single >= +5 PPL}|
  CONFIRMED    iff n_destructive >= 6
  REFUTED      iff n_destructive <= 2
  INCONCLUSIVE otherwise
"""
import itertools
import json
import math
import random
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ── Config ────────────────────────────────────────────────────────────────────
BLOCKS_RANKS = {1: 2, 2: 2, 3: 24, 10: 5, 11: 22}
BLOCK_ORDER = [1, 2, 3, 10, 11]  # row/col order of the 5x5 matrix
PAIRS = sorted(itertools.combinations(BLOCK_ORDER, 2))  # 10 pairs
assert len(PAIRS) == 10, len(PAIRS)

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 2000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42]  # item 320 verbatim: SEED=42 only for 10 pairs
LOAD_SEED = 42
DESTRUCTIVE_THRESHOLD = 5.0

OUTPUT_PATH = "autoresearch/results/cycle_310_pairwise_laser.json"
SUMMARY_PATH = "autoresearch/results/cycle_310_summary.md"

FLOAT_BASELINE_REF = 53.895

# cycle-282-family single-block anchor best_mean references (diagonal cells)
SINGLE_BLOCK_ANCHORS = {
    1: 49.0324,   # cycle 280 b1 r=2
    2: 49.3290,   # cycle 285 b2 r=2
    3: 50.5935,   # cycle 288 b3 r=24
    10: 51.4659,  # cycle 289 b10 r=5
    11: 54.4687,  # cycle 270 b11 r=22
}

# Prior pair anchors for sanity / comparison (full 4000-step fits)
CYCLE_284_B1_B11_BEST_MEAN = 52.8803   # (1,11) at N_STEPS=4000, 3 seeds
CYCLE_291_B1_B10_BEST_MEAN = 51.8008   # (1,10) at N_STEPS=4000, 3 seeds
CYCLE_305_5STACK_BEST_MEAN = 73.0060   # all 5 blocks together

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)
print(f"BLOCKS_RANKS: {BLOCKS_RANKS}", flush=True)
print(f"PAIRS ({len(PAIRS)}): {PAIRS}", flush=True)

# ── Load model + data ────────────────────────────────────────────────────────
print("\nLoading GPT-2 and data...", flush=True)
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
    """HF Conv1D c_proj rank-r replacement (cycle 270/280/284 template)."""

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


# ── Float baseline + SVD cache (one GPT-2 load) ──────────────────────────────
print("\nFloat baseline PPL...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
base_model.eval()
t0 = time.time()
float_ppl = eval_ppl(base_model, eval_ids)
print(f"Float PPL: {float_ppl:.4f}  ({time.time() - t0:.1f}s)", flush=True)
assert abs(float_ppl - FLOAT_BASELINE_REF) < 0.5, (
    f"Float PPL {float_ppl} differs from ref {FLOAT_BASELINE_REF} by > 0.5"
)

svd_cache = {}
for blk, r in BLOCKS_RANKS.items():
    print(f"\nSVD of block {blk} c_proj.weight (target rank {r})...", flush=True)
    W_orig = base_model.transformer.h[blk].mlp.c_proj.weight.data.detach().clone()
    b_orig = base_model.transformer.h[blk].mlp.c_proj.bias.data.detach().clone()
    assert W_orig.shape == (3072, 768), (blk, W_orig.shape)
    W_cpu = W_orig.float().cpu()
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    print(f"  SVD done in {time.time() - t0:.1f}s; "
          f"U {tuple(U.shape)}, S {tuple(S.shape)}, Vh {tuple(Vh.shape)}", flush=True)
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

del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# ── Per-pair joint Adam retrain (SEED=42 only) ───────────────────────────────
pair_results = []  # list of dicts, one per pair
wall_start = time.time()

for pair_idx, pair in enumerate(PAIRS):
    blk_i, blk_j = pair
    r_i = BLOCKS_RANKS[blk_i]
    r_j = BLOCKS_RANKS[blk_j]
    expected_trainable = (r_i * 768 + 768) + (r_j * 768 + 768)
    print(f"\n=== Pair {pair_idx + 1}/{len(PAIRS)}: "
          f"b{blk_i} r={r_i} + b{blk_j} r={r_j}  "
          f"(trainable {expected_trainable:,}) ===", flush=True)

    seeds_for_this_pair = []
    for seed in SEEDS:
        print(f"\n--- pair ({blk_i},{blk_j}) seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        random.seed(seed)

        model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model_s.eval()

        lowrank_modules = {}
        for blk in pair:
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
        print(f"  init PPL (pair no-retrain): {init_ppl:.4f}", flush=True)

        for p in model_s.parameters():
            p.requires_grad_(False)
        trainable = []
        for blk in pair:
            lowrank_modules[blk].B.requires_grad_(True)
            lowrank_modules[blk].bias.requires_grad_(True)
            trainable.append(lowrank_modules[blk].B)
            trainable.append(lowrank_modules[blk].bias)
        n_trainable = sum(p.numel() for p in trainable)
        assert n_trainable == expected_trainable, (
            f"pair {pair} trainable count {n_trainable} != {expected_trainable}"
        )

        optimizer = torch.optim.Adam(trainable, lr=LR)

        calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids)

        best_ppl = init_ppl
        best_step = 0
        best_state = {
            f"B_{blk}": lowrank_modules[blk].B.detach().clone() for blk in pair
        }
        for blk in pair:
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
                            for blk in pair
                        }
                        for blk in pair:
                            best_state[f"bias_{blk}"] = (
                                lowrank_modules[blk].bias.detach().clone()
                            )
                    if step % 500 == 0:
                        print(f"    step {step:4d}: eval_ppl={ppl_e:.4f}  "
                              f"best={best_ppl:.4f}@{best_step}", flush=True)

        final_ppl = eval_ppl(model_s, eval_ids)
        with torch.no_grad():
            for blk in pair:
                lowrank_modules[blk].B.data.copy_(best_state[f"B_{blk}"])
                lowrank_modules[blk].bias.data.copy_(best_state[f"bias_{blk}"])
        restored_ppl = eval_ppl(model_s, eval_ids)
        assert abs(restored_ppl - best_ppl) < 1e-3, (
            f"pair {pair} seed {seed}: restored {restored_ppl} != best {best_ppl}"
        )
        dt = time.time() - t_train
        print(f"  pair ({blk_i},{blk_j}) seed {seed}: init={init_ppl:.4f} "
              f"best={best_ppl:.4f}@{best_step} final@{N_STEPS}={final_ppl:.4f} "
              f"restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

        seeds_for_this_pair.append({
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

    # pair aggregate over seeds (n=1 for now)
    best_ppls = [s["best_ppl"] for s in seeds_for_this_pair]
    best_mean = float(sum(best_ppls) / len(best_ppls))
    if len(best_ppls) > 1:
        best_var = float(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
        best_std = math.sqrt(best_var)
    else:
        best_std = 0.0
    init_ppls = [s["init_ppl"] for s in seeds_for_this_pair]
    final_ppls = [s["final_ppl_at_n_steps"] for s in seeds_for_this_pair]

    anchor_i = SINGLE_BLOCK_ANCHORS[blk_i]
    anchor_j = SINGLE_BLOCK_ANCHORS[blk_j]
    worst_single = max(anchor_i, anchor_j)
    best_single = min(anchor_i, anchor_j)
    delta_vs_worst = best_mean - worst_single
    delta_vs_best_single = best_mean - best_single
    destructive = delta_vs_worst >= DESTRUCTIVE_THRESHOLD

    print(f"\n  ==> pair ({blk_i},{blk_j}) best_mean = {best_mean:.4f}  "
          f"worst_single = {worst_single:.4f}  "
          f"delta = {delta_vs_worst:+.4f}  "
          f"destructive={destructive}", flush=True)

    pair_results.append({
        "pair": [blk_i, blk_j],
        "ranks": [r_i, r_j],
        "trainable_params": expected_trainable,
        "seeds": seeds_for_this_pair,
        "n_seeds": len(seeds_for_this_pair),
        "init_mean": float(sum(init_ppls) / len(init_ppls)),
        "best_mean": best_mean,
        "best_std": best_std,
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
        "final_mean": float(sum(final_ppls) / len(final_ppls)),
        "anchor_i": anchor_i,
        "anchor_j": anchor_j,
        "worst_single": worst_single,
        "best_single": best_single,
        "delta_vs_worst_single": delta_vs_worst,
        "delta_vs_best_single": delta_vs_best_single,
        "destructive_at_5ppl": bool(destructive),
    })

wall_min = (time.time() - wall_start) / 60
print(f"\nTotal pair sweep wall: {wall_min:.1f} min", flush=True)


# ── Build the 5x5 matrix ─────────────────────────────────────────────────────
n = len(BLOCK_ORDER)
idx = {blk: i for i, blk in enumerate(BLOCK_ORDER)}
pair_ppl_matrix = [[None for _ in range(n)] for _ in range(n)]
for i, blk in enumerate(BLOCK_ORDER):
    pair_ppl_matrix[i][i] = SINGLE_BLOCK_ANCHORS[blk]
for pr in pair_results:
    bi, bj = pr["pair"]
    i, j = idx[bi], idx[bj]
    pair_ppl_matrix[i][j] = pr["best_mean"]
    pair_ppl_matrix[j][i] = pr["best_mean"]

# symmetry + diagonal sanity
for i in range(n):
    for j in range(n):
        assert pair_ppl_matrix[i][j] is not None, (i, j)
        if i != j:
            assert abs(pair_ppl_matrix[i][j] - pair_ppl_matrix[j][i]) < 1e-9


# ── Verdict ───────────────────────────────────────────────────────────────────
n_destructive = sum(1 for pr in pair_results if pr["destructive_at_5ppl"])
if n_destructive >= 6:
    verdict = "CONFIRMED"
elif n_destructive <= 2:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

worst_pair_entry = max(pair_results, key=lambda pr: pr["delta_vs_worst_single"])
best_pair_entry = min(pair_results, key=lambda pr: pr["delta_vs_worst_single"])

print(f"\n=== Verdict ===", flush=True)
print(f"  n_destructive (delta >= +{DESTRUCTIVE_THRESHOLD} PPL): "
      f"{n_destructive} / {len(PAIRS)}", flush=True)
print(f"  CONFIRMED iff n >= 6; REFUTED iff n <= 2", flush=True)
print(f"  VERDICT: {verdict}", flush=True)
print(f"  worst pair: {worst_pair_entry['pair']} "
      f"(best_mean {worst_pair_entry['best_mean']:.4f}, "
      f"delta {worst_pair_entry['delta_vs_worst_single']:+.4f})", flush=True)
print(f"  best pair: {best_pair_entry['pair']} "
      f"(best_mean {best_pair_entry['best_mean']:.4f}, "
      f"delta {best_pair_entry['delta_vs_worst_single']:+.4f})", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 310,
    "item": 320,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "block_order": BLOCK_ORDER,
    "blocks_ranks": {str(k): v for k, v in BLOCKS_RANKS.items()},
    "pairs": [list(p) for p in PAIRS],
    "pair_results": pair_results,
    "pair_ppl_matrix": pair_ppl_matrix,
    "single_block_anchors": {str(k): v for k, v in SINGLE_BLOCK_ANCHORS.items()},
    "n_destructive": n_destructive,
    "destructive_threshold_ppl": DESTRUCTIVE_THRESHOLD,
    "verdict_logic": {
        "n_destructive": n_destructive,
        "threshold_confirmed_ge": 6,
        "threshold_refuted_le": 2,
        "verdict": verdict,
        "worst_pair": worst_pair_entry["pair"],
        "worst_pair_best_mean": worst_pair_entry["best_mean"],
        "worst_pair_delta": worst_pair_entry["delta_vs_worst_single"],
        "best_pair": best_pair_entry["pair"],
        "best_pair_best_mean": best_pair_entry["best_mean"],
        "best_pair_delta": best_pair_entry["delta_vs_worst_single"],
    },
    "config": {
        "BLOCKS_RANKS": {str(k): v for k, v in BLOCKS_RANKS.items()},
        "BLOCK_ORDER": BLOCK_ORDER,
        "N_CALIB": N_CALIB,
        "N_EVAL": N_EVAL,
        "SEQ_LEN": SEQ_LEN,
        "DEVICE": DEVICE,
        "N_STEPS": N_STEPS,
        "LR": LR,
        "EVAL_EVERY": EVAL_EVERY,
        "SEEDS": SEEDS,
    },
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "cycle_284_b1_b11_best_mean": CYCLE_284_B1_B11_BEST_MEAN,
        "cycle_291_b1_b10_best_mean": CYCLE_291_B1_B10_BEST_MEAN,
        "cycle_305_5stack_best_mean": CYCLE_305_5STACK_BEST_MEAN,
    },
    "wall_min": wall_min,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# ── Summary markdown ──────────────────────────────────────────────────────────
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append("# Cycle 310 — Item 320: Pairwise LASER composition matrix")
lines.append("")
lines.append(f"**Verdict: {verdict}**  "
             f"(n_destructive = {n_destructive} / {len(PAIRS)}; "
             f"CONFIRMED iff ≥6, REFUTED iff ≤2)")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Block order: {BLOCK_ORDER}")
lines.append(f"- Ranks: {BLOCKS_RANKS}")
lines.append(f"- N_STEPS={N_STEPS}, LR={LR}, SEEDS={SEEDS}, EVAL_EVERY={EVAL_EVERY}")
lines.append(f"- Pair sweep wall: {wall_min:.1f} min")
lines.append("")
lines.append("## 5×5 pair PPL matrix (diagonal = single-block anchor)")
lines.append("")
header = "| block |" + "|".join(f" {b} " for b in BLOCK_ORDER) + "|"
sep = "|---|" + "|".join("---" for _ in BLOCK_ORDER) + "|"
lines.append(header)
lines.append(sep)
for i, blk in enumerate(BLOCK_ORDER):
    row = f"| **{blk}** |"
    for j in range(n):
        cell = pair_ppl_matrix[i][j]
        if i == j:
            row += f" *{fmt(cell)}* |"  # italic for diagonal
        else:
            row += f" {fmt(cell)} |"
    lines.append(row)
lines.append("")
lines.append("## Per-pair detail (sorted by delta_vs_worst_single)")
lines.append("")
lines.append("| pair | ranks | init | best | delta_vs_worst_single | "
             "delta_vs_best_single | destructive |")
lines.append("|---|---|---|---|---|---|---|")
for pr in sorted(pair_results, key=lambda x: x["delta_vs_worst_single"]):
    bi, bj = pr["pair"]
    ri, rj = pr["ranks"]
    lines.append(
        f"| ({bi},{bj}) | ({ri},{rj}) | "
        f"{fmt(pr['init_mean'])} | {fmt(pr['best_mean'])} | "
        f"{pr['delta_vs_worst_single']:+.4f} | "
        f"{pr['delta_vs_best_single']:+.4f} | "
        f"{'yes' if pr['destructive_at_5ppl'] else 'no'} |"
    )
lines.append("")
lines.append("## Verdict logic (item 320 verbatim)")
lines.append("")
lines.append(f"- n_destructive := |{{pair : delta_vs_worst_single ≥ +{DESTRUCTIVE_THRESHOLD}}}| "
             f"= **{n_destructive}** / {len(PAIRS)}")
lines.append(f"- CONFIRMED iff n ≥ 6  →  {n_destructive >= 6}")
lines.append(f"- REFUTED   iff n ≤ 2  →  {n_destructive <= 2}")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append(f"- worst pair: **{worst_pair_entry['pair']}** "
             f"best_mean={fmt(worst_pair_entry['best_mean'])} "
             f"delta={worst_pair_entry['delta_vs_worst_single']:+.4f}")
lines.append(f"- best pair:  **{best_pair_entry['pair']}** "
             f"best_mean={fmt(best_pair_entry['best_mean'])} "
             f"delta={best_pair_entry['delta_vs_worst_single']:+.4f}")
lines.append("")
lines.append("## Comparisons to prior cycles")
lines.append("")
lines.append(f"- cycle 284 (b1,b11) 4000-step: {CYCLE_284_B1_B11_BEST_MEAN}")
lines.append(f"- cycle 291 (b1,b10) 4000-step: {CYCLE_291_B1_B10_BEST_MEAN}")
lines.append(f"- cycle 305 5-stack:           {CYCLE_305_5STACK_BEST_MEAN}")
lines.append("")
lines.append("Note: cycle 310 uses N_STEPS=2000 (item 320 budget), cycles 284/291 used 4000; ")
lines.append("cycle 310 per-pair best_ppl is expected to be slightly worse than the ")
lines.append("matched-anchor 4000-step fits of 284/291 at the same pair. Sanity target: ")
lines.append("cycle 310 (1,11) best_mean close to cycle 284 52.8803 ± a few PPL (budget).")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)

print("\nDone.", flush=True)
