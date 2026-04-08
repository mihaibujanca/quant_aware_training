"""
Cycle 314 — Item 324: Rank-1 pairwise LASER composition matrix over
blocks {1, 2, 3, 10, 11} with EVERY block at rank-1.

Extends cycle 310 (heterogeneous cycle-282-family ranks) to the universal
rank-1 level established by cycle 311. For each of C(5, 2) = 10 pairs:
install LowRankCProj(r=1) at both blocks, freeze every other model
parameter, joint Adam LR=1e-4, N_STEPS=2000, EVAL_EVERY=100, best-state
restore, SEEDS=[42] (item 324 verbatim: single seed to match cycle 310
budget; extend only if the matrix shows new destructive cells).

For each pair, measure best_ppl, init_ppl, final_ppl, and report
  delta_vs_worst_single_r1 = best_ppl - max(r1_anchor_i, r1_anchor_j)
  pair_ppl_matrix[5, 5] (symmetric; diagonal = cycle 292/311 r=1 anchors)

Trainable per pair = 2 * (1 * 768 + 768) = 3,072 (~6x smaller than
cycle 310's heterogeneous pairs).

Item 324 verbatim verdict:
  CONFIRMED iff (b10, b11) r=1 delta_vs_worst_single >= +5 PPL
             ("destruction is basis-geometric, survives r=1 reduction")
  REFUTED   iff (b10, b11) r=1 delta_vs_worst_single <= +1 PPL
             AND n_destructive_other_pairs <= 1
             ("cycle 310's destruction was capacity competition at r=22")
  INCONCLUSIVE otherwise.
Secondary: n_destructive_at_5 across all 10 pairs.
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
BLOCK_ORDER = [1, 2, 3, 10, 11]
BLOCKS_RANKS = {b: 1 for b in BLOCK_ORDER}  # every block at rank-1
PAIRS = sorted(itertools.combinations(BLOCK_ORDER, 2))
assert len(PAIRS) == 10, len(PAIRS)

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 2000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42]  # item 324 verbatim: single seed to match cycle 310 budget
LOAD_SEED = 42
DESTRUCTIVE_THRESHOLD = 5.0

OUTPUT_PATH = "autoresearch/results/cycle_314_rank1_pairwise_laser.json"
SUMMARY_PATH = "autoresearch/results/cycle_314_summary.md"

FLOAT_BASELINE_REF = 53.895

# Single-block r=1 anchors (diagonal of the 5x5 matrix)
# b1 from cycle 292 (rank1_all_concentrators was only b2/b3/b10/b11)
# b2, b3, b10, b11 from cycle 311
SINGLE_BLOCK_R1_ANCHORS = {
    1: 50.0794,   # cycle 292 b1 r=1
    2: 49.7539,   # cycle 311 b2 r=1
    3: 55.0485,   # cycle 311 b3 r=1
    10: 52.9107,  # cycle 311 b10 r=1
    11: 59.7074,  # cycle 311 b11 r=1
}

# Cycle 310 heterogeneous pair references for sanity comparison
CYCLE_310_PAIR_RESULTS = {
    (1, 2): 54.2029,    # (+4.8739)
    (1, 3): 51.6544,    # (+1.0609)
    (1, 10): 52.1901,   # (+0.7242)
    (1, 11): 53.5812,   # (-0.8875 BEST PAIR)
    (2, 3): 52.3641,    # (+1.7706)
    (2, 10): 52.9059,   # (+1.4400)
    (2, 11): 53.8455,   # (-0.6232)
    (3, 10): 52.2278,   # (+0.7619)
    (3, 11): 54.9010,   # (+0.4323)
    (10, 11): 60.7161,  # (+6.2474 ONLY DESTRUCTIVE)
}

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
    """HF Conv1D c_proj rank-r replacement (cycle 270/280/284/292/310/311 template)."""

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
    assert r == 1, f"block {blk} rank {r} != 1"
    print(f"\nSVD of block {blk} c_proj.weight (target rank {r})...", flush=True)
    W_orig = base_model.transformer.h[blk].mlp.c_proj.weight.data.detach().clone()
    b_orig = base_model.transformer.h[blk].mlp.c_proj.bias.data.detach().clone()
    assert W_orig.shape == (3072, 768), (blk, W_orig.shape)
    W_cpu = W_orig.float().cpu()
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    print(f"  SVD done in {time.time() - t0:.1f}s; "
          f"U {tuple(U.shape)}, S {tuple(S.shape)}, Vh {tuple(Vh.shape)}  "
          f"S[0]={float(S[0]):.4f}", flush=True)
    A_r = (U[:, :r] * S[:r]).contiguous()  # (3072, 1)
    B_r = Vh[:r, :].contiguous()           # (1, 768)
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
pair_results = []
wall_start = time.time()

for pair_idx, pair in enumerate(PAIRS):
    blk_i, blk_j = pair
    r_i = BLOCKS_RANKS[blk_i]
    r_j = BLOCKS_RANKS[blk_j]
    expected_trainable = (r_i * 768 + 768) + (r_j * 768 + 768)
    assert expected_trainable == 3072, expected_trainable
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

    anchor_i = SINGLE_BLOCK_R1_ANCHORS[blk_i]
    anchor_j = SINGLE_BLOCK_R1_ANCHORS[blk_j]
    worst_single = max(anchor_i, anchor_j)
    best_single = min(anchor_i, anchor_j)
    delta_vs_worst = best_mean - worst_single
    delta_vs_best_single = best_mean - best_single
    destructive = delta_vs_worst >= DESTRUCTIVE_THRESHOLD

    cycle_310_ref = CYCLE_310_PAIR_RESULTS[pair]
    delta_vs_c310 = best_mean - cycle_310_ref

    print(f"\n  ==> pair ({blk_i},{blk_j}) best_mean = {best_mean:.4f}  "
          f"worst_single_r1 = {worst_single:.4f}  "
          f"delta = {delta_vs_worst:+.4f}  "
          f"destructive={destructive}  "
          f"vs c310: {delta_vs_c310:+.4f}", flush=True)

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
        "r1_anchor_i": anchor_i,
        "r1_anchor_j": anchor_j,
        "worst_single_r1": worst_single,
        "best_single_r1": best_single,
        "delta_vs_worst_single_r1": delta_vs_worst,
        "delta_vs_best_single_r1": delta_vs_best_single,
        "destructive_at_5ppl": bool(destructive),
        "cycle_310_best_mean": cycle_310_ref,
        "delta_vs_cycle_310": delta_vs_c310,
    })

wall_min = (time.time() - wall_start) / 60
print(f"\nTotal pair sweep wall: {wall_min:.1f} min", flush=True)


# ── Build the 5x5 matrix ─────────────────────────────────────────────────────
n = len(BLOCK_ORDER)
idx = {blk: i for i, blk in enumerate(BLOCK_ORDER)}
pair_ppl_matrix = [[None for _ in range(n)] for _ in range(n)]
for i, blk in enumerate(BLOCK_ORDER):
    pair_ppl_matrix[i][i] = SINGLE_BLOCK_R1_ANCHORS[blk]
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


# ── Verdict per item 324 verbatim ─────────────────────────────────────────────
b10_b11_pair = next(pr for pr in pair_results if pr["pair"] == [10, 11])
b10_b11_delta = b10_b11_pair["delta_vs_worst_single_r1"]

n_destructive = sum(1 for pr in pair_results if pr["destructive_at_5ppl"])
# Other pairs (excluding b10,b11)
n_other_destructive = sum(
    1 for pr in pair_results
    if pr["destructive_at_5ppl"] and pr["pair"] != [10, 11]
)

# Item 324 verbatim logic:
#   CONFIRMED iff (b10, b11) delta >= +5 PPL
#   REFUTED iff (b10, b11) delta <= +1 PPL AND other destructive <= 1
#   INCONCLUSIVE otherwise
if b10_b11_delta >= DESTRUCTIVE_THRESHOLD:
    verdict = "CONFIRMED"
    verdict_label = "CONFIRMED-persistent-destruction"
elif b10_b11_delta <= 1.0 and n_other_destructive <= 1:
    verdict = "REFUTED"
    verdict_label = "REFUTED-at-r1"
else:
    verdict = "INCONCLUSIVE"
    verdict_label = "INCONCLUSIVE"

worst_pair_entry = max(pair_results, key=lambda pr: pr["delta_vs_worst_single_r1"])
best_pair_entry = min(pair_results, key=lambda pr: pr["delta_vs_worst_single_r1"])

print(f"\n=== Verdict ===", flush=True)
print(f"  (b10, b11) r=1 best_mean = {b10_b11_pair['best_mean']:.4f}  "
      f"worst_single_r1 = {b10_b11_pair['worst_single_r1']:.4f}  "
      f"delta = {b10_b11_delta:+.4f}", flush=True)
print(f"  n_destructive_at_5 (all) = {n_destructive} / {len(PAIRS)}", flush=True)
print(f"  n_destructive_other (excluding b10,b11) = {n_other_destructive} / 9", flush=True)
print(f"  CONFIRMED iff (b10,b11) delta >= +{DESTRUCTIVE_THRESHOLD}", flush=True)
print(f"  REFUTED iff (b10,b11) delta <= +1.0 AND n_other_destructive <= 1", flush=True)
print(f"  VERDICT: {verdict} ({verdict_label})", flush=True)
print(f"  worst pair: {worst_pair_entry['pair']} "
      f"(best_mean {worst_pair_entry['best_mean']:.4f}, "
      f"delta {worst_pair_entry['delta_vs_worst_single_r1']:+.4f})", flush=True)
print(f"  best pair: {best_pair_entry['pair']} "
      f"(best_mean {best_pair_entry['best_mean']:.4f}, "
      f"delta {best_pair_entry['delta_vs_worst_single_r1']:+.4f})", flush=True)


# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "cycle": 314,
    "item": 324,
    "verdict": verdict,
    "verdict_label": verdict_label,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "block_order": BLOCK_ORDER,
    "blocks_ranks": {str(k): v for k, v in BLOCKS_RANKS.items()},
    "pairs": [list(p) for p in PAIRS],
    "pair_results": pair_results,
    "pair_ppl_matrix": pair_ppl_matrix,
    "single_block_r1_anchors": {str(k): v for k, v in SINGLE_BLOCK_R1_ANCHORS.items()},
    "n_destructive_at_5": n_destructive,
    "n_destructive_other_pairs": n_other_destructive,
    "destructive_threshold_ppl": DESTRUCTIVE_THRESHOLD,
    "verdict_logic": {
        "b10_b11_delta_vs_worst_single_r1": b10_b11_delta,
        "b10_b11_best_mean": b10_b11_pair["best_mean"],
        "b10_b11_worst_single_r1": b10_b11_pair["worst_single_r1"],
        "n_destructive_all": n_destructive,
        "n_destructive_other": n_other_destructive,
        "confirmed_bar": f"(b10, b11) delta >= +{DESTRUCTIVE_THRESHOLD}",
        "refuted_bar": "(b10, b11) delta <= +1.0 AND n_other_destructive <= 1",
        "verdict": verdict,
        "verdict_label": verdict_label,
        "worst_pair": worst_pair_entry["pair"],
        "worst_pair_best_mean": worst_pair_entry["best_mean"],
        "worst_pair_delta": worst_pair_entry["delta_vs_worst_single_r1"],
        "best_pair": best_pair_entry["pair"],
        "best_pair_best_mean": best_pair_entry["best_mean"],
        "best_pair_delta": best_pair_entry["delta_vs_worst_single_r1"],
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
        "trainable_per_pair": 3072,
    },
    "baselines": {
        "float": FLOAT_BASELINE_REF,
        "b1_r1_cycle292": 50.0794,
        "b2_r1_cycle311": 49.7539,
        "b3_r1_cycle311": 55.0485,
        "b10_r1_cycle311": 52.9107,
        "b11_r1_cycle311": 59.7074,
        "cycle_310_b10_b11_heterogeneous": 60.7161,
        "cycle_310_n_destructive": 1,
        "cycle_305_5stack_best_mean": 73.0060,
    },
    "cycle_310_comparison": {
        f"({p[0]},{p[1]})": {
            "cycle_310_best_mean": CYCLE_310_PAIR_RESULTS[tuple(p)],
            "cycle_314_best_mean": next(
                pr["best_mean"] for pr in pair_results if pr["pair"] == list(p)
            ),
            "delta_314_minus_310": (
                next(pr["best_mean"] for pr in pair_results if pr["pair"] == list(p))
                - CYCLE_310_PAIR_RESULTS[tuple(p)]
            ),
        }
        for p in PAIRS
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
lines.append("# Cycle 314 — Item 324: Rank-1 pairwise LASER composition matrix")
lines.append("")
lines.append(f"**Verdict: {verdict}** ({verdict_label})")
lines.append("")
lines.append(f"- (b10, b11) r=1 delta_vs_worst_single_r1 = **{b10_b11_delta:+.4f}**  "
             f"(best_mean {b10_b11_pair['best_mean']:.4f} vs worst_single_r1 "
             f"{b10_b11_pair['worst_single_r1']:.4f})")
lines.append(f"- n_destructive_at_5 (all pairs) = {n_destructive} / {len(PAIRS)}")
lines.append(f"- n_destructive_other (excluding b10,b11) = {n_other_destructive} / 9")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Block order: {BLOCK_ORDER}")
lines.append(f"- All blocks at rank-1; trainable per pair = 3,072 = 2 × (1·768 + 768)")
lines.append(f"- N_STEPS={N_STEPS}, LR={LR}, SEEDS={SEEDS}, EVAL_EVERY={EVAL_EVERY}")
lines.append(f"- Pair sweep wall: {wall_min:.1f} min")
lines.append("")
lines.append("## 5×5 pair PPL matrix (diagonal = cycle 292/311 r=1 anchors)")
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
lines.append("## Per-pair detail (sorted by delta_vs_worst_single_r1)")
lines.append("")
lines.append("| pair | init | best | delta_vs_worst_r1 | delta_vs_best_r1 | "
             "destructive | cycle310 ref | delta_314-310 |")
lines.append("|---|---|---|---|---|---|---|---|")
for pr in sorted(pair_results, key=lambda x: x["delta_vs_worst_single_r1"]):
    bi, bj = pr["pair"]
    lines.append(
        f"| ({bi},{bj}) | {fmt(pr['init_mean'])} | {fmt(pr['best_mean'])} | "
        f"{pr['delta_vs_worst_single_r1']:+.4f} | "
        f"{pr['delta_vs_best_single_r1']:+.4f} | "
        f"{'yes' if pr['destructive_at_5ppl'] else 'no'} | "
        f"{fmt(pr['cycle_310_best_mean'])} | "
        f"{pr['delta_vs_cycle_310']:+.4f} |"
    )
lines.append("")
lines.append("## Verdict logic (item 324 verbatim)")
lines.append("")
lines.append(f"- CONFIRMED iff (b10, b11) r=1 delta_vs_worst_single ≥ +{DESTRUCTIVE_THRESHOLD} "
             f"→ {b10_b11_delta >= DESTRUCTIVE_THRESHOLD}")
lines.append(f"- REFUTED iff (b10, b11) r=1 delta_vs_worst_single ≤ +1.0 "
             f"AND n_other_destructive ≤ 1 "
             f"→ {b10_b11_delta <= 1.0 and n_other_destructive <= 1}")
lines.append(f"- VERDICT: **{verdict}** ({verdict_label})")
lines.append("")
lines.append(f"- worst pair: **{worst_pair_entry['pair']}** "
             f"best_mean={fmt(worst_pair_entry['best_mean'])} "
             f"delta={worst_pair_entry['delta_vs_worst_single_r1']:+.4f}")
lines.append(f"- best pair:  **{best_pair_entry['pair']}** "
             f"best_mean={fmt(best_pair_entry['best_mean'])} "
             f"delta={best_pair_entry['delta_vs_worst_single_r1']:+.4f}")
lines.append("")
lines.append("## Cycle 310 comparison (heterogeneous r vs universal r=1)")
lines.append("")
lines.append("| pair | cycle 310 (het r) | cycle 314 (r=1) | delta |")
lines.append("|---|---|---|---|")
for pr in pair_results:
    bi, bj = pr["pair"]
    lines.append(
        f"| ({bi},{bj}) | {fmt(pr['cycle_310_best_mean'])} | "
        f"{fmt(pr['best_mean'])} | {pr['delta_vs_cycle_310']:+.4f} |"
    )
lines.append("")
lines.append("## Single-block r=1 anchors (diagonal)")
lines.append("")
for blk in BLOCK_ORDER:
    src = "cycle 292" if blk == 1 else "cycle 311"
    lines.append(f"- b{blk} r=1: {SINGLE_BLOCK_R1_ANCHORS[blk]} ({src})")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)

print("\nDone.", flush=True)
