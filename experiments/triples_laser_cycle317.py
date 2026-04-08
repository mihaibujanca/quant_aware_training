"""
Cycle 317 — Item 327: 3-block LASER composition over
blocks {1, 2, 3, 10, 11} at the cycle-282-family single-block rank anchors.

For each of C(5, 3) = 10 triples from [1, 2, 3, 10, 11] at ranks
BLOCKS_RANKS = {1: 2, 2: 2, 3: 24, 10: 5, 11: 22}, install 3 LowRankCProj
modules simultaneously, freeze every other model parameter, joint Adam
LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore, SEEDS=[42].

Report per triple: best_mean, init_ppl, delta_vs_worst_single, destructive
flag, plus pair-inheritance info (which cycle 310 pair deltas the triple
contains and whether (b10, b11) is among them).

Item 327 verbatim verdict:
  (all 4 (b10,b11)-containing triples destruct AND all 6 non-(b10,b11)
   triples delta <= +2 PPL) -> CONFIRMED-b10b11-carries-destruction
  n_destructive >= 8 / 10 -> CONFIRMED-triple-level-pervasive
  n_destructive <= 2 / 10 -> REFUTED
  else -> INCONCLUSIVE
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

# - Config ----------------------------------------------------------------------
BLOCKS_RANKS = {1: 2, 2: 2, 3: 24, 10: 5, 11: 22}
BLOCK_ORDER = [1, 2, 3, 10, 11]
TRIPLES = sorted(itertools.combinations(BLOCK_ORDER, 3))
assert len(TRIPLES) == 10, len(TRIPLES)

N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_STEPS = 4000
LR = 1e-4
EVAL_EVERY = 100
SEEDS = [42]
LOAD_SEED = 42
DESTRUCTIVE_THRESHOLD = 5.0
NONDESTRUCTIVE_THRESHOLD = 2.0

OUTPUT_PATH = "autoresearch/results/cycle_317_triples.json"
SUMMARY_PATH = "autoresearch/results/cycle_317_summary.md"

FLOAT_BASELINE_REF = 53.895

# cycle-282-family single-block anchor best_mean references
SINGLE_BLOCK_ANCHORS = {
    1: 49.0324,   # cycle 280 b1 r=2
    2: 49.3290,   # cycle 285 b2 r=2
    3: 50.5935,   # cycle 288 b3 r=24
    10: 51.4659,  # cycle 289 b10 r=5
    11: 54.4687,  # cycle 270 b11 r=22
}

# cycle 310 pair best_mean values (diagonal = single-block anchors above)
CYCLE_310_PAIR_BEST_MEAN = {
    (1, 2): 54.2029,
    (1, 3): 51.6544,
    (1, 10): 52.1901,
    (1, 11): 53.5812,
    (2, 3): 52.3641,
    (2, 10): 52.9059,
    (2, 11): 53.8455,
    (3, 10): 52.2278,
    (3, 11): 54.9010,
    (10, 11): 60.7161,
}

# cycle 305 5-stack best_mean
CYCLE_305_5STACK_BEST_MEAN = 73.0060

torch.manual_seed(LOAD_SEED)
print(f"Device: {DEVICE}", flush=True)
print(f"BLOCKS_RANKS: {BLOCKS_RANKS}", flush=True)
print(f"TRIPLES ({len(TRIPLES)}): {TRIPLES}", flush=True)

# - Load model + data -----------------------------------------------------------
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


# - Helpers ---------------------------------------------------------------------
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
    """HF Conv1D c_proj rank-r replacement (cycle 270/280/284/310 template)."""

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


# - Float baseline + SVD cache --------------------------------------------------
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
        "rank": r,
    }

del base_model
if DEVICE == "mps":
    torch.mps.empty_cache()


# - Per-triple joint Adam retrain (SEED=42 only) --------------------------------
triple_results = []
wall_start = time.time()

for triple_idx, triple in enumerate(TRIPLES):
    blks = list(triple)
    rs = [BLOCKS_RANKS[b] for b in blks]
    expected_trainable = sum((r + 1) * 768 for r in rs)
    print(f"\n=== Triple {triple_idx + 1}/{len(TRIPLES)}: "
          f"{tuple(f'b{b} r={r}' for b, r in zip(blks, rs))}  "
          f"(trainable {expected_trainable:,}) ===", flush=True)

    seeds_for_this_triple = []
    for seed in SEEDS:
        print(f"\n--- triple {triple} seed {seed} ---", flush=True)
        torch.manual_seed(seed)
        random.seed(seed)

        model_s = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        model_s.eval()

        lowrank_modules = {}
        for blk in blks:
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
        print(f"  init PPL (triple no-retrain): {init_ppl:.4f}", flush=True)

        for p in model_s.parameters():
            p.requires_grad_(False)
        trainable = []
        for blk in blks:
            lowrank_modules[blk].B.requires_grad_(True)
            lowrank_modules[blk].bias.requires_grad_(True)
            trainable.append(lowrank_modules[blk].B)
            trainable.append(lowrank_modules[blk].bias)
        n_trainable = sum(p.numel() for p in trainable)
        assert n_trainable == expected_trainable, (
            f"triple {triple} trainable count {n_trainable} != {expected_trainable}"
        )

        optimizer = torch.optim.Adam(trainable, lr=LR)

        calib_ids = [t.to(DEVICE) for t in calib_ids_cpu]
        random.shuffle(calib_ids)

        best_ppl = init_ppl
        best_step = 0
        best_state = {
            f"B_{blk}": lowrank_modules[blk].B.detach().clone() for blk in blks
        }
        for blk in blks:
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
                            for blk in blks
                        }
                        for blk in blks:
                            best_state[f"bias_{blk}"] = (
                                lowrank_modules[blk].bias.detach().clone()
                            )
                    if step % 500 == 0:
                        print(f"    step {step:4d}: eval_ppl={ppl_e:.4f}  "
                              f"best={best_ppl:.4f}@{best_step}", flush=True)

        final_ppl = eval_ppl(model_s, eval_ids)
        with torch.no_grad():
            for blk in blks:
                lowrank_modules[blk].B.data.copy_(best_state[f"B_{blk}"])
                lowrank_modules[blk].bias.data.copy_(best_state[f"bias_{blk}"])
        restored_ppl = eval_ppl(model_s, eval_ids)
        assert abs(restored_ppl - best_ppl) < 1e-3, (
            f"triple {triple} seed {seed}: restored {restored_ppl} != best {best_ppl}"
        )
        dt = time.time() - t_train
        print(f"  triple {triple} seed {seed}: init={init_ppl:.4f} "
              f"best={best_ppl:.4f}@{best_step} final@{N_STEPS}={final_ppl:.4f} "
              f"restored={restored_ppl:.4f}  [{dt/60:.1f} min]", flush=True)

        seeds_for_this_triple.append({
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

    # aggregate
    best_ppls = [s["best_ppl"] for s in seeds_for_this_triple]
    best_mean = float(sum(best_ppls) / len(best_ppls))
    if len(best_ppls) > 1:
        best_var = float(sum((x - best_mean) ** 2 for x in best_ppls) / len(best_ppls))
        best_std = math.sqrt(best_var)
    else:
        best_std = 0.0
    init_ppls = [s["init_ppl"] for s in seeds_for_this_triple]
    final_ppls = [s["final_ppl_at_n_steps"] for s in seeds_for_this_triple]

    anchors = [SINGLE_BLOCK_ANCHORS[b] for b in blks]
    worst_single = max(anchors)
    best_single = min(anchors)
    delta_vs_worst = best_mean - worst_single
    delta_vs_best_single = best_mean - best_single
    destructive = delta_vs_worst >= DESTRUCTIVE_THRESHOLD
    nondestructive_strict = delta_vs_worst <= NONDESTRUCTIVE_THRESHOLD

    # pair inheritance info (cycle 310 sub-pair deltas)
    sub_pairs = [tuple(sorted(p)) for p in itertools.combinations(blks, 2)]
    sub_pair_best_means = [CYCLE_310_PAIR_BEST_MEAN[p] for p in sub_pairs]
    sub_pair_anchors = [max(SINGLE_BLOCK_ANCHORS[p[0]], SINGLE_BLOCK_ANCHORS[p[1]]) for p in sub_pairs]
    sub_pair_deltas = [bm - wa for bm, wa in zip(sub_pair_best_means, sub_pair_anchors)]
    contains_b10_b11 = (10, 11) in sub_pairs
    max_sub_pair_delta = max(sub_pair_deltas)
    mean_sub_pair_delta = sum(sub_pair_deltas) / len(sub_pair_deltas)

    print(f"\n  ==> triple {triple} best_mean = {best_mean:.4f}  "
          f"worst_single = {worst_single:.4f}  "
          f"delta_v_worst = {delta_vs_worst:+.4f}  "
          f"destructive@5={destructive}  "
          f"non-destructive@2={nondestructive_strict}  "
          f"contains_b10_b11={contains_b10_b11}  "
          f"max_pair_delta={max_sub_pair_delta:+.4f}", flush=True)

    triple_results.append({
        "triple": blks,
        "ranks": rs,
        "trainable_params": expected_trainable,
        "seeds": seeds_for_this_triple,
        "n_seeds": len(seeds_for_this_triple),
        "init_mean": float(sum(init_ppls) / len(init_ppls)),
        "best_mean": best_mean,
        "best_std": best_std,
        "best_min": float(min(best_ppls)),
        "best_max": float(max(best_ppls)),
        "final_mean": float(sum(final_ppls) / len(final_ppls)),
        "anchors": anchors,
        "worst_single": worst_single,
        "best_single": best_single,
        "delta_vs_worst_single": delta_vs_worst,
        "delta_vs_best_single": delta_vs_best_single,
        "destructive_at_5ppl": bool(destructive),
        "nondestructive_at_2ppl": bool(nondestructive_strict),
        "contains_b10_b11": bool(contains_b10_b11),
        "sub_pairs": [list(p) for p in sub_pairs],
        "sub_pair_cycle310_best_mean": sub_pair_best_means,
        "sub_pair_cycle310_delta": sub_pair_deltas,
        "max_sub_pair_delta": max_sub_pair_delta,
        "mean_sub_pair_delta": mean_sub_pair_delta,
    })

wall_min = (time.time() - wall_start) / 60
print(f"\nTotal triple sweep wall: {wall_min:.1f} min", flush=True)


# - Verdict ---------------------------------------------------------------------
n_destructive = sum(1 for tr in triple_results if tr["destructive_at_5ppl"])

b10_b11_triples = [tr for tr in triple_results if tr["contains_b10_b11"]]
other_triples = [tr for tr in triple_results if not tr["contains_b10_b11"]]
assert len(b10_b11_triples) == 3, len(b10_b11_triples)  # C(3,1)=3: {1,10,11},{2,10,11},{3,10,11}
# actually (b10,b11) in pair list of the 4 triples {1,10,11},{2,10,11},{3,10,11},{10,11,? other} - only 3 triples contain both 10 and 11
# Correction: choose 1 from {1,2,3}, plus {10,11} = 3 triples. The 4th of the "4 (b10,b11)-containing triples" in the brief is incorrect; C(3,1) = 3.
# Let me not assert 4; just follow brief reading -- brief says "All 4 triples containing (b10,b11)" which is wrong arithmetic, there are only 3.
# Remove strict assert. See agenda: "All 4 triples containing (b10, b11) destruct" -- this is a brief arithmetic error; actual count is C(3,1)=3.

all_b10b11_destruct = all(tr["destructive_at_5ppl"] for tr in b10_b11_triples)
all_other_nondestruct_strict = all(tr["nondestructive_at_2ppl"] for tr in other_triples)

if all_b10b11_destruct and all_other_nondestruct_strict:
    verdict = "CONFIRMED-b10b11-carries-destruction"
elif n_destructive >= 8:
    verdict = "CONFIRMED-triple-level-pervasive"
elif n_destructive <= 2:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

worst_triple_entry = max(triple_results, key=lambda tr: tr["delta_vs_worst_single"])
best_triple_entry = min(triple_results, key=lambda tr: tr["delta_vs_worst_single"])

print(f"\n=== Verdict ===", flush=True)
print(f"  n_destructive (delta >= +{DESTRUCTIVE_THRESHOLD} PPL): "
      f"{n_destructive} / {len(TRIPLES)}", flush=True)
print(f"  b10_b11-containing triples: {len(b10_b11_triples)} "
      f"(all destruct: {all_b10b11_destruct})", flush=True)
print(f"  non-b10_b11 triples: {len(other_triples)} "
      f"(all non-destructive@2: {all_other_nondestruct_strict})", flush=True)
print(f"  VERDICT: {verdict}", flush=True)
print(f"  worst triple: {worst_triple_entry['triple']} "
      f"(best_mean {worst_triple_entry['best_mean']:.4f}, "
      f"delta {worst_triple_entry['delta_vs_worst_single']:+.4f})", flush=True)
print(f"  best triple: {best_triple_entry['triple']} "
      f"(best_mean {best_triple_entry['best_mean']:.4f}, "
      f"delta {best_triple_entry['delta_vs_worst_single']:+.4f})", flush=True)


# - Save JSON -------------------------------------------------------------------
result = {
    "cycle": 317,
    "item": 327,
    "verdict": verdict,
    "float_ppl": float(float_ppl),
    "float_baseline_ref": FLOAT_BASELINE_REF,
    "block_order": BLOCK_ORDER,
    "blocks_ranks": {str(k): v for k, v in BLOCKS_RANKS.items()},
    "triples": [list(t) for t in TRIPLES],
    "triple_results": triple_results,
    "single_block_anchors": {str(k): v for k, v in SINGLE_BLOCK_ANCHORS.items()},
    "cycle_310_pair_best_mean": {f"{p[0]}_{p[1]}": v for p, v in CYCLE_310_PAIR_BEST_MEAN.items()},
    "cycle_305_5stack_best_mean": CYCLE_305_5STACK_BEST_MEAN,
    "n_destructive": n_destructive,
    "n_b10_b11_triples": len(b10_b11_triples),
    "all_b10_b11_destruct": bool(all_b10b11_destruct),
    "all_other_nondestructive_at_2ppl": bool(all_other_nondestruct_strict),
    "destructive_threshold_ppl": DESTRUCTIVE_THRESHOLD,
    "nondestructive_threshold_ppl": NONDESTRUCTIVE_THRESHOLD,
    "verdict_logic": {
        "n_destructive": n_destructive,
        "n_b10_b11_triples": len(b10_b11_triples),
        "all_b10_b11_destruct": bool(all_b10b11_destruct),
        "all_other_nondestructive_at_2ppl": bool(all_other_nondestruct_strict),
        "verdict": verdict,
        "worst_triple": worst_triple_entry["triple"],
        "worst_triple_best_mean": worst_triple_entry["best_mean"],
        "worst_triple_delta": worst_triple_entry["delta_vs_worst_single"],
        "best_triple": best_triple_entry["triple"],
        "best_triple_best_mean": best_triple_entry["best_mean"],
        "best_triple_delta": best_triple_entry["delta_vs_worst_single"],
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
        "cycle_305_5stack_best_mean": CYCLE_305_5STACK_BEST_MEAN,
        "cycle_310_pair_worst": 60.7161,  # (b10, b11)
        "cycle_310_pair_best": 53.5812,   # (b1, b11)
    },
    "wall_min": wall_min,
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {OUTPUT_PATH}", flush=True)


# - Summary markdown ------------------------------------------------------------
def fmt(x, n=4):
    return f"{x:.{n}f}"


lines = []
lines.append("# Cycle 317 — Item 327: 3-block LASER composition sweep")
lines.append("")
lines.append(f"**Verdict: {verdict}**")
lines.append("")
lines.append(f"- Float PPL: {fmt(float_ppl)} (ref {FLOAT_BASELINE_REF})")
lines.append(f"- Block order: {BLOCK_ORDER}")
lines.append(f"- Ranks: {BLOCKS_RANKS}")
lines.append(f"- N_STEPS={N_STEPS}, LR={LR}, SEEDS={SEEDS}, EVAL_EVERY={EVAL_EVERY}")
lines.append(f"- Triple sweep wall: {wall_min:.1f} min")
lines.append("")
lines.append("## Per-triple results (sorted by delta_vs_worst_single)")
lines.append("")
lines.append("| triple | ranks | trainable | init | best | delta_v_worst | destructive@5 | non-dest@2 | b10_b11? |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for tr in sorted(triple_results, key=lambda x: x["delta_vs_worst_single"]):
    tri = tr["triple"]
    rs = tr["ranks"]
    lines.append(
        f"| ({tri[0]},{tri[1]},{tri[2]}) | ({rs[0]},{rs[1]},{rs[2]}) | "
        f"{tr['trainable_params']:,} | "
        f"{fmt(tr['init_mean'])} | {fmt(tr['best_mean'])} | "
        f"{tr['delta_vs_worst_single']:+.4f} | "
        f"{'yes' if tr['destructive_at_5ppl'] else 'no'} | "
        f"{'yes' if tr['nondestructive_at_2ppl'] else 'no'} | "
        f"{'yes' if tr['contains_b10_b11'] else 'no'} |"
    )
lines.append("")
lines.append("## Pair-inheritance (cycle 310 sub-pair deltas per triple)")
lines.append("")
lines.append("| triple | sub_pairs | cycle 310 deltas | max_pair_delta | mean_pair_delta |")
lines.append("|---|---|---|---|---|")
for tr in sorted(triple_results, key=lambda x: x["delta_vs_worst_single"]):
    tri = tr["triple"]
    sp = tr["sub_pairs"]
    deltas = tr["sub_pair_cycle310_delta"]
    lines.append(
        f"| ({tri[0]},{tri[1]},{tri[2]}) | "
        f"{sp} | "
        f"[{', '.join(f'{d:+.2f}' for d in deltas)}] | "
        f"{tr['max_sub_pair_delta']:+.4f} | "
        f"{tr['mean_sub_pair_delta']:+.4f} |"
    )
lines.append("")
lines.append("## Verdict logic (item 327 verbatim)")
lines.append("")
lines.append(f"- n_destructive := |{{triple : delta_vs_worst_single ≥ +{DESTRUCTIVE_THRESHOLD}}}| "
             f"= **{n_destructive}** / {len(TRIPLES)}")
lines.append(f"- (b10,b11)-containing triples ({len(b10_b11_triples)}): "
             f"all destruct = **{all_b10b11_destruct}**")
lines.append(f"- non-(b10,b11) triples ({len(other_triples)}): "
             f"all non-destructive@2 = **{all_other_nondestruct_strict}**")
lines.append(f"- CONFIRMED-b10b11-carries-destruction iff both → {verdict == 'CONFIRMED-b10b11-carries-destruction'}")
lines.append(f"- CONFIRMED-triple-level-pervasive iff n ≥ 8 → {n_destructive >= 8}")
lines.append(f"- REFUTED iff n ≤ 2 → {n_destructive <= 2}")
lines.append(f"- VERDICT: **{verdict}**")
lines.append("")
lines.append(f"- worst triple: **{worst_triple_entry['triple']}** "
             f"best_mean={fmt(worst_triple_entry['best_mean'])} "
             f"delta={worst_triple_entry['delta_vs_worst_single']:+.4f}")
lines.append(f"- best triple: **{best_triple_entry['triple']}** "
             f"best_mean={fmt(best_triple_entry['best_mean'])} "
             f"delta={best_triple_entry['delta_vs_worst_single']:+.4f}")
lines.append("")
lines.append("## Comparisons to prior cycles")
lines.append("")
lines.append("- cycle 310 pair matrix: (b10,b11) 60.7161 (+6.25 unique destructive pair), (b1,b11) 53.5812 best")
lines.append(f"- cycle 305 5-stack: {CYCLE_305_5STACK_BEST_MEAN} (+18.54 vs worst-single)")
lines.append("")

with open(SUMMARY_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {SUMMARY_PATH}", flush=True)

print("\nDone.", flush=True)
