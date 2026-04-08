# Cycle 305 Audit — Item 315

**Audit status: PASS**

## Scope

Verify cycle 305 execution against item 315's verbatim hypothesis and
outcome thresholds. Verify the 5-block joint LASER measurement reproduces
the cycle 280/284/291 float baseline, the LowRankCProj template from cycle
284 is correctly extended from 2 to 5 blocks with no parameter leakage,
the joint Adam loop respects the freeze/train partition, the per-seed
best-state restore is exact, and the verdict matches the verbatim
threshold logic.

## Verdict verification

Item 315 outcomes (verbatim from brief / agenda):
- **CONFIRMED** iff `best_mean ≤ 51.0`
- **REFUTED** iff `best_mean ≥ 55.0`
- **INCONCLUSIVE** otherwise

Auditor independently recomputed `best_mean` from
`cycle_305_multi_block_laser.json::retrain_seeds`:

| seed | best_ppl |
|---:|---:|
| 42   | 72.8047 |
| 137  | 73.1217 |
| 2024 | 73.0915 |

mean = (72.8047 + 73.1217 + 73.0915) / 3 = 218.0179 / 3 = 73.00597 ≈ 73.0060
std (population) = sqrt( ((72.8047-73.006)^2 + (73.1217-73.006)^2 + (73.0915-73.006)^2) / 3 )
                 = sqrt( (0.0405 + 0.0134 + 0.0073) / 3 )
                 = sqrt( 0.0204 )
                 = 0.1429

| criterion | value | threshold | result |
|---|---|---|---|
| best_mean ≤ 51.0 (CONFIRMED) | 73.0060 | ≤ 51.0 | **fail** |
| best_mean ≥ 55.0 (REFUTED) | 73.0060 | ≥ 55.0 | **pass** (by +18.006) |

REFUTED is the literal verbatim threshold result. The 5-block joint LASER
recipe at the cycle-282-family individual rank anchors (b1 r=2 + b2 r=2 +
b3 r=24 + b10 r=5 + b11 r=22, 46,080 trainable params) lands at 73.01 PPL,
+19.11 above float and +18.54 above the worst single-block anchor (b11
r=22 = 54.47).

## Reproduction fidelity — float baseline

Cycle 305 float PPL on the 100-seq eval slice = 53.8949. Cycle 280 ref
53.895 (cycles 268, 269, 270, 280, 284, 285, 288, 289, 291 all share this
slice and same baseline). Difference = 0.0001, within the 0.5-PPL
assertion. Same disjoint 200/100 split with seed 42 token selection,
SEQ_LEN=128, deterministic dataset iteration. ✓

## Reproduction fidelity — init PPL determinism

Init PPL across all 3 seeds = 150.3568, identical to 4 decimals. The
init is deterministic because:
1. SVD on each W_l ∈ R^(3072, 768) is reproducible in CPU float32 (single
   precision is bitwise-identical for the same input on the same hardware).
2. No randomness in the LowRankCProj installation order — it iterates
   `for blk in TARGET_BLOCKS`.
3. Eval is deterministic (no dropout, model.eval(), torch.no_grad()).

The 5-block hybrid no-retrain init reproduces deterministically across the
3 seeds, confirming the SVD truncation and the LowRankCProj forward path
are correct. ✓

## Geometric object check

The script computes, for each l in TARGET_BLOCKS:
- `W_orig = base_model.transformer.h[l].mlp.c_proj.weight.data` shape (3072, 768) HF Conv1D ✓ (asserted)
- `W_cpu = W_orig.float().cpu()` for SVD precision
- `U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)` with `U.shape == (3072, 768)`, `S.shape == (768,)`, `Vh.shape == (768, 768)`
- `A_r = (U[:, :r_l] * S[:r_l])` shape (3072, r_l) — frozen `register_buffer("A", ...)`
- `B_r = Vh[:r_l, :]` shape (r_l, 768) — `nn.Parameter`
- `bias` shape (768,) initialised from original `c_proj.bias` — `nn.Parameter`
- forward: `x.view(-1, 768) @ (A @ B) + bias` then reshape to original — matches HF Conv1D `addmm(bias, x, weight)` semantics

This is the standard cycle 270/280/284/291 LowRankCProj template,
extended trivially from 1-2 blocks to 5 blocks. The forward path is
identical at every block; the only change vs cycle 284 is the iteration
over TARGET_BLOCKS = (1, 2, 3, 10, 11) instead of (1, 11) and the
RANKS tuple (2, 2, 24, 5, 22) instead of (2, 22). ✓

## Trainable parameter accounting

Per fit, the script asserts:
```
expected_trainable = sum(r * 768 + 768 for r in RANKS)
                   = (2*768 + 768) + (2*768 + 768) + (24*768 + 768)
                   + (5*768 + 768) + (22*768 + 768)
                   = 2304 + 2304 + 19200 + 4608 + 17664
                   = 46,080
assert expected_trainable == 46080
n_trainable = sum(p.numel() for p in trainable)
assert n_trainable == expected_trainable
```

Auditor independently computed: 5 LowRankCProj modules contribute B and
bias each, and every other model parameter is set `requires_grad_(False)`
before the per-block enables. The `trainable` list contains exactly 10
tensors (5 B + 5 bias), and `sum(.numel())` matches 46,080.

Note that the brief says "≈42K trainable" but the exact value is 46,080
(the brief was approximating). The plan and script use the exact value.

The total is exactly the sum of the 5 single-block trainable budgets:
2,304 + 2,304 + 19,200 + 4,608 + 17,664 = 46,080. So the cycle 305 fit
has exactly the same parameter capacity as if you trained each anchor
independently and summed the budgets. The destructive composition is NOT
a parameter shortage. ✓

## Joint Adam optimisation check

The optimisation pattern:
```
for p in model_s.parameters():
    p.requires_grad_(False)
trainable = []
for blk in TARGET_BLOCKS:
    lowrank_modules[blk].B.requires_grad_(True)
    lowrank_modules[blk].bias.requires_grad_(True)
    trainable.append(lowrank_modules[blk].B)
    trainable.append(lowrank_modules[blk].bias)
optimizer = torch.optim.Adam(trainable, lr=LR)
```

Auditor checks:
1. The freeze pass `p.requires_grad_(False)` happens AFTER the LowRankCProj
   modules are installed, so the new B / bias parameters are also frozen
   first; then the per-block enables turn just those 10 tensors back on.
2. The Adam optimiser receives only the 10 tensors in `trainable`. No
   other parameter receives gradient.
3. The training loop is the standard 4000-step / EVAL_EVERY=100 / best-
   state-restore pattern from cycle 270/280/284, with the only change
   being the per-block dict comprehension over TARGET_BLOCKS for
   best_state.

Per-seed restored vs best:
| seed | best | restored | |restored - best| |
|---:|---:|---:|---:|
| 42   | 72.8047 | 72.8047 | 0.0000 |
| 137  | 73.1217 | 73.1217 | 0.0000 |
| 2024 | 73.0915 | 73.0915 | 0.0000 |

All 3 within the 1e-3 assertion (in fact bit-for-bit identical), confirming
the best-state save/restore over a 5-block dict works correctly. ✓

## Plan adherence

Plan (`autoresearch/plans/cycle_305.md`) vs execution:
- Item 315 verbatim Hypothesis section ✓
- Step 1: new `experiments/multi_block_laser_cycle305.py` adapted from
  cycle 284 with TARGET_BLOCKS=(1, 2, 3, 10, 11), RANKS=(2, 2, 24, 5, 22) ✓
- Step 2: 3 seeds, fresh model reload per seed, 5 LowRankCProj installed
  simultaneously, every other parameter frozen ✓
- Step 3: joint Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state
  restore, |restored - best| < 1e-3 asserted ✓
- Step 4: verbatim verdict per item 315 thresholds (51.0 / 55.0) ✓
- Config: N_CALIB=200, N_EVAL=100, SEQ_LEN=128, MPS, GPT-2 small ✓
- Files: all listed artefacts present ✓
- 8 H2 plan headings (Item 315 verbatim Hypothesis, Dependencies, Approach,
  Files, Config, Verification, Verdict logic, Baselines to record) ✓
- Plan marked `# APPROVE` ✓
- Plan length 36 lines, within 45-line plan budget ✓

All plan steps executed cleanly on the first run.

## Convergence check (CLAUDE.md verify-task-performance rule)

Float PPL on 100-seq eval slice = 53.8949, matches cycle 280/284/291
baseline 53.895 to 4 decimals. Off-the-shelf gpt2, no task training. Task
verification reduces to float PPL match. ✓

The model is task-converged at the float baseline (off-the-shelf gpt2 on
wikitext-2 val). Quantization analysis (here: 5-block low-rank
substitution) is meaningful because the task baseline is correct.

## Trainable param count

`Trainable params: 46,080.` Verified by independent .numel() sum of the 10
tensors in `trainable` (5 B factors at r_l*768 each + 5 biases at 768
each). All other model parameters have `requires_grad == False` after
the freeze pass. ✓

## Wall time

Total ~8 min on mps (under the ~10-min plan estimate):
- Model load + tokenisation + float baseline ~5 s
- 5 SVDs of (3072, 768) on CPU float32 ~0.5 s total
- 3 seeds × (model reload + 4000 Adam steps + 41 eval rounds) ≈ 2.6 min
  per seed × 3 = ~7.8 min
- JSON + summary write ~0.2 s

Each seed reports `[2.6 min]` in the run log, consistent.

## Substantive reading — auditor endorses

1. **REFUTED is the verbatim-correct verdict at +18 PPL above the bar.**
   The 5-block joint LASER lands at 73.01 PPL, well above the 55.0 REFUTED
   threshold and ~22 PPL above the 51.0 CONFIRMED threshold. This is a
   strong, unambiguous REFUTED outcome — not a marginal one.

2. **The 5 single-block anchors are NOT additively composable.** The
   sum of single-block trainable budgets equals the cycle 305 budget
   (46,080 exactly), but the 5-stack lands +18.5 PPL above the worst
   single anchor (b11 r=22 = 54.47). The deficit is geometric, not a
   parameter shortage.

3. **Cycle 304's prediction is empirically validated.** Cycle 304 audit
   wrote "destructive composition (>55 PPL) is more likely than the item
   315 best-mean-≤-51 CONFIRMED threshold" because only b11 is a true
   rank-2 terminator (k_95 measurement). Cycle 305 confirms this at +18
   PPL above the REFUTED bar — the prediction was correct in sign and
   conservative in magnitude.

4. **Cycle 284 (b1+b11 CONFIRMED) does NOT extrapolate.** Cycle 284's
   non-destructive 2-pair composition (b1 r=2 + b11 r=22 → 52.88) gave
   the false impression that the cycle-282-family anchors compose. Adding
   3 more anchors (b2 r=2, b3 r=24, b10 r=5) on top of the cycle 284
   pair adds +20.13 PPL — far more than a linear extrapolation predicts.
   The cycle 291 (b1+b10 REFUTED) finding that "high-rank partners (r=22)
   absorb low-rank damage" also breaks: cycle 305 has both r=22 (b11)
   and r=24 (b3) high-rank partners present, yet the 5-stack is still
   destructive.

5. **Final-step vs best-step delta indicates Adam is budget-bounded.**
   Across the 3 seeds, final@4000 - best is +1.69 / +0.61 / 0.00 PPL
   (mean +0.77). best_step is in [3500, 4000] for all 3 seeds. The Adam
   trajectory is still descending at the end of the 4000-step budget;
   a longer run might lower best_mean by another ~1-2 PPL but would not
   close the ~18 PPL gap to the 55 threshold (let alone the ~22 PPL
   gap to 51). The REFUTED verdict is robust to budget extension.

6. **Practical implication for future multi-block LASER schedules.** Joint
   Adam from a simultaneous 5-replacement init does not reach the
   single-block floors. Three plausible alternatives for any future
   multi-block c_proj rank schedule:
   (a) raise per-block ranks beyond the single-block floors (more slack);
   (b) use sequential / coordinate descent over blocks rather than
       joint Adam from a 5-replacement init;
   (c) use a curvature-aware composition (e.g., propagate cycle 304's
       k_95-aware ranks instead of cycle 282 single-block training-derived
       ranks).
   The cycle 305 result alone does not pick between these.

## Corrections

None on the verdict. REFUTED is the literal verbatim threshold result.
best_mean = 73.0060 ≥ 55.0. The CONFIRMED bar at 51.0 is missed by 22.01;
the REFUTED bar at 55.0 is exceeded by 18.01.

Minor note: the brief stated trainable ≈42.5K and the agenda item 315
similarly used "42.5K trainable". The exact value is 46,080. The plan
and the script use the exact value and assert it. No correction needed
to brief / agenda — they were approximating from "(2+2+24+5+22)·768"
without including the 5·768 bias contribution.

## Artifact existence

- `experiments/multi_block_laser_cycle305.py` — exists, ~330 lines (driver)
- `autoresearch/plans/cycle_305.md` — exists, 36 lines (within 45-line
  budget), 8 H2 headings (Item 315 verbatim Hypothesis, Dependencies,
  Approach, Files, Config, Verification, Verdict logic, Baselines to
  record), marked `# APPROVE`
- `autoresearch/briefs/cycle_305.md` — exists (pre-existing from brief
  stage)
- `autoresearch/results/cycle_305_multi_block_laser.json` — exists, valid
  JSON, verdict=REFUTED, retrain_seeds populated for 3 seeds, full per-seed
  history, baselines and deltas populated
- `autoresearch/results/cycle_305_summary.md` — exists, per-seed table and
  comparisons table
- `autoresearch/results/cycle_305_manifest.txt` — exists (this cycle)
- `autoresearch/results/cycle_305_exec_result.txt` — exists (this cycle)
- `autoresearch/audits/cycle_305.md` — this file
- `autoresearch/results/cycle_305_audit.txt` — to be written
- `autoresearch/logs/cycle_305_run.log` — exists, contains all 3 seeds'
  per-500-step progress lines and the final VERDICT: REFUTED
- `autoresearch/agenda.md` — item 315 to be moved from NEXT P0 to PAST
  Cycle 305
- `autoresearch/logs/experiments.tsv` — row 305 to be appended
- `autoresearch/STATUS.md` — title to be bumped to "305 Cycles (001-305)";
  new "Latest: Cycle 305" section to be added; "Latest: Cycle 304" demoted

## Final status

**PASS** — keep provisional outcome REFUTED. The 5-block joint LASER stack
at the cycle-282-family individual rank anchors (b1 r=2 + b2 r=2 + b3 r=24
+ b10 r=5 + b11 r=22, 46,080 trainable params) lands at 73.01 PPL ± 0.14
across 3 seeds, +18.01 PPL above the 55.0 REFUTED bar and +22.01 PPL above
the 51.0 CONFIRMED bar. The 5 single-block "concentrator" rank floors do
NOT compose. Cycle 304's prediction is validated. The cycle 282-family
single-block LASER recipe is not additively composable into a multi-block
recipe at these rank assignments.

Auditor recommends: (1) Open a follow-up item with rank-up-by-50%
per-block schedule to test whether destructive composition is a rank-slack
issue rather than an architectural impossibility (e.g., r = (3, 3, 36, 8,
33) at ~69K trainable). (2) Open a sequential/coordinate-descent variant
(install one block, train to convergence, freeze, install next) to
distinguish "joint init is the problem" from "the joint optimum doesn't
exist". (3) Item 316 (block 8 deep dive) is unaffected by cycle 305 — it
remains a pure-geometry question about W_proj at b8.
