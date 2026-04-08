# Cycle 311 Audit — Item 321

**Audit status: PASS**

## Scope

Verify cycle 311 execution against item 321's verbatim hypothesis and
outcome thresholds. Verify the rank-1 retrain measurement reproduces the
cycle 280/285/288/289/292/305/310 float baseline, the LowRankCProj template
from cycle 292 is correctly extended to blocks {2, 3, 10, 11}, the per-block
joint Adam loop respects the freeze/train partition, the per-(block, seed)
best-state restore is exact, the b2 r=1 noretrain cross-check matches cycle
285 to ≤ 0.01 PPL, and the verdict matches the verbatim threshold logic.

## Verdict verification

Item 321 outcomes (verbatim from agenda):

- **CONFIRMED** iff `n_pass >= 3` (≥3/4 blocks at PPL ≤ 60)
- **REFUTED** iff `n_pass <= 1` (≤1/4 blocks)
- **INCONCLUSIVE** otherwise (2)

Auditor independently recomputed `n_pass` from
`cycle_311_rank1_all_concentrators.json::per_block`:

| block | best_mean | passes ≤60 |
|---:|---:|:---:|
| 2  | 49.7539 | yes |
| 3  | 55.0485 | yes |
| 10 | 52.9107 | yes |
| 11 | 59.7074 | yes (narrow, +0.29 PPL margin) |

n_pass = 4.

| criterion | n | threshold | result |
|---|---:|---|---|
| n_pass ≥ 3 (CONFIRMED) | 4 | ≥ 3 | **pass** (by +1) |
| n_pass ≤ 1 (REFUTED)   | 4 | ≤ 1 | fail |

CONFIRMED is the literal verbatim threshold result. Margin: n_pass is
+1 above the CONFIRMED floor and +3 above the REFUTED ceiling. The
verdict is unambiguous at the prescribed bar; sensitivity to the bar is
discussed in the substantive findings (b11 fails at PASS_THRESHOLD = 59).

## Reproduction fidelity — float baseline

Cycle 311 float PPL on the 100-seq eval slice = 53.8949. Cycle 280/284/
291/305/310 reference 53.895 (cycles 268-310 share this slice and same
baseline). Difference < 0.0001, within the 0.5-PPL assertion. Same
disjoint 200/100 split with seed 42 token selection, SEQ_LEN=128,
deterministic dataset iteration. Pass.

## Reproduction fidelity — b2 r=1 noretrain cross-check

cycle 311 b2 r=1 noretrain PPL = 72.3808
cycle 285 b2 r=1 noretrain ref = 72.3808
|delta| = 4.29e-06, asserted < 0.01

The b2 r=1 noretrain SVD truncation of c_proj_2 reproduces the cycle 285
no-retrain ref to 6 decimal places. This is the only block with a prior
r=1 noretrain anchor (cycle 282 used r=5/17/50 for b10; cycle 288 used
r=5/10/24 for b3; cycle 270 used r=5/22/50/100 for b11), so b2 is the
sole cross-check available. The match is bit-tight, confirming the SVD
basis and shape are identical to cycle 285. ✓

## Geometric object check

The script computes, for each block l in BLOCKS = [2, 3, 10, 11]:

- `W_orig = base_model.transformer.h[l].mlp.c_proj.weight.data` shape (3072, 768) HF Conv1D ✓ (asserted)
- `W_cpu = W_orig.float().cpu()` for SVD precision
- `U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)` with shapes (3072, 768), (768,), (768, 768)
- `A_l = (U[:, :1] * S[:1])` (3072, 1) — frozen `register_buffer`
- `B_l = Vh[:1, :]` (1, 768) — `nn.Parameter`
- `bias_l = c_proj.bias` (768,) — `nn.Parameter`
- forward: `x.view(-1, 768) @ (A @ B) + bias` then reshape — matches HF Conv1D `addmm(bias, x, weight)` semantics

The LowRankCProj class is identical to cycle 270/280/284/291/292/305/310
(line-for-line same `__init__` and `forward`); the only difference between
cycles 292 and 311 is the outer `for blk in BLOCKS` loop and the SVD
cache covering 4 blocks instead of 1. Per-block fresh GPT-2 reload per
seed; no model-state leak across blocks or seeds. ✓

## Trainable parameter accounting

Per (block, seed), the script asserts:
```
expected_trainable = 1 * 768 + 768  # = 1,536
trainable = [lowrank.B, lowrank.bias]
n_trainable = sum(p.numel() for p in trainable)
assert n_trainable == expected_trainable
```

Auditor independently computed expected count = 1·768 + 768 = 1,536 for
all 12 (block, seed) cells. All 12 satisfy the assertion at the top of
the per-seed Adam loop. Cross-check: cycle 311 trainable 1,536 matches
cycle 292 trainable 1,536 exactly (same template). ✓

## Joint Adam optimisation check

The optimisation pattern (per (block, seed)):
```
for p in model_s.parameters():
    p.requires_grad_(False)
lowrank.B.requires_grad_(True)
lowrank.bias.requires_grad_(True)
trainable = [lowrank.B, lowrank.bias]
optimizer = torch.optim.Adam(trainable, lr=LR)
```

Auditor checks:
1. The freeze pass `p.requires_grad_(False)` happens AFTER the
   LowRankCProj module is installed at `model_s.transformer.h[blk].mlp.c_proj`,
   so the new B / bias parameters are also frozen first; then the
   per-block enables turn just those 2 tensors back on.
2. Adam receives only the 2 tensors in `trainable`. No other parameter
   receives gradient.
3. Training loop is the standard 4000-step / EVAL_EVERY=100 / best-state
   restore pattern from cycle 270/280/284/291/292/305/310.
4. Fresh GPT-2 reload per (block, seed); no model-state leak across cells.

Per-(block, seed) restored vs best (from JSON):

| block | seed | best | restored | |restored − best| |
|---:|---:|---:|---:|---:|
| 2  |   42 | 49.6521 | 49.6521 | 0.0000 |
| 2  |  137 | 49.6408 | 49.6408 | 0.0000 |
| 2  | 2024 | 49.9688 | 49.9688 | 0.0000 |
| 3  |   42 | 55.0865 | 55.0865 | 0.0000 |
| 3  |  137 | 55.0434 | 55.0434 | 0.0000 |
| 3  | 2024 | 55.0155 | 55.0155 | 0.0000 |
| 10 |   42 | 52.9358 | 52.9358 | 0.0000 |
| 10 |  137 | 52.8993 | 52.8993 | 0.0000 |
| 10 | 2024 | 52.8971 | 52.8971 | 0.0000 |
| 11 |   42 | 59.7154 | 59.7154 | 0.0000 |
| 11 |  137 | 59.6968 | 59.6968 | 0.0000 |
| 11 | 2024 | 59.7099 | 59.7099 | 0.0000 |

All 12 cells within the 1e-3 assertion (in fact bit-for-bit identical),
confirming the best-state save/restore over a 2-tensor dict works
correctly across all blocks and seeds. ✓

## Per-block aggregate cross-check

Auditor independently computed best_mean / std from per-seed best PPLs:

| block | seeds best_ppl | mean | std (population) | matches script |
|---:|---|---:|---:|:---:|
| 2  | [49.6521, 49.6408, 49.9688] | 49.7539 | 0.1520 | ✓ |
| 3  | [55.0865, 55.0434, 55.0155] | 55.0485 | 0.0292 | ✓ |
| 10 | [52.9358, 52.8993, 52.8971] | 52.9107 | 0.0177 | ✓ |
| 11 | [59.7154, 59.6968, 59.7099] | 59.7074 | 0.0078 | ✓ |

All 4 aggregates match the script's reported best_mean and best_std to
4 decimal places. ✓

## Best-step distribution check (substantive — for budget assessment)

Per-(block, seed) best_step (from JSON):

| block | seeds best_step | last quartile (≥3000)? | plateau evidence |
|---:|---|:---:|---|
| 2  | [3700, 3900, 3300] | 3/3 | oscillating; final@4000 > best |
| 3  | [3900, 3900, 4000] | 3/3 | **monotone descending — UPPER BOUND** |
| 10 | [2500, 3100, 2800] | 1/3 | plateau by step ~2500-3100 |
| 11 | [3600, 3600, 3900] | 3/3 | **monotone descending — UPPER BOUND** |

b3 and b11 are still descending at N_STEPS = 4000 — best_means are
upper bounds; b10 plateaued; b2 oscillates. The CONFIRMED verdict is
robust to budget extension because b10 (the only plateau-converged
block) and b2 (oscillating) clearly pass the bar by ≥7 PPL, and the two
upper-bound blocks (b3, b11) only get LOWER (= further below the bar)
with more training. The narrow b11 pass at +0.29 margin would only
widen with more steps. ✓

## Plan adherence

Plan (`autoresearch/plans/cycle_311.md`) vs execution:
- Item 321 verbatim Hypothesis section ✓
- Step 1: new `experiments/rank1_all_concentrators_cycle311.py` adapted
  from cycle 292 with outer loop over BLOCKS = [2, 3, 10, 11] ✓
- Step 2: load GPT-2 once for float baseline; per-(block, seed) fresh
  reload (cycle 292 pattern) ✓
- Step 3: per block SVD on CPU float32; per-block r=1 no-retrain check;
  per-seed install LowRankCProj, freeze rest, Adam LR=1e-4 / N_STEPS=4000
  / EVAL_EVERY=100, best-state restore (assert |restored - best| < 1e-3) ✓
- Step 4: per-block aggregate (best_mean, best_std, best_min, best_max);
  verdict per item 321 (count blocks with best_mean ≤ 60) ✓
- Step 5: b2 r=1 no-retrain cross-check vs cycle 285 ref 72.3808
  within 0.01 ✓ (delta 4.29e-06)
- Files: all listed artefacts present ✓
- 8 H2 plan headings (Item 321 verbatim Hypothesis, Dependencies, Approach,
  Files, Config, Verification, Verdict logic, Baselines to record) ✓
- Plan marked `# APPROVE` ✓
- Plan length 30 lines, within 30-line plan budget ✓

All plan steps executed cleanly on the first run.

## Convergence check (CLAUDE.md verify-task-performance rule)

Float PPL on 100-seq eval slice = 53.8949, matches cycle 280/284/291/305/310
baseline 53.895 to 4 decimals. Off-the-shelf gpt2, no task training. Task
verification reduces to float PPL match. ✓

The model is task-converged at the float baseline (off-the-shelf gpt2 on
wikitext-2 val). Quantization analysis (here: per-block rank-1 LASER
substitution) is meaningful because the task baseline is correct.

## Wall time

Total ~28 min on mps (above the ~16-min plan estimate; the plan was
optimistic — actual was 2.0-2.5 min/seed × 12 seeds vs the 4-min/block
× 4 = 16-min plan budget; per-seed wall is consistent with cycle 292
~2.5 min/seed at 4000 steps):
- Model load + tokenisation + float baseline ~5 s
- 4 SVDs of (3072, 768) on CPU float32 ~0.5 s total
- 4 noretrain r=1 evals ~5 s
- 4 blocks × 3 seeds × (model reload + 4000 Adam steps + 41 eval rounds)
  ≈ 2.0-2.5 min per fit × 12 = ~26-30 min
- JSON + summary write ~0.2 s

Each (block, seed) reports `[2.0-2.5 min]` in the run log, consistent
with cycle 292 per-seed wall.

## Substantive reading — auditor endorses

1. **CONFIRMED is the verbatim-correct verdict at 4/4 blocks passing.**
   The block-level pass count is unambiguously above the CONFIRMED floor
   (≥3) and far above the REFUTED ceiling (≤1). Combined with cycle 292
   (b1 r=1 = 50.08), the result extends to ALL 5 cycle-282-family
   concentrators having a 1,536-param rank-1 LASER floor at PPL ≤ 60.

2. **b11 is the unique narrow-pass concentrator** at +0.29 PPL margin.
   The verdict is robust at PASS_THRESHOLD = 60 but would flip to
   INCONCLUSIVE at PASS_THRESHOLD = 59 (b11 fails). The narrowness is
   substantively interesting: b11's r=1 floor at 59.71 is +5.24 PPL above
   its r=22 anchor (54.47), reflecting cycle 270's reading that b11 is
   the least compressible cycle-282-family block. b11 is genuinely
   "rank-1 minimal" — it just barely fits within the bar.

3. **b2 r=1 BEATS its r=2 multi-rank anchor** by 0.42 PPL (within seed
   variation). This is substantively interesting: at 1,536 vs 2,304
   trainable params (33% fewer), the rank-1 fit at b2 is competitive
   with the cycle 285 rank-2 anchor. Two readings: (i) cycle 285's b2
   r=2 anchor was under-fit at 4000 steps (likely — most cycle 285 r=2
   best_steps were also in the last quartile per the original report);
   (ii) b2 has minimal incremental gain from r=2 over r=1 — the second
   singular component contributes little task-relevant information once
   the first is trained.

4. **b3 and b11 best_steps are still descending at N_STEPS = 4000**
   (b3 [3900, 3900, 4000]; b11 [3600, 3600, 3900]). b3 and b11 reported
   best_means are UPPER BOUNDS; longer N_STEPS would reduce them. b10
   plateaued by step ~2500-3100; b2 oscillates. The CONFIRMED verdict is
   robust to budget extension — only the bar-tight b11 cell would benefit,
   and it would only get further below the bar.

5. **Energy fraction is a poor predictor of trainable r=1 quality.**
   b10 r=1 energy 2.31% gives best_mean 52.91; b11 r=1 energy 4.01% gives
   59.71; b2 r=1 energy 7.16% gives 49.75; b3 r=1 energy 8.37% gives 55.05.
   Higher top-1 energy does not correspond to lower trainable r=1 PPL
   across blocks. The trained-Adam r=1 fit jointly optimises B (1, 768)
   and bias (768) given the frozen U[:, :1] · S[:1] axis; the per-block
   floor depends on how well that single frozen axis aligns with the
   downstream LM head's preferred input direction, NOT on how much
   spectrum mass it captures.

6. **The universal r=1 5-stack hypothesis is now testable cheaply.**
   5 blocks × 1,536 params = 7,680 trainable params total, vs cycle 305
   5-stack actual 46,080 (6.0× larger). Cycle 305 5-stack at the
   cycle-282-family anchors gave +18 PPL above worst single (REFUTED).
   The all-r=1 5-stack is the constructive next test: if the per-block
   optimal floor at 1,536 params is enough to compose without
   destruction, the 5-stack would be a tiny LASER recipe; if not, the
   destructive interaction at (b10, b11) is preserved at lower per-block
   rank as well, giving more evidence for the cycle 310 reading that the
   destruction is intrinsic to the b10-b11 LATE-LATE coupling.

7. **Cycle 304 k_95 census reading needs revision.** Cycle 304 measured
   pure-curvature k_95 at b1 = 6, b2 = 1, b3 = ?, b10 = 132, b11 = 1.
   The k_95 = 1 prediction held only for b2 and b11 in the cycle 304
   sense; cycle 311 shows that the trained-Adam r=1 fit reaches PPL ≤ 60
   at ALL 4 of {b2, b3, b10, b11} regardless of k_95. The right reading:
   k_95 measures pure-SVD energy concentration in the unweighted
   spectrum; the trained-Adam r=1 floor measures task-aware compression
   capacity given a frozen left axis and trainable right axis + bias.
   These are different geometric quantities; cycle 311 shows the latter
   is much more lenient than cycle 304 suggested.

## Corrections

None on the verdict. CONFIRMED is the literal verbatim threshold result.
n_pass = 4 ≥ 3; the REFUTED bar at ≤1 is missed by 3; the CONFIRMED
bar at ≥3 is exceeded by 1.

Minor note: the cycle 311 b3 and b11 reported best_means are UPPER
BOUNDS (best_steps in the last quartile). The cycle 285/288/289/270
multi-rank anchors and cycle 292 b1 r=1 are the canonical 4000-step
references for those blocks; cycle 311 adds the b2/b3/b10/b11 r=1
column. Future cycles should regard cycle 311 b3 and b11 best_means as
"≤ this value at ≥4000 steps" rather than as exact floors.

## Artifact existence

- `experiments/rank1_all_concentrators_cycle311.py` — exists, ~430 lines (driver)
- `autoresearch/plans/cycle_311.md` — exists, 30 lines (within 30-line
  budget), 8 H2 headings (Item 321 verbatim Hypothesis, Dependencies,
  Approach, Files, Config, Verification, Verdict logic, Baselines to
  record), marked `# APPROVE`
- `autoresearch/briefs/cycle_311.md` — exists (pre-existing from brief
  stage)
- `autoresearch/results/cycle_311_rank1_all_concentrators.json` — exists,
  valid JSON, verdict=CONFIRMED, per_block populated for 4 blocks, n_pass
  populated, baselines populated, cross_checks populated
- `autoresearch/results/cycle_311_summary.md` — exists, no-retrain table,
  per-(block, seed) detail table, per-block aggregate table, comparisons
  table
- `autoresearch/results/cycle_311_manifest.txt` — exists (this cycle)
- `autoresearch/results/cycle_311_exec_result.txt` — exists (this cycle)
- `autoresearch/audits/cycle_311.md` — this file
- `autoresearch/results/cycle_311_audit.txt` — to be written
- `autoresearch/logs/cycle_311_run.log` — exists, contains all 12
  per-(block, seed) per-500-step progress lines and the final
  VERDICT: CONFIRMED
- `autoresearch/agenda.md` — item 321 moved from NEXT P0 to PAST Cycle 311
- `autoresearch/logs/experiments.tsv` — row 311 appended
- `autoresearch/STATUS.md` — title bumped to "311 Cycles (001-311)";
  new "Latest: Cycle 311" section added; "Latest: Cycle 310" demoted

## Final status

**PASS** — keep provisional outcome CONFIRMED. The rank-1 LASER census
across the 4 cycle-282-family concentrator blocks {b2, b3, b10, b11}
gives n_pass = 4 / 4 at the PPL ≤ 60 bar. Combined with cycle 292
(b1 r=1 = 50.08), the result establishes a 1,536-param rank-1 LASER
floor at PPL ≤ 60 for ALL 5 cycle-282-family concentrators. b11 is the
unique narrow-pass concentrator (margin 0.29 PPL); b2 r=1 BEATS its
multi-rank r=2 anchor; b3 and b11 best_means are UPPER BOUNDS due to
budget exhaustion at N_STEPS = 4000. Cycle 304's pure-curvature k_95
census understates the trainable rank-1 capacity. The verdict is robust
to a budget extension (only b11 has a tight pass and would only get
better with more training).

Auditor recommends the constructive follow-ups already listed in
exec_result/manifest:
(1) extend b3 and b11 to N_STEPS = 8000 / LR-decay to find true r=1 floor;
(2) test the universal r=1 5-stack {b1, b2, b3, b10, b11} at 7,680
    trainable params (6.0× smaller than cycle 305) to test whether per-
    block r=1 fits compose;
(3) pairwise composition at all-r=1 (cycle 310 analog) to test whether
    the (b10, b11) destructive pair from cycle 310 is mitigated when
    both are at r=1.
