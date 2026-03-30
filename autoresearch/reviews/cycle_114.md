verdict: APPROVE
round: 1

## R1: Criteria faithfulness
Agenda criteria quoted verbatim. Verdict logic (plan step 7): delta = (uniform - best_nonunif)
/ uniform x 100. >=3% -> CONFIRMED, <1% -> REFUTED, else INCONCLUSIVE. Matches agenda. Pass.

## R2: Measurement soundness
PPL comparison at matched total params. The budget-matching correction (plan table) fixes the
agenda's 8x/4x mismatch: 2x384 + 2x640 = 2048 = 4x512. Param verification assertion in
step 6 guards against implementation error. Pass.

## R3: Known pitfalls
- **MP5 (insufficient capacity):** All conditions use the same total channel budget (2048).
  The most compressed single block is 8x (k=384). Cycle 094 ran k=308 (10x) without
  divergence. The capacity is sufficient for comparison. Not triggered.
- **MP1 (trivial threshold):** The 3% confirmed / 1% refuted thresholds are about relative PPL
  difference between conditions with identical budgets. No null-distribution concern. N/A.
- **MP3 (underdetermined fitting):** lstsq fits 768 output params from 4096 tokens.
  5:1 ratio. Not triggered.
- **MP6 (verdict misclassification):** PPL is a meaningful metric that varies across conditions.
  Not triggered.
- **MP2, MP4, MP7, MP8:** Not applicable (no detection, no domain transfer, no LayerNorm
  geometry, no representation dimension).

## R4: Cost and feasibility
3 allocations x 3 seeds x 4 blocks = 36 block fits + 500 Adam steps each. ~45 min MPS.
Reuses cycle 094 infrastructure (nonuniform_compression_gpt2.py pattern). Feasible.

## R5: Prior probability assessment
Cycle 094 tested damage-based nonuniform allocation on blocks [7,5,4,1] with 100 polish
steps and found REFUTED (+0.24%). This experiment uses linearity-based allocation on
blocks [1,4,7,11] with 500 polish steps. The mechanism is different (allocate by linearity
not by damage) but the structural setup is identical. Polish absorbed allocation differences
in cycle 094 and may do so again. Estimated prior: ~25% confirmed, ~55% refuted, ~20%
inconclusive.

## Advisory

**A1 (Polish absorption).** 500 Adam steps may erase the advantage of giving more capacity
to nonlinear blocks, just as 100 steps erased damage-based allocation in cycle 094.
Consider reporting pre-polish (lstsq-only) PPL alongside post-polish to separate the
allocation effect from the polish effect. Not blocking — adds one eval per condition.

**A2 (Block set difference).** Cycle 094 used [7,5,4,1]; this uses [1,4,7,11]. Block 11 is
new and was the highest-damage block in cycle 089. The linearity hypothesis is specifically
about blocks 4,7 (linear) vs 1,11 (nonlinear), so this block set is correct. No issue.
