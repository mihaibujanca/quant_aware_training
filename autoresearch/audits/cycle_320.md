# Cycle 320 Audit — Item 329 (ReLU² spirals/moons eff_rank census + LASER)

## Audit status: PASS (provisional INCONCLUSIVE accepted; substantive findings flagged)

## Provisional outcome (from execute)
INCONCLUSIVE per item 329's strict tree because no model converged (0/6 cells reached the 95% train accuracy target). Substantive findings: (1) ReLU² random-init eff_rank profile is HIGHER than cycle 297 ReLU at layers 0/1/2 by +0.46/+0.65/+1.59, refuting item 329's "init within ±0.5 at every layer" sub-claim; (2) the cycle 295/296 LR=1e-3 / lr=1e-4 / grad-clip 1.0 recipe does not transfer to ReLU² at depth-12 32-wide on this 2D task — 5/6 cells diverge to NaN, 1/6 stays at chance.

## Verdict logic check (item 329 verbatim)
The agenda's verdict tree:
1. CONFIRMED-curvature-sparsifies iff init within ±0.5 of cycle 297 at every layer AND post mean strictly lower by ≥0.3 AND LASER r=2 retrain within 0.5%.
2. CONFIRMED-no-difference iff both within ±0.3 of cycle 297 at every layer.
3. REFUTED-curvature-expands iff post is higher than ReLU's by ≥0.5 anywhere.
4. INCONCLUSIVE otherwise.

Measured (from `cycle_320_relu2_spirals_eff_rank_laser.json` verdict_logic):
- `init_match_within_0_5`: false (max abs init diff is 1.59 at layer 2)
- `post_match_within_0_3`: false
- `post_strictly_lower_by_0_3_mean`: false
- `any_post_higher_by_0_5`: false (no post exceeds ReLU by 0.5; in fact post values are bounded above by the active set's max width which can't exceed ReLU's much because the models didn't train)
- `laser_r2_within_0_5pc`: false (spirals r=2 retrain mean 0.514 vs float held mean 0.522 — both at chance)

Branch 1 (CONFIRMED-curvature-sparsifies): init_match_within_0_5 false → FAIL.
Branch 2 (CONFIRMED-no-difference): init/post within ±0.3 — false → FAIL.
Branch 3 (REFUTED-curvature-expands): any_post_higher_0.5 — false → FAIL.
Branch 4: INCONCLUSIVE — TRUE.

The strict tree returns INCONCLUSIVE. The execute layer's verdict label is correct.

**Note about the strict tree**: branches 1-3 all assume the model converges. None of the verdict-tree branches handles "training failed" as a distinct outcome. INCONCLUSIVE is the only landing point and is honest about the data.

## Cross-checks (PASS where applicable)
- Init eff_rank measurements completed cleanly for all 6 cells (init is computed BEFORE training starts, so the NaN-explosion failures don't contaminate init data).
- The eff_rank computation switched from `eigvalsh(Y^T Y / N)` to `svdvals(Y / sqrt(N)) ** 2` after the first run hit `_LinAlgError` on a near-degenerate covariance — the SVD path is more numerically robust on rank-deficient matrices and produces the same eigenvalues for well-conditioned cases. PASS.
- Float64 promotion in `eff_rank_from_Y` ensures the entropy computation doesn't lose precision on small eigenvalues. PASS.
- Trainable param count = 11,745 (matches cycle 295/296's MLP(2, 32, 1, depth=12)).
- Convergence flags are accurate: 5/6 cells have train_acc=0.0 (NaN-induced), 1/6 has train_acc=0.512 (chance) — none reached the 0.95 target.

## Numerical sanity (re-checked)
- The `WARN: non-finite loss at epoch X` messages from train_until_acc fired at epochs {971, 15143, 768, 723, 717} for the 5 diverging cells. The single non-diverging cell (spirals seed=137) ran the full 40k budget without NaN but stayed at 0.512 accuracy.
- The LASER phase ran on the un-trained / failed models, so the rank_results are not interpretable as compression measurements. The script does not assert convergence before running LASER (this is a script-design choice — it produces LASER numbers regardless, but the audit MUST flag them as uninterpretable if the source model didn't train).
- The substantive init comparison: ReLU² spirals init mean (3 seeds) = [3.461, 3.351, 3.293, 1.080, 1.001, 1.000, ..., 1.000]. cycle 297 ReLU spirals init seed 42 = [3.0, 2.7, 1.7, 1.13, 1.05, 1.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]. The differences at layers 0/1/2 are +0.46/+0.65/+1.59. Layer 2's +1.59 is the load-bearing finding: ReLU² init at depth-2 has substantially more rank diversity than ReLU init at depth-2.

## Independent interpretation
**The headline negative result**: ReLU² depth-12 32-wide on spirals/moons does NOT train under any LR ∈ {1e-3, 1e-4} with grad-clip 1.0 on 2000-sample full-batch BCEWithLogits. This is the FIRST training-stability measurement for ReLU² in the project and answers (negatively) the user's "where are the ReLU² results" question with: the cycle 295/296 recipe needs an init or normalization fix before it can be used as a cross-activation control.

**The interpretable substantive finding**: ReLU² random-init eff_rank at the FIRST 3 hidden layers is higher than ReLU's by +0.46 to +1.59. This refutes item 329's hypothesis "init is activation-independent for this width/input ratio". The mechanism is plausible: under ReLU at random init, ~50% of channels are active and their post-activation values are uniform over their pre-activation range; under ReLU², the same active set's per-channel post-values are now `z²` which is non-uniform (concentrated near zero), but the COVARIANCE structure across channels has higher entropy because the squaring magnifies any correlation differences between channels (rank-2 input → rank-3 squared due to z² being non-linear). The +1.59 jump at layer 2 is the largest, consistent with cumulative compounding of the squaring effect through depth.

**Layer 3+ collapse to 1.0** under both activations: this is consistent with cycle 297's CONFIRMED-init-rank finding (Hanin-Rolnick: 2D input MLPs collapse to rank 1 by depth ~6 from random init regardless of activation choice).

**Why the post-training data is uninterpretable**: 5/6 cells diverged to NaN, then the LASER weights were SVDed AFTER restoring from a NaN-affected state. The "post_eff_rank" values [3.385, 2.336, 2.128, 1.299, 1.180, 1.073, 1.006, 1.0, ...] for spirals look like a sane decreasing profile, but they reflect either pre-divergence weights (since the models hit NaN in mid-training and the script didn't restore to last-finite weights) or zero weights from numerical underflow. Either way, they're not measurements of a "trained ReLU² model" — they're measurements of a "broken ReLU² model".

## Auditor recommendation
PASS the INCONCLUSIVE label. The strict tree forces INCONCLUSIVE because none of branches 1-3 can fire on a model that didn't converge. The execute layer is honest about the convergence failures (the JSON records `converged: false` for all 6 cells and includes the divergence epoch numbers).

The substantive value of cycle 320 is:
1. **Negative result**: cycle 295/296 LR recipe doesn't translate from ReLU to ReLU² at depth 12. Documented for future cross-activation work.
2. **Init eff_rank measurement**: ReLU² random init has +0.46 to +1.59 higher eff_rank than ReLU at the first 3 layers. This is the first ReLU² eff_rank measurement on the project. Refutes item 329's "init activation-independent" sub-hypothesis.

A constructive follow-up cycle should re-attempt with Kaiming init scaled for ReLU² (gain factor adjusted for the variance amplification of z²) OR with normalization layers OR at shallower depth. File as a new agenda item; do NOT retroactively modify cycle 320.

## Status: PASS (provisional outcome accepted unchanged; substantive negative result and init finding flagged)
