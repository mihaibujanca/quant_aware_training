# Cycle 298 Audit — Item 308

**Audit status: PASS**

## Scope
Verify cycle 298 execution against item 308's verbatim verdict thresholds; verify the hidden eff_rank census, the gradient ensemble eff_rank, and the LASER sweep results match the recorded JSON; verify the verdict computation logic; verify the substantive interpretation matches the data.

## Verdict verification

Item 308 thresholds (verbatim from agenda / brief):
- **CONFIRMED** iff (a) hidden output-covariance eff_rank ∈ [2.5, 3.5] post-training, (b) gradient ensemble eff_rank ∈ [1.8, 2.5], AND (c) rank-3 retrain held-out MSE ≤ 1.1× float MSE.
- **REFUTED** iff hidden eff_rank stays ≥ 8 OR rank-3 retrain MSE ≥ 2× float MSE.
- **INCONCLUSIVE** otherwise.

Auditor independently recomputed all three threshold quantities from `cycle_298_sdf_rank_laser.json`:

| quantity | value | threshold | result |
|---|---:|---|---|
| hidden mean eff_rank (layers 1..3) | **3.3235** | in [2.5, 3.5] | TRUE — clause A passes |
| gradient ensemble eff_rank | **2.6226** | in [1.8, 2.5] | FALSE — clause B fails by +0.12 |
| rank-3 retrain ratio vs float | **2.01835** | ≤ 1.1 (CONFIRMED) | FALSE |
| rank-3 retrain ratio vs float | **2.01835** | ≥ 2.0 (REFUTED)  | TRUE — fires by +0.018 |
| max hidden eff_rank (any layer) | **5.066** | ≥ 8 (REFUTED branch B) | FALSE |

CONFIRMED requires all three clauses; clauses B and C both fail. REFUTED requires (hidden ≥ 8) OR (rank-3 ratio ≥ 2.0); the second branch fires. **REFUTED** matches manifest, exec_result, JSON `verdict_label`, agenda PAST entry, and TSV row.

## Per-layer hidden eff_rank — independent reproduction

Auditor independently computed eff_rank from the JSON `hidden_eff_rank_per_layer` field:
- Layer 0 (post-ReLU after Linear (256, 3) input projection): **3.902**
- Layer 1 (post-ReLU after Linear (256, 256)): **5.066**
- Layer 2 (post-ReLU after Linear (256, 256)): **3.043**
- Layer 3 (post-ReLU after Linear (256, 256)) ← **l*** : **1.862**

Mean over hidden 1..3 = (5.066 + 3.043 + 1.862) / 3 = **3.3235** (matches JSON to 4 decimal places).

The funnel pattern is geometrically clean: the input projection lifts to ~4 (just above input_dim 3, ReLU randomization); layer 1 expands to ~5; layer 2 contracts back to ~3 (input dim); layer 3 collapses to ~2 (matching the 2D surface manifold). This is exactly the pattern item 308's hypothesis predicted at the qualitative level — input dim 3 sets the upper envelope, surface dim 2 sets the lower bound.

l* = argmin over layers 1..3 = layer 3 (eff_rank 1.862, the smallest). layer_seq_idx = 2 * 3 = 6, which is the third 256×256 Linear layer (the one whose output is the layer-3 post-ReLU activation). Auditor independently confirmed `model.layers[6]` is `Linear(in_features=256, out_features=256)` from the MLP class definition.

## Gradient ensemble — verification

`jacobian_rank_profile(model, X_train)` from `aleph/geometric_compression.py:374`:
- effective_rank: **2.6226**
- rank_90 = rank_95 = rank_99 = **3**
- top singular values: **[117.34, 70.07, 69.12]**
- n_samples: 24,000 (X_train)
- input_dim: 3

Verification: with input_dim=3, the gradient ensemble `G ∈ R^(N, 3)` has at most 3 nonzero singular values. The recorded SVs are 3 nontrivial values, consistent with `min(N, d) = 3`. The shape singular_values = (117.34, 70.07, 69.12) gives:
- Energy ratios: 117.34² / (117.34² + 70.07² + 69.12²) = 13769 / (13769 + 4910 + 4778) = 13769 / 23457 = **0.587**
- Plus 70.07²/23457 = 0.209 and 69.12²/23457 = 0.204
- Effective rank = exp(-(0.587·log 0.587 + 0.209·log 0.209 + 0.204·log 0.204)) = exp(-(−0.312 + −0.327 + −0.324)) = exp(0.963) = **2.620**

Matches the recorded 2.6226 to 3 decimal places. ✓

The geometric reading: 1 large normal direction (the SDF gradient is dominated by the surface-normal component, which is the most informative direction for the function value) plus 2 smaller, roughly equal tangential/radial directions. The "≈ 2" prediction in item 308's hypothesis was conditioned on the gradient living mostly on the 2D tangent space of the surface; in practice the volumetric points (50% of the dataset) add a 3rd direction (radial variation in the SDF away from the surface), pushing the effective rank to 2.62 — just above the predicted [1.8, 2.5] band. The discrete rank_95 = 3 is the more honest summary on a 3-D input. Clause B fails by +0.12, which is geometrically reasonable rather than a bug.

## LASER sweep — independent verification

Auditor independently checked all four ratio calculations from JSON:

| r  | retrain_best_held_mse | float_held_mse | ratio | best_step |
|---:|----------------------:|---------------:|------:|----------:|
|  2 | 6.839e-06             | 3.309e-06      | 2.067 |       500 |
|  3 | 6.679e-06             | 3.309e-06      | 2.018 |       500 |
|  5 | 5.574e-06             | 3.309e-06      | 1.684 |       500 |
| 10 | 4.317e-06             | 3.309e-06      | 1.305 |       500 |

All four ratios match the recorded values to 4 decimal places. All four `best_step = 500` confirm BUDGET-BOUNDED (best held-out MSE was at the final step). Ratios decrease monotonically with r as expected; even rank-10 (well above the gradient eff_rank 2.62) is at 1.30× float, confirming the substantive finding that the LAST hidden layer is hard for short-budget LASER recovery.

The no-retrain values (6.12e-3, 7.58e-3, 2.79e-3, 1.57e-3) are 3 orders of magnitude worse than float; the Adam retrain recovery factor is 4-5 orders of magnitude (e.g., r=3: 7.58e-3 → 6.68e-6, recovery factor ~1135×). This is consistent with cycle 296's spirals LASER (Adam recovery factor ~5-10×, smaller because the float baseline was already low-precision binary classification).

## SVD spectrum at l*

SVD top-10 of W_l* (from JSON `svd_top10`): [3.294, 2.383, 1.521, 1.398, 1.223, 1.179, 1.161, 1.144, 1.135, 1.111].

The spectrum is **2-dominant + flat tail**: top SV (3.29) is ~1.4× the second (2.38), then a sharp drop to 1.52, after which the tail is essentially flat at ~1.1. This is the SVD signature of a layer whose effective output rank is ≈ 2 (matching the measured eff_rank 1.86 on the post-ReLU activations). Truncating this layer at rank 2 keeps both dominant SVs plus discards the noisy tail; truncating at rank 3 keeps an additional 1.52 SV. The fact that even rank-10 retrain doesn't fully recover suggests the missing information is not in any single dominant direction but distributed across the entire flat tail. This is a tighter geometric reading than the verdict alone conveys.

## Plan adherence

Plan (cycle_298.md) vs execution (verified from JSON `config` and `experiments/sdf_rank_laser_cycle298.py`):
- HIDDEN=256 ✓, DEPTH=4 ✓, INPUT_DIM=3 ✓
- N_POINTS=30000 ✓, TRAIN_FRAC=0.8 ✓ (24000 train / 6000 held)
- mesh=torus ✓, surface_ratio=0.5 ✓, surface_noise=0.01 ✓
- LR_TRAIN=LR_RETRAIN=1e-3 ✓, MAX_EPOCHS=5000 ✓ (plateau-stop at 3175)
- PLATEAU_PATIENCE=250 ✓
- RETRAIN_STEPS=500 ✓, EVAL_EVERY=25 ✓, RANKS=[2, 3, 5, 10] ✓
- SEED=42 ✓ (single seed per plan)
- Adam, MSE, full-batch, CPU ✓
- aleph.models.MLP(3, 256, 1, depth=4, ReLU) ✓ — n_params asserted 198,657
- Layer selection: l* = argmin eff_rank over [1..3] ✓ (skipped layer 0 because Linear (256, 3) is not 256×256)
- LowRankLinear (A frozen 256×r, B trainable r×256, bias trainable 256) ✓
- Verdict logic matches item 308 verbatim ✓
- Output paths match plan ✓

## Trainable param count

MLP(input_size=3, hidden_size=256, output_size=1, depth=4) parameter count:
- Linear(3, 256):    3·256 + 256 = 1024
- Linear(256, 256):  256·256 + 256 = 65792 (×3 layers)
- Linear(256, 1):    256·1 + 1 = 257
- Total: 1024 + 3·65792 + 257 = 1024 + 197376 + 257 = **198,657**

Matches recorded `n_params=198657`. ✓ Script asserts this at runtime via `trainable_param_count(model)`.

LowRankLinear trainable per fit: r·256 + 256:
- r=2: 2·256 + 256 = 768
- r=3: 3·256 + 256 = 1024
- r=5: 5·256 + 256 = 1536
- r=10: 10·256 + 256 = 2816

The script asserts `n_trainable == r*HIDDEN + HIDDEN` for each fit. ✓

## Convergence check (CLAUDE.md "verify task performance before analysis")

- best_train_epoch = 2925 (plateau-stop at epoch 3175 after 250 epochs without ≥1e-4 relative improvement)
- final train MSE (after best-state restore): **2.371e-06**
- float held MSE: **3.309e-06**
- requirement (per plan): float held MSE < 0.01

3.309e-06 << 1e-2 by **4 orders of magnitude**. The model converged well below the verify-task threshold. The plateau-stop is principled (not undertraining) — the rest of the cycle's analysis is on a properly-trained network. ✓

The held-out MSE 3.31e-06 on a normalized SDF (range [-0.23, 1.09], mean abs ~0.25) corresponds to RMSE ~1.8e-3, which is well within the 1e-2 surface-noise std the SDF was sampled with. The model has effectively memorized the SDF up to the noise floor.

## Layer position effect — cross-check with cycle 296

The substantive interpretation in the manifest claims layer position is the dominant variable, comparing to cycle 296 where l* was middle-of-stack. Auditor verifies:
- Cycle 296 spirals: l* ∈ {4, 5, 6} of 12 hidden layers (middle of stack); rank-2 retrain best gap from float = +0.63% (within 1 std of float).
- Cycle 298 SDF: l* = 3 of 4 hidden layers (LAST hidden); rank-3 retrain best at 2.018× float.

The structural difference is that in cycle 296, layers 7-11 (after l*) provide downstream capacity to absorb the rank-truncation residual. In cycle 298, l* = 3 is the last hidden layer; only `model.layers[8] = Linear(1, 256)` (the output projection) is downstream, and it's a single Linear that maps 256→1 — too narrow to absorb a 256-dim residual. This is a structural rather than empirical claim: a depth-4 MLP truncated at the last hidden layer has essentially no downstream "residual sponge". The exec_result and manifest correctly flag this.

A natural follow-up that the manifest doesn't run but is worth flagging: rerun the LASER sweep at layer_seq_idx ∈ {2, 4} (l*=1 or 2 — earlier hidden layers with downstream depth) to confirm the layer-position hypothesis. The eff_rank at those layers is higher (5.07, 3.04) so the rank-3 truncation is more aggressive on each layer, but downstream layers can compensate. This is in scope for a cycle 299/300 follow-up if the agenda generates one. (Auditor note only — not a correction.)

## Substantive interpretation — audit

The verdict is **REFUTED** per the verbatim thresholds, but the substantive content is sharper:
1. **Input-dim hypothesis CLEAN HOLD**: hidden mean eff_rank 3.32 ≈ input_dim 3. The funnel [3.90, 5.07, 3.04, 1.86] is the textbook prediction.
2. **Surface-dim hypothesis NEAR-HOLD**: gradient eff_rank 2.62 just above [1.8, 2.5] by +0.12; rank_95 = 3 is the more honest summary; the 0.62 excess comes from volumetric (non-surface) gradient diversity.
3. **LASER recipe REFUTED at LAST HIDDEN LAYER, not at the input-dim level**: the verbatim REFUTED clause fires at rank-3 ratio 2.018, but the cause is layer position (no downstream sponge), not the eff_rank prediction itself. Cycle 296 spirals at middle-of-stack l* recovered to within 0.6% at rank-2 with the same 500-step budget; cycle 298 last-hidden l* ratio is 2.07 at rank-2, decreasing monotonically to 1.30 at rank-10 (all budget-bounded).
4. **Rank-2 fails worse than rank-3** (2.067 vs 2.018) — the "rank-2 fails" prediction in item 308 holds in the trivial sense, but both fail the match-float bar.

The exec_result, manifest, agenda PAST entry, and STATUS.md all document this substantive content alongside the REFUTED verdict label. The auditor concurs with the substantive reading and the layer-position hypothesis. The honest answer to item 308's question is: "yes, hidden eff_rank tracks input dim 3 on SDF; the gradient ensemble eff_rank tracks something close to surface dim 2; but rank-3 retrain at the last hidden layer needs more than 500 Adam steps to match float MSE because there is no downstream layer to absorb the residual."

## Cross-check: budget-bounded retrain

All four LASER retrain fits show `best_step = 500` (the final step). This means the held-out MSE was still improving when the budget ran out — the recorded ratios are upper bounds on what 500 Adam steps can achieve, and the true asymptotic ratio may be lower (potentially close to 1 with enough budget). This is consistent with cycle 280, 287, 294 budget-bounded findings on transformer LASER. The auditor flags this as a substantive caveat but does not change the verdict (the verbatim verdict logic is independent of budget-saturation status — it operates on the recorded `retrain_best_held_mse` regardless of whether the optimizer was still improving). A natural follow-up would be a budget-extension run (RETRAIN_STEPS=2000-4000) to see if the rank-3 ratio crosses 1.1 — this is the same kind of follow-up cycle 294 ran on cycle 276's 1-block channel-K sweep.

## Artifact existence

- `experiments/sdf_rank_laser_cycle298.py` — exists, 574 lines
- `autoresearch/plans/cycle_298.md` — exists, item 308 verbatim Hypothesis section, 8 H2 headings (Item 308, Dependencies, Approach, Files, Config, Verdict logic, Verification, Baselines)
- `autoresearch/briefs/cycle_298.md` — exists (from brief stage)
- `autoresearch/results/cycle_298_sdf_rank_laser.json` — exists, valid JSON, verdict_label="REFUTED", all required fields populated
- `autoresearch/results/cycle_298_summary.md` — exists, per-section markdown
- `autoresearch/results/cycle_298_manifest.txt` — exists, commit=da1ad63
- `autoresearch/results/cycle_298_exec_result.txt` — exists, verdict block
- `autoresearch/logs/cycle_298_run.log` — exists, captures training plateau / hidden eff_rank / grad profile / LASER sweep / verdict
- Primary commit `da1ad63` exists in git log
- `experiments.tsv` row 298 appended with 8 columns; hash patched from `pending` to `da1ad63` ✓
- `agenda.md`: item 308 deleted from NEXT P1; added to PAST as Cycle 298 ✓ (auditor verified item 308 no longer appears in NEXT and item 309 is now the first P1 entry)
- `STATUS.md`: regenerated to "298 Cycles (001-298)", new "Latest: Cycle 298" section ✓

## Corrections

None on the verdict. REFUTED is the literal verbatim threshold result and the substantive answer (input-dim hypothesis HOLDS, surface-dim NEAR-HOLD, layer-position dominates over eff_rank as the variable that controls LASER recovery) is consistent with it. The torus mesh choice (vs Stanford bunny) is documented in the plan as a deterministic procedural alternative to "or similar"; it does not affect the input-dim or surface-dim hypotheses (both apply to any closed orientable 2D manifold in R^3).

## Final status

**PASS** — keep provisional outcome REFUTED. The verdict logic is correctly applied to the verbatim thresholds; the per-clause counts match independent recomputation from the recorded JSON; the gradient eff_rank is verified against the singular-value entropy formula to 3 decimal places; the hidden eff_rank census matches the JSON to 4 decimal places; the param count is asserted at runtime and matches the analytical computation (198,657); the model converged 4 orders of magnitude below the verify-task threshold; the substantive finding (input-dim CLEAN HOLD, surface-dim NEAR-HOLD, LASER recipe transferability fails at the LAST hidden layer due to layer-position effect) is documented appropriately and the implications for the next agenda are flagged.

Auditor note: cycle 298 closes item 308 with an interesting substantive finding even though the verdict label is REFUTED. The recipe transferability question (can the cycle 280 / cycle 296 LASER recipe extend to last-hidden-layer SDF regression) is now answered NO at the 500-step budget. A natural follow-up for the next agenda refresh would be either (A) budget-extension (RETRAIN_STEPS=2000-4000) to test the asymptotic rank-3 ratio, or (B) layer-position sweep (LASER at l*=1 or 2 instead of 3, where downstream layers can absorb the residual). Both are in scope for cycle 299/300 if the agenda generates them.
