# Item
246 (P0): T2 — Causal intervention on top bump directions.
Take top 5 directions per block from cycle 235's T1 (skip blocks with <5 coherent). On a held-out 1000-token batch from wikitext-2 val, intervene at residual stream after block l with `x_l + α · v_{l,i}` for α ∈ {-3, -1, 0, +1, +3} (units = std of U_l[:, i]). Measure top vocab logit shift, compare to random direction baseline, and pick 10 concrete example records with biggest L2 logit shift.

# Context
- Cycle 235 (CONFIRMED with G3 caveat): T1 found ≥10 coherent directions per block but mostly `repeated` label (not strict semantic).
- T2 is the gating causal test — does intervening on these directions change predictions in matching ways?

# Dependencies
- 245 / cycle 235 (done) — provides V_l, S_l, mean_delta_l, and the U_l std (need to recompute or save).

# Key Files
- experiments/bump_t1_features_cycle235.py — for the Δ̃ pipeline + V/S/mean_delta loading.
- autoresearch/results/cycle_235_t1_features.pt — V (768,32), S (32,), mean_delta (768,) per block.
- autoresearch/results/cycle_235_t1_decoded.json — top boost/supp vocab per direction (used for T1-T2 Jaccard overlap).
- notebooks/bump_features_explore.py — loader contract for T2 outputs.

# Constraints
- Held-out: wikitext-2 VAL split, distinct from calibration (cycle 235 used train).
- Take top 5 directions per block by variance_explained from cycle 235.
- For each direction, the unit α MUST be the std of U_l[:, i] from the calibration corpus. cycle 235 didn't save U_l (it'd be ~100K × 32 × 12 = ~38M floats). RECOMPUTE U_l for the calibration corpus → take std per direction.
- Held-out batch: 1000 tokens (e.g., 8 sequences × 128 tokens = 1024 — round down).
- For each (block, direction, α): forward pass through the model, intercept x_l, modify, continue. Use forward hook on transformer.h[l+1] (the next block) that takes the input tensor and adds α · std_l_i · v_l_i (broadcast over batch and time).
- Measure logit shift = logits_intervened - logits_clean per token.
- Top 20 boost/supp vocab items by mean logit shift across the batch.
- Concrete example records: 10 tokens with largest L2 of logit shift.
- Random baseline: 10 v_random ∈ R^768 unit-norm Gaussian seeds at each block, same procedure, mean over seeds.
- Output filenames per agenda item 246:
  - experiments/bump_t2_intervention_cycle236.py
  - autoresearch/results/cycle_236_t2_logit_shifts.pt
  - autoresearch/results/cycle_236_t2_intervention.json
  - autoresearch/results/cycle_236_t2_examples.md
  - autoresearch/results/cycle_236_t2_summary.md
- Confirmed if: ≥3 directions per block in ≥6 of 12 blocks have BOTH (a) logit shift L2 at α=3 ≥ 3× random baseline mean, AND (b) Jaccard overlap of top-20 boosted vocab between intervention and T1's prediction ≥ 0.3.
- Refuted if: no direction in any block produces a logit shift distinguishable from random (effect size <1.2× baseline).
- Inconclusive if: directions cause shifts but they don't match T1's predictions.

# Handoff
Plan agent: design the script. Tricky parts:
1. Recompute U_l std from cycle 235's calibration data (re-run the forward pass + Δ̃ → SVD, but only get U). Or: save U_l std at T1 time. Since T1 is already done, recompute.
2. Hook insertion: after block l's forward, the residual stream entering block l+1 is `transformer.h[l+1]`'s input. Use forward_pre_hook on `model.transformer.h[l+1]` (or `ln_f` for block 11).
3. logit_shifts.pt may be huge — subsample to top-1000 token L2 shifts only.
4. Wall time: 12 blocks × 5 directions × 5 alphas × 1024 tokens × forward = 300 forward passes. Should be fine.

New script `experiments/bump_t2_intervention_cycle236.py`. Output 5 files per the spec.
