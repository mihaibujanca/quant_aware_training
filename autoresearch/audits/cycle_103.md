# Audit: Cycle 103 — GELU-ReLU Disagreement Anatomy — GPT-2 FFN

**Item:** 108
**Claimed outcome:** REFUTED
**Audited outcome:** REFUTED
**Status:** PASS (with flag)

## Phase 1: Verdict Check

Refutation gate: bump energy >20% OR masking changes >10% predictions. Results: max energy_frac = 42.8% (block 3 at T=2.5), max pred_change = 53.8% (block 0). Both gates clearly triggered. All-blocks masking: +1318 PPL, 91.2% prediction change. **Refutation is unambiguous.**

## Phase 2: Quality Checks

### G1: Implementation correctness

**Pre-activation collection: CORRECT.** Hooks on `ffn.act` capture `input[0]`, which is the pre-activation z = W_in @ x + b_in. The hook placement on `model.transformer.h[b].mlp.act` is correct for GPT-2's architecture where `forward()` calls `self.act(hidden_states)`.

**Bump zone partition: CORRECT.** `z.abs() < T` partitions channels x tokens into bump vs non-bump. Thresholds [0.5, 1.0, 1.5, 2.0, 2.5] provide a sweep from tight to wide.

**Energy definition: CORRECT.** `energy_frac = sum(|GELU(z) - ReLU(z)|^2 in bump) / sum(|GELU(z)|^2)`. This measures disagreement energy relative to total GELU output energy. Appropriate for quantifying how much the GELU-ReLU difference matters relative to the signal.

**Masking experiment: CORRECT.** `MaskedGELU` replaces GELU output with ReLU output only where `|z| < T` (bump zone), keeping GELU output elsewhere. This correctly isolates the bump zone contribution. `copy.deepcopy(model)` ensures no cross-contamination between per-block evaluations.

**Prediction change: CORRECT.** `(preds != base_preds).float().mean()` measures fraction of top-1 predictions that differ between original and masked model. Straightforward and appropriate.

### G2: Reproducibility

3 seeds [42, 123, 7] are present in the code, but **all std values are 0.0 in the results**. The seeds only set `torch.manual_seed()` inside the masking loop, which has no effect because the masking operation is entirely deterministic (no random ops in `MaskedGELU`, no data re-sampling). The data is loaded once before the loop and reused. The 3 "seed" runs are identical replications producing identical results.

**FLAG: Effectively single-run.** The experiment lacks true independent replication. To get meaningful variance, the seeds should vary the data subset (call `load_owt_tokens` with different seeds inside the loop). Given the massive effect sizes (42.8% energy, 53.8% pred change), the conclusion would almost certainly survive, but the std=0 numbers are misleading.

### G3: Results interpretation

At T=2.5 (primary threshold):
- Max energy_frac = 0.428 (block 3) vs 0.20 refutation threshold: **2.1x above threshold**
- Max pred_change = 0.538 (block 0) vs 0.10 refutation threshold: **5.4x above threshold**
- All-blocks: +1318 PPL, 91.2% pred change

Even at tighter thresholds, the refutation holds:
- T=0.5: mean energy_frac = 0.071, max = 0.139 (block 1). Several blocks exceed 5%.
- T=1.0: mean energy_frac = 0.207, max = 0.276 (block 3). Refutation threshold crossed.

The per-block pattern is consistent with cycle 102: block 0 is most sensitive to masking (53.8% pred change, +81.7 PPL), middle blocks moderate (25-27%), late blocks 9-10 lowest (12-14%).

### G4: Plausibility

**94.6% of activations in |z|<2.5: PLAUSIBLE.** GPT-2 applies LayerNorm before FFN, normalizing inputs to ~unit variance. With W_in dimensions [3072, 768], the resulting pre-activations z have a distribution centered near the bias with spread determined by weight norms. For a roughly Gaussian distribution, |z|<2.5 captures ~98.8% of mass. The observed 94.6% is slightly below the Gaussian expectation, suggesting mild heavy tails, which is expected for real activations.

**31.5% mean energy: PLAUSIBLE.** Most pre-activations are near zero where |GELU(z) - ReLU(z)| is largest (the bump peaks at z ~ -0.66 with GELU(-0.66) ~ -0.16 vs ReLU(-0.66) = 0). With 94.6% of activations in this zone, the disagreement energy being ~1/3 of total output energy is consistent.

**53.8% prediction change at block 0: PLAUSIBLE.** Block 0 processes raw embeddings where the activation distribution is least structured. Consistent with cycle 102's finding that block 0 is catastrophically sensitive to GELU→ReLU changes (+2664% PPL for full swap).

### G5: What did we learn?

1. **The GELU bump is pervasive, not edge-case.** 94.6% of FFN pre-activations live in the bump zone (|z|<2.5), and the most bump-active tokens are common words ("winning", "ad", "There", "the"), not rare or special tokens. The bump zone IS the operating regime of GPT-2 FFNs.

2. **Disagreement energy is substantial and non-uniform.** Mean 31.5%, max 42.8% (block 3). Early and middle blocks carry more disagreement energy; block 11 has only 13.1% (its pre-activations are more spread out).

3. **Block 0 is uniquely sensitive to bump masking.** 53.8% prediction change and +81.7 PPL from masking block 0 alone. This reinforces cycle 102's finding and suggests block 0's GELU semantics are deeply baked into the model's representations.

4. **The negative side dominates.** At T=2.5, 83-86% of bump-zone activations are on the z<0 side (neg_frac), and 84-88% of total disagreement energy comes from the negative side (neg_energy_frac). The critical GELU-ReLU difference is the negative leak (GELU passes small negative values that ReLU kills), not the positive attenuation (GELU slightly suppresses small positive values).

5. **Piecewise-linear compression cannot treat the bump zone as negligible.** Any scheme that approximates GELU as ReLU in the transition region will lose substantial information. The bump zone must be explicitly modeled.

### G6: Measurement error?

**Not a measurement error.** The effect sizes are too large and too consistent across blocks to be artifacts. The masking logic is simple and correct (verified in G1). The baseline PPL of 25.1 is reasonable for GPT-2 on OWT with seq_len=128.

The only genuine concern is G2 (zero variance from fake seeds), which means we have one data sample rather than three. However, with 4096 tokens across 128-length sequences, the sample is large enough that block-level statistics should be stable. The refutation margins (2-5x above thresholds) provide ample buffer even if different data samples showed moderate variance.

## Verdict

REFUTED is correct. The GELU bump zone carries 31.5-42.8% of disagreement energy and its removal changes 12-54% of predictions per block. The bump is not noise; it is a core part of GPT-2's FFN computation. Flag: seed loop is a no-op (zero variance), effectively single-run.
