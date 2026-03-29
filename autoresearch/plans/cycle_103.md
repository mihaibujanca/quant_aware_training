# Cycle 103 Plan: GELU-ReLU Disagreement Anatomy — GPT-2 FFN

**Item:** 108
**Script:** `experiments/gelu_relu_anatomy_gpt2.py` (new)
**Revision:** R1 — multi-threshold sweep, corrected cost, signed-side tracking

## Pre-flights

**P1 -- Null hypothesis:** The GELU-ReLU bump zone carries task-critical signal; masking it changes >10% of top-1 predictions.

**P2 -- Criteria:** Confirmed if bump energy <5% AND pred change <1%. Refuted if >20% energy OR >10% pred change. Else inconclusive.

**P3 -- Novelty:** Cycle 102 showed swap is catastrophic (+8320% PPL). This dissects WHERE damage lives by zone occupancy, energy, and token/channel anatomy.

**P4 -- Confounds:** (1) Threshold sensitivity — sweep {0.5, 1.0, 1.5, 2.0, 2.5} and report cumulative disagreement energy captured at each. (2) Energy = squared L2, normalized per block. (3) Masking replaces GELU(z) with ReLU(z) in bump zone, not zero.

**P5 -- Cost:** 12 blocks × 3 seeds × (1 baseline + 1 stats + 12 per-block masks) = 42 forward passes per seed, 126 total. ~30-45 min on MPS.

**P6 -- Data:** OWT eval split, 4096 tokens, same tokenizer as cycle 102.

**P7 -- Abort:** Baseline PPL outside [15, 65]. Bump fraction <0.1% (threshold too tight) or >80% (too loose) at threshold 1.0.

## Implementation

1. **Baseline.** Load `gpt2`, eval PPL on OWT (200 seqs). Verify ~22.26 (cycle 102).

2. **Zone partition with threshold sweep.** Hook each block's FFN pre-activation z. For each threshold T in {0.5, 1.0, 1.5, 2.0, 2.5}: define bump zone as |z| < T. Compute per-block: zone fraction, energy = sum(|GELU(z)-ReLU(z)|^2) / sum(|GELU(z)|^2). Report cumulative disagreement energy captured vs T. Use T=2.5 as primary analysis threshold (captures >99% of |diff|>0.01 region).

3. **Signed-side tracking.** For bump-zone entries, separately tally z<0 (negative leak: GELU leaks small negative, ReLU gives 0) vs z>0 (positive attenuation: GELU suppresses, ReLU passes through). Report fraction and energy contribution of each side per block.

4. **Token & channel anatomy.** For bump zone (T=2.5): aggregate by token position/ID and by channel. Report top-20 tokens/channels by bump fraction. Note correlation with cycle 102 per-block sensitivity.

5. **Masking experiment.** For each of 12 blocks independently, at T=2.5: replace GELU(z) with ReLU(z) for bump-zone entries only. Measure PPL delta and fraction of top-1 prediction changes. 3 seeds.

6. **Verdict logic.**
   ```
   bump_energy = max over blocks of energy fraction at T=2.5 (mean over seeds)
   pred_change = max over blocks of pred change fraction (mean over seeds)
   if bump_energy < 0.05 and pred_change < 0.01:  CONFIRMED
   elif bump_energy > 0.20 or pred_change > 0.10:  REFUTED
   else:                                            INCONCLUSIVE
   ```

7. **Artifacts.** `autoresearch/results/cycle_103_gelu_relu_anatomy.json` (all metrics incl. threshold sweep and signed-side breakdown), `cycle_103_gelu_relu_anatomy.png` (4-panel: threshold sweep curve, zone fraction heatmap, energy pie per block with neg/pos split, prediction change bar chart).

8. **Commit & agenda.** Update item 108. Commit: `[autoresearch] cycle 103: GELU-ReLU disagreement anatomy — GPT-2`.
