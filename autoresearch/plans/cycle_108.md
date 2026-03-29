# Cycle 108 Plan: Pre-activation Distribution Anatomy — GPT-2 FFN

**Item:** 119
**Script:** `experiments/preactivation_anatomy_gpt2.py` (new)

## Pre-flights

**P1 — Null hypothesis:** Pre-activation concentration in |z|<0.5 is fully explained by per-channel variance; Gaussian N(μ_j, σ²_j) predicts zone fractions within 5%.

**P2 — Criteria:** Confirmed if empirical zone fractions match Gaussian within 5% relative for ≥10/12 blocks. Refuted if >20% excess for ≥6/12 blocks. Else inconclusive.

**P3 — Novelty:** First per-channel distributional characterization. Prior cycle 103 reported aggregate zone fractions without separating variance from active concentration.

**P4 — Confounds:** (1) Same 4096 mixed-corpus tokens as cycles 089+. (2) KS test p-values Bonferroni-corrected for 3072 channels. (3) Importance scores from `probe_ffn_geometry` use same token set.

**P5 — Cost:** ~3 min. Single forward pass + per-channel stats. No compression, no PPL eval.

**P6 — Data:** `data/openwebtext_sample.txt`.

**P7 — Abort:** Baseline PPL outside [25, 65]. Any block yields NaN in z.

## Implementation

1. **Load model.** GPT-2 (124M), eval mode. Verify baseline PPL (200 seqs, len=128, batch=4).

2. **Collect pre-activations.** For all 12 blocks: hook `mlp.c_fc` output (pre-GELU). Collect z of shape (4096, 3072) from mixed-corpus tokens.

3. **Per-channel Gaussian fit.** For each channel j in each block: μ_j = mean(z_j), σ_j = std(z_j). KS test of z_j against N(μ_j, σ²_j), store p-value.

4. **Zone fractions.** For T in {0.25, 0.5, 1.0, 1.5, 2.0}: empirical_j = fraction of tokens with |z_j| < T, gaussian_j = Φ(T-μ_j)/σ_j) - Φ((-T-μ_j)/σ_j). Aggregate per-block: mean over channels. Compute relative excess = (empirical - gaussian) / gaussian × 100%.

5. **Histograms.** Per-block: histogram of σ_j (50 bins), histogram of μ_j (50 bins). Store bin edges and counts.

6. **Bump-dominated channels.** Flag channels with σ_j < 0.5 AND |μ_j| < 0.5. Report count and fraction per block.

7. **Importance correlation.** Run `probe_ffn_geometry` on same tokens to get `channel_scores` per block. Compute Spearman(σ_j, channel_scores_j) per block.

8. **Verdict.**
   ```
   Per block: max relative excess across T thresholds.
   match_blocks = blocks where max excess <= 5%
   excess_blocks = blocks where max excess > 20%
   match_blocks >= 10:    CONFIRMED
   excess_blocks >= 6:    REFUTED
   else:                  INCONCLUSIVE
   ```

9. **Output.** JSON: `autoresearch/results/cycle_108_preactivation_anatomy.json` — per-block: σ histogram, μ histogram, zone fractions (empirical + gaussian + relative excess) at each T, KS p-values summary, bump-dominated count, Spearman ρ, verdict. Plot: `autoresearch/results/cycle_108_preactivation_anatomy.png` — 2×2 grid: (a) σ distributions across blocks, (b) zone fraction comparison at T=0.5, (c) bump-dominated fraction by block, (d) Spearman ρ by block.

10. **Agenda update + commit** per standard flow.
