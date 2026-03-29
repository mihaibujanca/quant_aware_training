# Cycle 112 Plan: GELU Negative-Leak Aware Compression (Corrected) — GPT-2 FFN

**Item:** 123 | **Script:** `experiments/gelu_leak_aware_compression_corrected_gpt2.py` (new)

## Criteria (verbatim from agenda)
- **Confirmed if:** leak-aware (B) outperforms standard (A) by >=3% relative PPL in >=2/4 blocks.
- **Refuted if:** leak-aware within 1% of standard in all blocks.
- **Inconclusive if:** 1-3% or block-dependent.

## Pre-flights
**P1 -- Null:** Filtering positive channels + constant bias offset = standard topk+lstsq.
**P2 -- Cost:** 4 blocks x 3 conditions x 3 seeds = 36 runs. ~25 min MPS.
**P3 -- Abort:** Baseline PPL outside [25, 65]. Any condition PPL > 10x baseline.
**P4 -- Data:** OWT sample, 200 seqs, seq_len=128. Split: first 100 = probe (b_leak + census), last 100 = lstsq fit. PPL on held-out data (separate eval set).
**P5 -- Prior:** Cycle 109: leak is 87-99% constant bias. Cycle 111: neg_frac ~73-87%.

## Implementation

1. **Baseline PPL.** Load GPT-2, eval on held-out data. Abort if outside [25, 65].

2. **Channel sign census (probe split).** For blocks [1,4,7,11], collect pre-GELU z on
   probe data. Per channel j: `neg_frac[j] = mean(z[:,j] < 0)`. Partition: pos-dominated
   (neg_frac < 0.5), neg-dominated (neg_frac >= 0.5). Log N_pos, N_neg per block.
   **Abort guard:** If N_pos < 512 for any block, set k_eff = N_pos for that block's B/C
   conditions and set k_eff for A to match (fair comparison at same k).

3. **Leak bias (probe split).** On probe data, for neg-dominated channels:
   `b_leak = W_out[:, neg_ch] @ mean_per_channel(GELU(z) * (z < 0))[neg_ch]`
   Log ||b_leak|| and ||b_leak|| / ||y|| per block.

4. **Condition A -- Standard topk+lstsq.** k from all 3072 channels (k=512, or k_eff if
   reduced per step 2). lstsq on fit data, target = y. +500 Adam steps. Record PPL.

5. **Condition B -- Leak-aware.** k from positive-dominated channels only.
   **lstsq target = y - b_leak** (THE fix from cycle 105).
   +500 Adam steps on same target. Final reconstruction = lstsq_approx + b_leak.
   Record PPL.

6. **Condition C -- Positive-only, no b_leak.** k from positive-dominated only.
   lstsq target = y. +500 Adam steps. No bias correction. Record PPL.

7. **Metrics.** Per block per seed: PPL(A), PPL(B), PPL(C).
   `gain_B = (PPL_A - PPL_B) / PPL_A * 100` (positive = B better).
   `gain_C = (PPL_A - PPL_C) / PPL_A * 100`.

8. **Verdict logic.**
   - Exclude blocks where abort guard fired AND N_pos < 256 (too few channels).
   - wins = count(eligible blocks where mean-over-seeds gain_B >= 3.0).
   - If wins >= 2: CONFIRMED.
   - If all eligible |gain_B| <= 1.0: REFUTED.
   - Else: INCONCLUSIVE.
   - Report gain_C to decompose: filtering effect vs bias correction.

9. **Output.**
   - JSON: `autoresearch/results/cycle_112_leak_aware.json`
   - Plot: grouped bar chart (A/B/C x 4 blocks), `autoresearch/results/cycle_112_leak_aware.png`
   - Manifest: `autoresearch/results/cycle_112_manifest.txt`
