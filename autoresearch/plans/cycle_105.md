# Cycle 105 Plan: GELU Negative-Leak Aware Compression — GPT-2 FFN

**Item:** 116
**Script:** `experiments/gelu_leak_aware_compression_gpt2.py` (new)
**Revision:** R1 (addresses I1–I3, A1–A2)

## Pre-flights

**P1 -- Null hypothesis:** Channel sign statistics don't help; topk on positive-dominated channels + bias offset ≈ standard topk+lstsq.
**P2 -- Criteria:** Confirmed if leak-aware ≥3% relative PPL gain vs standard topk+lstsq in ≥2 eligible blocks at matched k. Refuted if ≤1%. Else inconclusive.
**P3 -- Cost:** 4 blocks × 3 seeds × 3 conditions = 36 runs. ~20 min MPS.
**P4 -- Abort:** Baseline PPL outside [25, 65]. Any compressed PPL > 10× baseline.
**P5 -- Data:** `data/openwebtext_sample.txt`, 200 seqs, seq_len=128. Split: first 100 seqs = probe (b_leak + channel census), last 100 seqs = lstsq fitting. PPL eval on held-out data.
**P6 -- Prior:** Cycle 103: 84-88% disagreement from z<0. Cycle 104: cascade attention-driven. If negative-dominated channels carry mostly leak, excluding them focuses topk.
**P7 -- Confounds:** Condition C (I1) disentangles filtering from bias. Probe/fit split (I2) prevents b_leak overfitting. lstsq order clarified (I3).

## Implementation

1. **Baseline PPL.** Load GPT-2, eval PPL. Abort if outside [25, 65].

2. **Channel sign census (probe split).** For blocks [1, 4, 7, 11], collect pre-GELU activations z on probe data (first 100 seqs). Per channel j: `neg_frac[j] = mean(z[:, j] < 0)`. Partition: positive-dominated (`neg_frac < 0.5`), negative-dominated (`neg_frac >= 0.5`). Log N_pos, N_neg per block. Also log per-channel variance of GELU(z_neg) — if variance >> mean, b_leak is unreliable.

3. **Leak bias (probe split).** On probe data only, for negative-dominated channels:
   `b_leak = W_out[:, neg_ch] @ mean_over_tokens(GELU(z) * (z < 0), per channel)[neg_ch]`
   Log `||b_leak||` per block.

4. **Condition A — Standard topk+lstsq.** k=512 from all 3072 channels. lstsq+500 Adam on fit data (last 100 seqs), target = original block output y. Record PPL.

5. **Condition B — Leak-aware topk+lstsq.** k=512 from positive-dominated channels only. lstsq+500 Adam on fit data, target = original output y (NOT residual after b_leak). Then add b_leak as post-hoc constant to reconstruction. Log `||b_leak|| / ||lstsq_residual||`. Record PPL.

6. **Condition C — Positive-only topk, no b_leak (I1).** k=512 from positive-dominated channels only. lstsq+500 Adam on fit data, target = original output y. No bias correction. Record PPL. This isolates filtering effect from bias correction.

7. **Verdict logic.**
   - Exclude blocks where N_pos < k (A1). Log excluded blocks.
   - If N_neg < 100 in all 4 blocks → force INCONCLUSIVE (A2).
   - On eligible blocks, compute per-block mean over seeds:
     `gain_B = (ppl_A - ppl_B) / ppl_A * 100` (leak-aware vs standard)
     `gain_C = (ppl_A - ppl_C) / ppl_A * 100` (filtering-only vs standard)
   - wins = count(eligible blocks where gain_B ≥ 3.0)
   - If wins ≥ 2: CONFIRMED. If all |gain_B| ≤ 1.0: REFUTED. Else: INCONCLUSIVE.
   - Report gain_C alongside to attribute filtering vs bias correction.

8. **Output:** JSON `autoresearch/results/cycle_105_leak_aware.json` — per-block: N_pos, N_neg, b_leak_norm, lstsq_residual_norm, PPL (A/B/C per seed), gains, verdict. Plot: grouped bar (A/B/C × 4 blocks). Save `autoresearch/results/cycle_105_leak_aware.png`.

9. **Agenda update + commit** per standard autoresearch flow.
