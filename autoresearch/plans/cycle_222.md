# Task

Test whether Frobenius-weighted channel selection reduces 3-block cascade init PPL on GPT-2 small (BLOCKS=[1,4,7], K=512), compared to the standard topk by act_std × W_proj_norm. Cycle 207 already refuted this question for single-block PPL; here we measure the cascade metric (3-block init PPL), which is what cycles 206/213/214 optimize.

# Hypothesis

From agenda item 227 (verbatim):
- Confirmed if: Frobenius-optimal selection improves 3-block init PPL by ≥1 PPL (≤60.75)
- Refuted if: improvement <0.3 PPL vs standard init
- Inconclusive if: 0.3-1 PPL improvement

Working hypothesis: Variant B (Frobenius-importance per channel) gives ≥1 PPL cascade improvement. Prior (cycle 207): low — selection was within 1 PPL on single-block. The cascade may amplify any difference, but lstsq W_out adapted in cycle 207 regardless.

# Pre-flight Results

- P1 duplicate: cycle 207 (item 218, REFUTED) measured single-block PPL on block 1 only. Cycle 222 measures 3-block CASCADE init PPL. Different metric. NOT a duplicate.
- P2 dependency: none.
- P3 ≤30 min: yes — selection + lstsq fit + 1 PPL eval per variant, sub-minute compute per variant.
- P4 program/prompts: no edits required.

# Steps

1. Create `experiments/frobenius_selection_cycle222.py`.
2. Reuse cycle 206 calibration loader (200 OWT sequences × 128 tokens) and the same eval set used in cycle 206 (so init PPL is directly comparable to 61.746).
3. Define selection variants. Apply each to BLOCKS=[1,4,7] with K=512:
   - Variant A (BASELINE): standard topk by `act_std[c] * ||W_proj[c]||`. Cite cycle 206 init PPL = 61.746. Re-run with seeds (42, 137, 2024) to get ±std (only the calibration sampling is random; selection itself is deterministic given the calibration set).
   - Variant B (Frobenius-importance, deterministic): for each MLP intermediate channel c, compute
     `score[c] = ||W_proj[c]||^2 * sum_t gelu(z[t,c])^2`
     where z = c_fc(x) on the calibration set and W_proj[c] is row c of c_proj.weight (Conv1D layout (3072, 768)). Sort scores descending; take top K=512.
   - Variant C (greedy forward, optional): start S=∅; for k in 1..K, pick c* = argmin_{c∉S} ||Y - W_proj[S+c] · gelu(z_{S+c})||_F^2 with W_proj effective rows recomputed each step. Skip if per-block runtime >5 min.
4. For each variant and each block independently:
   a. Compute the K=512 channel set.
   b. Build CompressedMLP using only those channels (slice c_fc rows + c_proj cols accordingly).
   c. Fit W_out via lstsq on the calibration set (same procedure as `aleph/lm_geometric_compression.py` topk+lstsq init).
5. Replace blocks [1,4,7] in float GPT-2 with each variant's CompressedMLP.
6. Eval full-model PPL on the same eval split as cycle 206 (no Adam correction — pure init PPL).
7. Compare init PPL across variants vs standard 61.746 baseline:
   - Variant B init PPL vs Variant A mean.
   - Variant C if run.
8. Verdict mapping (item 227 criteria):
   - Confirmed if best variant init PPL ≤60.75 (Δ ≥1).
   - Refuted if best Δ <0.3 PPL.
   - Inconclusive otherwise.
9. Write JSON + summary md (schema below).

Required JSON schema:
```
{
  "cycle": 222, "item": 227, "verdict": "Confirmed|Refuted|Inconclusive",
  "config": {"blocks": [1,4,7], "K": 512, "n_calib": 200, "seq_len": 128, "seeds": [42,137,2024]},
  "variants": [
    {"name": "topk_act_std", "init_ppl_mean": ..., "init_ppl_std": ..., "ppl_per_seed": [...]},
    {"name": "frobenius_importance", "init_ppl": ...},
    {"name": "greedy_forward", "init_ppl": ...}  // optional
  ],
  "improvement_vs_baseline": {"frobenius_importance": ..., "greedy_forward": ...}
}
```

# Validation

- Standard topk reproduces cycle 206 init PPL ≈ 61.746 (within ±0.1 across seeds).
- Variant B's chosen channel set differs from Variant A by ≥1 channel on at least one block (assert non-identity to confirm scoring path is exercised).
- All variants use exactly K=512 selected channels per block (assert).
- Float GPT-2 baseline PPL printed once for reference.

# Artifacts

- `experiments/frobenius_selection_cycle222.py`
- `autoresearch/results/cycle_222_frobenius_selection.json`
- `autoresearch/results/cycle_222_summary.md`
- `autoresearch/results/cycle_222_manifest.txt` (written by execute step)

# Fallbacks

- If Variant C (greedy) per-block runtime exceeds 5 min, drop it and report only Variants A and B. Note in summary.
- If Variant B init PPL is worse than baseline by any margin, report verdict=Refuted with the explicit Δ and a one-line note that lstsq W_out compensates for selection on the cascade as well as on single block.
- If standard topk fails to reproduce 61.746 (delta >0.5), HALT and check calibration loader / eval split parity with cycle 206 before proceeding.

# Update bookkeeping

- Move item 227 from agenda NEXT/P1 to PAST/Cycle 222 with verdict + 1-line outcome.
- TSV row in `autoresearch/results.tsv`.
- Commit: `[autoresearch] cycle 222: Frobenius-weighted channel selection — <verdict>`.
- Update `autoresearch/STATUS.md`.
