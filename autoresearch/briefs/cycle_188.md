# Cycle 188 / Item 198: Progressive C3 composition — per-channel mean replacement across all 12 blocks

**Date**: 2026-04-01

---

## Hypothesis

C3 (per-channel conditional mean replacement) composes across layers when applied progressively to blocks 0..k. Cycle 173 established that C3 costs only +0.4–1.4% per block in isolation. **The critical missing test**: Does this cost accumulate, cancel, or remain bounded when C3 is installed on all 12 blocks simultaneously?

This isolates composition quality of **constant replacement** (no shape change) from the confounds of GELU→ReLU distortion that failed in cycles 181/182. Result discriminates between:
- **Local quality** (C3 per-block: +1.4%) ≠ **stacking quality** (C3 all-12: unknown)
- Whether per-channel conditional means capture sufficient structure to compose

---

## Method

**C3 definition** (from cycle 173):
```
For each token t, channel c where z_tc < 0:
    output_tc = mu_neg_c    (precomputed per-channel conditional mean)
For channels where z_tc >= 0:
    output_tc = GELU(z_tc)  (preserved exactly)
```

- **Data dimension**: 3072 (full intermediate size, per-channel)
- **Mask granularity**: Token-dependent (which channels get replaced varies per input)
- **Shape preservation**: GELU shape unchanged for z ≥ 0
- **Conditioning**: mu_neg and mu_pos learned from calibration data on cycle 173

**Progressive installation**:
1. Compute mu_neg_c, mu_pos_c, mu_all_c for all 12 blocks via hooks over calibration data (N_SEQS=200, SEQ_LEN=1024)
2. Install C3 hooks progressively on blocks 0..k for k=0,1,2,...,11
3. Evaluate PPL at each k
4. Report per-step composition cost and cumulative delta from baseline

**Fork**: `experiments/regime_ablation_table_cycle173.py`
- Modify loop from "6 conditions × 12 blocks in isolation" → "k=0..11 cumulative hooks"
- Reuse `ConditionHook` class and `collect_gelu_stats()` infrastructure
- Output: JSON with ppl_baseline, [ppl_k0, ppl_k1, ..., ppl_k11], incremental costs

---

## Success Criteria

**Confirmed**: C3-all-12 PPL < 26.1 (+15% over baseline 22.7)
- Demonstrates composition preserves quality; per-block cost stays bounded under stacking

**Refuted**: C3-all-12 PPL > 31.8 (+40% over baseline)
- Indicates composition fails; errors accumulate or introduce instability

**Inconclusive**: 26.1 ≤ PPL ≤ 31.8 (+15% to +40%)
- Composition possible but degraded; requires further investigation (e.g., layer-wise tuning)

---

## Expected Outcome

**Most likely**: +5–8% cumulative (PPL ~24.2–24.5)
- Per-block cost (+0.7% avg) compounds sub-linearly across 12 blocks
- Token-dependent masking helps: different channels replaced at different tokens reduces error coherence

**Possible alternate**: +20–30% cumulative (PPL ~27–29.5)
- Errors accumulate despite token dependence
- Per-channel means insufficient to preserve task geometry across composition
- Suggests local quality (per-block) doesn't translate to stacking (cf. MEMORY: "composition barrier")

**Why this matters**: Separates whether the failure of cycles 181/182 was due to (a) GELU→ReLU shape mismatch or (b) fundamental inadequacy of constant replacement. If C3-all-12 succeeds, shape is the culprit; if it fails, replacement alone is insufficient.

---

## Runtime Estimate

- Calibration (collect_gelu_stats): ~3 min (200 sequences × 1024 tokens, 12 blocks)
- PPL evaluations: 12 k-values × ~2 min = ~24 min
- **Total**: ~30 min on MPS

---

## References

- Cycle 173: `experiments/regime_ablation_table_cycle173.py` (per-block regime ablation; C3 within +2%)
- Cycle 181: `autoresearch/briefs/cycle_181.md` (C3-all-12 with GELU→ReLU swap: +88%, REFUTED)
- Cycle 182: `autoresearch/briefs/cycle_182.md` (iterative bias recomputation: +63%, INCONCLUSIVE)
- `docs/gelu_bump_audit.md` (decomposition of error sources: GELU shape cost 3–68x replacement cost)
