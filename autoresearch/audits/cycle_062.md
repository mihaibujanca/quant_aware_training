# Audit — Cycle 062 (Item 78): Reduced Ratio Stacking — GPT-2 FFN

**Auditor**: audit agent, cycle 062
**Status**: FAIL (G1, G2, G3)
**Verdict override**: INCONCLUSIVE result is UNRELIABLE due to confounded baselines.

---

## G1: Implementation

**FAIL — Baseline/eval split mismatch.**

The baseline PPL (57.653) is measured on `eval_val` from `shuffle_seed=42` (line 106 of the experiment script). Per-seed student models are evaluated on `eval_val` from `shuffle_seed=seed` (line 130). Since `_read_text_chunks` uses `shuffle_seed` to shuffle documents before the 80/20 train/val split (lm_geometric_compression.py lines 340-344), each seed gets a **different eval set**.

Delta percentages for seeds 123 and 7 are therefore computed against a mismatched baseline. Evidence: seed 123 at 2x gives PPL=57.24 — *below* the uncompressed teacher baseline of 57.65. A compressed model cannot outperform the teacher on identical data; this only happens because seed 123's eval split is intrinsically easier.

The correct implementation would either:
1. Compute a per-seed baseline PPL (evaluate teacher on each seed's eval_val), or
2. Fix the eval set across all seeds (use `shuffle_seed=42` for eval, vary only the probe/atlas seed).

**Secondary**: `_set_seed(seed)` is called before `probe_ffn_geometry` AND before `derive_ffn_compression` within the ratio loop (line 150). The second `_set_seed` resets the RNG state identically for each ratio within a seed, which is correct (deterministic per config). No issue here.

## G2: Reproducibility

**FAIL — Confounded design makes reproduction meaningless.**

The 3-seed sweep is intended to measure variance of the *compression method*. But the seeds simultaneously vary:
1. The probe corpus (atlas fitting data)
2. The eval corpus (what PPL is measured on)
3. The baseline comparison (fixed at seed=42's eval set)

The observed ~15pp std is a mixture of method variance, corpus difficulty variance, and baseline mismatch. These cannot be separated from the current data. The "seed 42 consistently worst" claim (manifest line 27) likely reflects that seed 42 is the only seed where baseline and student are evaluated on the *same* data, making it the only honest comparison.

To verify: seed 42 at 3x gives +35.4%, but this is the only apples-to-apples measurement. Seeds 123 (+6.2%) and 7 (+14.0%) at 3x are deflated by easier eval splits.

## G3: Results Interpretation

**FAIL — High variance is an artifact, not a finding.**

The manifest claims "High seed variance (std ~15pp) indicates sensitivity to corpus split, not a method bug." This is backwards — the variance *is* a method bug (confounded eval). The manifest further claims "Seed 42 consistently worst" as if seed 42 is an unlucky outlier. In reality, seed 42 is the only correctly-baselined measurement.

If we take only the seed-42 result (the only fair comparison), we get:
- 2x: +27.8%, 3x: +35.4%, 4x: +39.1%, 6x: +42.6%

These are all well above the +20% REFUTED threshold. The verdict would flip from INCONCLUSIVE to **REFUTED**.

The INCONCLUSIVE verdict is driven by averaging in the two deflated (unfairly baselined) seeds.

## G5: Method Failure vs Genuine

The stacking degradation is genuine and monotonic with ratio — this trend is consistent across all seeds even with the confound. The *magnitude* is unreliable due to G1. The claim that "reduced ratio helps" is qualitatively supported (6x > 4x > 3x > 2x in all seeds), but the quantitative improvement (+25.9% to +11.1%) is inflated by the baseline mismatch.

## Calibration Check

6x calibration deviation of 4.3pp (exceeds 3pp threshold) is flagged as attributable to 8192 vs 4096 probe tokens. This explanation is plausible but unverified — a proper calibration requires same eval split.

---

## Recommendations

1. **Fix the experiment**: compute per-seed teacher baseline on the same eval split used for students, OR hold eval fixed across seeds.
2. **Re-run and re-verdict**: the corrected experiment will likely show REFUTED (based on seed-42 data).
3. **Do NOT cite the +18.53% mean or ~15pp std** — these numbers are artifacts of the confound.
4. **Monotonic ratio trend is salvageable**: qualitative conclusion (lower ratio = less degradation) likely holds, just not the magnitudes.
