# Cycle 093 Plan: Lower Compression Ratio Stacking — GPT-2 FFN

**Item:** 108
**Script:** `experiments/lower_ratio_stacking_gpt2.py` (new, forked from `experiments/compositional_accumulation_curve_gpt2.py`)

## Pre-flights

**P1 — Null hypothesis:** c is independent of compression ratio (stays ~0.41 at 3x and 4x). Would mean per-block error is dominated by output-projection fitting noise, not information loss.

**P2 — Criteria (verbatim from agenda):**
- Confirmed if: c drops to ≤0.20 at 3× AND k=8 stacked PPL ≤ +20% (doubling viable block count).
- Refuted if: c ≥ 0.30 at 3× (compounding barely improves with weaker compression).
- Inconclusive if: c drops but k=8 PPL still > +30%.

**P3 — Novelty vs replication:** Cycle 089 measured c at 6x only. This cycle adds 3x and 4x, fitting the same multiplicative model to each ratio. The ratio-sweep + cross-ratio c comparison is new.

**P4 — Confounds:**
- Polish (500 Adam steps) was tuned for 6x. At 3x/4x the student has more capacity; same step count may undershoot. Mitigated: same polish config across ratios (fair comparison of the compression itself, not the polish).
- Block ordering was optimized for 6x single-block damage. At lower ratios the damage ranking may differ. Mitigated: use the same forward order [7,5,4,1,3,6,8,9] for all ratios (controlled comparison).

**P5 — Cost estimate:** 3 ratios x 3 seeds x (8 block probes + 8 block polishes + 8 accumulation evals) = 216 forward/backward passes. ~15-20 min on MPS.

**P6 — Data dependencies:** Requires `data/openwebtext_sample.txt` (same as cycle 089).

**P7 — Abort conditions:** Any single-ratio run PPL > 5000 at k=8 (diverged). Baseline PPL outside [40, 65] (data issue).

## Implementation

Fork `experiments/compositional_accumulation_curve_gpt2.py` with these changes:

1. **Config:** Replace single `HIDDEN_FRACTION = 1/6` with:
   ```python
   RATIOS = {"3x": 1/3, "4x": 1/4, "6x": 1/6}
   ```
   Keep: `SEEDS = [42, 123, 7]`, `FORWARD_ORDER = [7,5,4,1,3,6,8,9]`, `POLISH_STEPS = 500`, `POLISH_LR = 1e-4`, all corpus sizes unchanged.

2. **Drop reverse ordering.** Only forward (least-damage-first). Saves 50% runtime; reverse adds no signal for this question.

3. **Outer loop:** `for ratio_name, hidden_frac in RATIOS.items()` wraps the existing seed loop. Each ratio gets its own `CompressionBudgetSpec(hidden_fraction=hidden_frac, ...)`.

4. **Reuse atlas across ratios within a seed.** `probe_ffn_geometry` is ratio-independent (it captures hidden activations). Only `_derive_block_pack_from_atlas` uses the budget. Build atlas once per seed, derive packs per ratio.

5. **Accumulation:** For each (ratio, seed), k=1..8 with forward order. Record PPL.

6. **Fitting:** For each ratio, fit `_fit_multiplicative(ks, mean_ppls)` to get c. Also compute k=8 delta%.

7. **Output:**
   - JSON: `autoresearch/results/cycle_093_lower_ratio_stacking.json` — per-ratio, per-seed PPL at each k, fitted c values, baseline.
   - Plot: `autoresearch/results/cycle_093_lower_ratio_stacking.png` — 3 curves (one per ratio) of PPL vs k, with ±1 std bands. Horizontal dashed line at baseline * 1.20 (the +20% threshold).
   - Print summary table: ratio | c | R2 | k=4 delta% | k=8 delta%.

8. **Verdict logic:**
   ```
   if c_3x <= 0.20 and delta_k8_3x <= 20.0: CONFIRMED
   elif c_3x >= 0.30: REFUTED
   else: INCONCLUSIVE
   ```

## Verification

- Sanity: 6x results should approximately match cycle 089 (c ~ 0.41, k=8 PPL ~ 500+).
- Monotonicity: c should satisfy c_3x <= c_4x <= c_6x (weaker compression = less error per block).
- If monotonicity violated, flag as potential bug and inspect single-block deltas.
