# Cycle 094 Plan: Non-Uniform Compression Ratio Allocation — GPT-2 FFN

**Item:** 110
**Script:** `experiments/nonuniform_compression_gpt2.py` (new, forked from `experiments/compositional_accumulation_curve_gpt2.py`)

## Pre-flights

**P1 — Null hypothesis:** Non-uniform allocation gives < 3% improvement over uniform 6×. Polish absorbs single-block damage differences, making allocation irrelevant.

**P2 — Criteria (verbatim from agenda):**
- Confirmed if: non-uniform allocation achieves ≥ 10% relative PPL improvement over uniform 6× at matched parameter budget.
- Refuted if: non-uniform allocation improves PPL by < 3% (single-block damage differences don't matter after polish).
- Inconclusive if: 3-10% improvement.

**P3 — Novelty vs replication:** Cycle 089 measured per-block damage at uniform 6×. Cycle 093 showed c≈0.41 is ratio-independent. This cycle is new: non-uniform per-block ratios at matched total budget.

**P4 — Confounds:**
- Polish tuned for 512 channels (6×). At 3× (1024 ch) or 10× (307 ch), 500 steps may over/underfit. Mitigated: same polish config for all allocations (fair comparison of compression allocation, not polish).
- Damage ordering from cycle 089 was measured at 6×. A block that's low-damage at 6× might not be low-damage at 10×. Mitigated: we test a small set of allocations, not an exhaustive search.

**P5 — Cost estimate:** 3 allocations × 3 seeds × (4 block probes + 4 block polishes + 1 stacked eval) = 45 forward/backward passes. ~8-12 min on MPS. Atlas reused across allocations within a seed.

**P6 — Data dependencies:** `data/openwebtext_sample.txt` (same as cycle 089/093).

**P7 — Abort conditions:** Any allocation PPL > 500 at k=4 (diverged). Baseline PPL outside [40, 65].

## Implementation

Fork `experiments/compositional_accumulation_curve_gpt2.py` with these changes:

1. **Config:**
   ```python
   BLOCKS = [7, 5, 4, 1]  # least-damage to most-damage (from cycle 089)
   SEEDS = [42, 123, 7]
   POLISH_STEPS = 500
   POLISH_LR = 1e-4

   # GPT-2 FFN hidden_dim = 3072. At 6×: 512 channels/block.
   # Total budget = 4 × 512 = 2048 channels.
   TOTAL_BUDGET = 2048

   # Strategy: give MORE channels to high-damage blocks, FEWER to low-damage blocks.
   # Damage order (cycle 089): block 7 = least damage, block 1 = most damage.
   ALLOCATIONS = {
       "uniform_6x":    {7: 512, 5: 512, 4: 512, 1: 512},   # 6× each, baseline
       "mild_nonunif":  {7: 410, 5: 512, 4: 512, 1: 614},   # ~7.5× block 7, 5× block 1
       "aggressive_nonunif": {7: 308, 5: 358, 4: 614, 1: 768},  # 10× block 7, ~8.6× block 5, 5× block 4, 4× block 1
   }
   # Verify: all sum to 2048
   ```

2. **Use `per_layer_hidden`:** For each allocation, build:
   ```python
   budget = CompressionBudgetSpec(
       hidden_fraction=1/6,  # fallback (unused since all 4 blocks have explicit entries)
       min_hidden=32,
       per_layer_hidden=allocation_dict,
       selection_mode="topk",
   )
   ```

3. **Reuse atlas across allocations within a seed.** `probe_ffn_geometry` is ratio-independent. Build atlas once per seed, derive packs per allocation.

4. **Per-block polish then stack.** For each (allocation, seed): polish each block independently (500 Adam steps), then swap all 4 into student and eval.

5. **Output:**
   - JSON: `autoresearch/results/cycle_094_nonuniform_compression.json` — per-allocation, per-seed PPL, plus baseline.
   - Plot: `autoresearch/results/cycle_094_nonuniform_compression.png` — bar chart: 3 allocations, PPL with ±1 std error bars.
   - Print summary table: allocation | mean_ppl | std | delta% vs uniform.

6. **Verdict logic:**
   ```
   best_nonunif_delta = min(mild_delta%, aggressive_delta%)
   if best_nonunif_delta <= -10.0: CONFIRMED
   elif best_nonunif_delta > -3.0: REFUTED
   else: INCONCLUSIVE
   ```

## Verification

- Sanity: uniform_6x results should match cycle 089 k=4 (mean_ppl ≈ 80.7, delta% ≈ 56%).
- Budget check: assert all allocations sum to exactly 2048 channels.
- If aggressive allocation is worse than uniform, the damage ranking may be ratio-dependent — note for future work.
