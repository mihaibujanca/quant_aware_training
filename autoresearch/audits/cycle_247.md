# Cycle 247 Audit — Item 257

**Status: PASS** (interesting=true)

## Verdict check
- Item 257 criteria: Confirmed ≤61.0, Refuted ≥65.0, else Inconclusive.
- Result: best_mean = 64.453 ± 0.123 (3 seeds). 64.453 ∈ [61.0, 65.0) → **INCONCLUSIVE** ✓.
- Only 0.547 PPL from the REFUTED threshold; leans toward refuted.

## Phase 2 quality
- 3 seeds {42, 137, 2024}. Per-seed best: 64.621, 64.405, 64.332. Std 0.123 (in-line with cycle 221/231 seed noise).
- Init PPL deterministic (103.931, std ~1e-6) — lstsq fit is seed-free ✓.
- Best-state restore clean (|restored−best|<1e-3 asserted in code).
- Param count 1,184,256 = 50.1% of cycle 221 K=512 6-block. Gap recovery only 38.6% — worse than linear.
- Trajectory: plateau-then-climb same shape as cycles 221/231.

## Phase 3: key finding
The per-block init residual MSE is the headline. Under K_PER_BLOCK={1:128, 2:128, 4:512, 5:512, 7:128, 8:128}:

| block | K | residual MSE |
|---|---|---|
| 1 | 128 | 0.126 |
| 2 | 128 | 0.172 |
| 4 | **512** | 0.162 |
| 5 | **512** | 0.224 |
| 7 | 128 | 0.581 |
| 8 | 128 | 0.818 |

Blocks 7 and 8 (at K=128) have residual MSE 2.6×–3.6× larger than blocks 4 and 5 (at K=512). The plan hypothesis that cycle 243's mid-cascade binding blocks {4, 5} are the K=128 cost drivers is **rejected in the strong form**: the binding constraint under heterogeneous K sits at blocks 7, 8.

**Two distinct measures of block difficulty disagree:**
1. Cycle 243: per-block restoration ΔPPL under fixed K=512 joint fit → ranks {4, 5} as binding.
2. Cycle 247: per-block residual MSE under K=128 topk+lstsq init → ranks {7, 8} as binding.

This is a genuinely interesting negative result: the cycle 243 ranking does not transfer to the heterogeneous-K allocation problem. The concrete pivot is a swap experiment (K=512 on {7, 8}) at the same 1.18M budget to directly test the residual-MSE ranking.

## Audit conclusion
Verdict mapping correct, experiment well-run, finding substantive. PASS, interesting=true.
