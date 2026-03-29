# Research Status — 93 Cycles (001–093)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 26 |
| Inconclusive | 43 |
| Crash | 1 |

## Latest Cycle
- **Cycle 093** (item 108): Lower compression ratio stacking — GPT-2 FFN — REFUTED
  - Compounding rate c≈0.41 is RATIO-INDEPENDENT: 3×=0.406, 4×=0.404, 6×=0.408
  - Single-block PPL improves at lower ratios (3×: −5.5% vs 6×: −3.0%) but washes out in stacking
  - The ~41%/block cascade is structural, not a compression-depth effect
  - Bottleneck is inter-block error propagation dynamics, not per-block quality

## Final Answer (consolidated through cycle 093)

### The Recipe
1. **Topk + lstsq** on probe data — basis and channel selection heuristic both sufficient
2. **Stack k≤4 blocks** — ceiling is STRUCTURAL (c≈0.41 regardless of compression ratio)
3. **500 Adam steps** → ~+5% PPL
4. Lower compression ratios help per-block but not in stacking

### Key Structural Finding
The ~41%/block stacking coefficient is a **network constant**, not compression-dependent. Per-block error is compression-dependent, but the cascade amplification factor is identical at 3×, 4×, and 6×. This cleanly separates the problem: the bottleneck is **inter-block error propagation**, not per-block approximation quality.

### Dead Directions
- All piecewise-linear approaches (085, 088)
- Analytic stacking correction (080-082)
- GELU pruning (083), MI channel selection (091)
- Lower compression ratios for better stacking (093)

## Remaining NEXT Items
- **P0**: Item 103 (low-rank Jacobian)
- **P1**: Items 109 (attention compression v2), 110 (non-uniform compression)
- **P2**: Items 87-88 (Procrustes, learned bottleneck AE)
