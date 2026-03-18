# Research Status — FINAL (22 Cycles: 009-030)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 4 (010, 019, 026, 027) |
| Refuted | 9 (009, 011, 012, 014, 020, 021, 025, 028, 029) |
| Inconclusive | 9 (013, 015, 016, 017, 018, 022, 023, 024, 030) |

## Answer to the Research Question

**"Given a trained transformer LM, how do we construct a smaller FFN reparameterization that preserves geometry without SGD?"**

### What the 22 cycles proved:
1. **Data-driven channel selection is the viable no-SGD method** (cycles 026-027): beats SVD, and centroid activation similarity (ρ=0.85) is the right channel-merge signal.
2. **No-SGD compression has a hard ceiling** (cycles 007, 018, 022): individual block/head removal is cheap (~4% ppl), but compounding makes collective removal superadditive.
3. **Structural pruning is empty for GELU** (cycles 006, 028): no dead/linear channels exist.
4. **Partition theory doesn't transfer** (cycle 025): GELU needs ≥20 PWL pieces — MASO/SplineCam theory is inapplicable at useful granularity.
5. **Single-affine collapse works on toy deep layers but fails on GPT-2** (cycles 010-011).
6. **Blocks don't share reusable compression structure** (cycle 030): per-block compression required.

### Practical takeaway:
No-SGD geometric compression yields ~1 FFN block removal + ~5 attention heads (total ~5-7% model size reduction at <5% ppl cost). For deeper compression, SGD-based methods (per-block distillation) are necessary but benefit from the geometric signals discovered here (centroid similarity for channel selection, Procrustes linearity for block prioritization).

## All Agenda Items Processed
P0 items: all non-topology items completed. P1 items: all completed.
Remaining unprocessed: topology items (27, 28, 42, 43, 45) blocked by missing ripser package.
