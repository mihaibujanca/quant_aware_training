# Cycle 062 — Item 78: Reduced Compression Ratio for Stacking

**Objective:** Test whether reducing per-block compression ratio improves stacking quality. Single-block 6× baseline is sub-baseline, but stacking k=4 blocks at 6× gives +30% PPL. If compounding is super-additive in ratio, reducing per-block ratio should yield disproportionate benefit.

**Task:** Run stacked k=4 (blocks [7,5,4,1]) on OpenWebText at per-block ratios {2×, 3×, 4×, 6× baseline}. Each block uses topk + lstsq (no gradient polish). Compare stacking degradation curves across ratios.

**Hypothesis:** k=4@3× achieves ≤ +5% PPL degradation (vs +30.2% at 6×, cycle 058).

**Context:**
- Cycle 061 confirmed 500 Adam gradient-polish steps reduce stacking degradation from +30.2% to +4.9%
- This item tests ratio reduction WITHOUT gradient polish — pure analytic compression
- Success would validate super-additive compounding (ratio reduction >> per-block improvement)
- Failure would suggest stacking bottleneck persists even at moderate compression

**Success Criteria:**
- Confirmed: k=4@3× achieves ≤ +5% PPL
- Refuted: k=4@3× shows ≥ +20% PPL
- Inconclusive: k=4@3× shows +5–20% PPL

**Dependencies:** None.

**Prior Art:** Cycles 058 (stacking penalty discovery), 061 (gradient polish confirmation).
