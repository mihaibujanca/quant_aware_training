# Cycle 106 — Gradient Polish on k=5 Stacking

## Agenda Item
Item 101: Gradient polish on k=5 stacking — GPT-2 FFN

## Context
- **k=4 baseline**: unpolished +30% PPL, polished +5% PPL (500 Adam steps reduce error by 6×)
- **k=5 baseline**: unpolished +17% PPL (cycles 033-038), **never polished**
- **Working recipe**: topk + lstsq + 500 Adam, k≤4 viable at 6× compression
- **Question**: Does polish work on k=5, or is the +17% wall structural?

## Plan
Apply gradient polish to best k=5 configs from cycle 038:
- Configs: [7,5,1,3,6] and [7,5,4,1,6] at 6× compression
- Optimizer: joint Adam on W_down, {0, 50, 200, 500} steps, lr=1e-4
- Data: 3 seeds, OWT, 200 eval sequences
- Metrics: target PPL polished ≤+10%

## Success Criteria
- **Confirmed**: polished k=5 ≤+10% for ≥1 config (polish generalizes)
- **Refuted**: polished k=5 >+15% (wall is structural, k≤4 hard boundary)
- **Inconclusive**: +10-15% range (ambiguous)

## Prior Work
- Cycles 033-038: k=5 +17% unpolished, not explored further
- Cycle 098: k=4+k=4 super-additive joint compression viable
- Cascade mechanism (cycle 104): attention c≈0.07 + FFN c≈0.41 = 1.93× additive amplification
