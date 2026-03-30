# Item 141: Attention entropy shift under FFN compression — GPT-2

## Context
Cycle 123 (item 140) decomposed attention error into QK vs V components. Found: V dominates L2-L4 (V_frac ~0.66-0.68), QK dominates L9-L11 (QK_frac ~0.55-0.58). No single component >70% in any layer; symmetric no-surgical-target conclusion. Cross-seed std < 0.02 (very stable).

Cycle 104 (item 124) identified cascade mechanism: attention amplifies FFN compression error by 1.14–2.26× in transitions 0–4, attention-driven in 8/11 transitions. Uniform across heads (cycle 113 confirmed).

Item 141 bridges these: measure whether FFN compression *changes* attention distribution itself (entropy), and whether this correlates with cycle 104's per-transition amplification fractions.

## Dependencies
None — can run independently. Item 140 complete (cycle 123). Cycle 104 cascade data available for correlation.

## Key Files
- **Cycle 123 decomposition**: `autoresearch/results/cycle_123_qk_v_decomposition.json` — QK/V fractions per layer/seed
- **Cycle 104 cascade**: `autoresearch/results/cycle_104_cascade_source.json` — attn_frac per transition (amplification source metric)
- **Item text**: `autoresearch/agenda.md` lines 9–14

## Constraints
- **Type**: analysis (interpretability)
- **Data**: float vs k=4 compressed (same as cycle 123), 2048 OWT tokens, 3 seeds [42, 123, 7]
- **Metric**: per-head attention entropy H = -Σ_j a_j log a_j (cross-entropy, not differential)
- **Report**: mean entropy shift (compressed - float) per head per layer; Spearman corr with cycle 104 attn_frac
- **Confirmed**: mean |entropy shift| > 0.1 nats in ≥8/12 layers AND |Spearman| > 0.3
- **Refuted**: |entropy shift| < 0.02 nats everywhere
- **Inconclusive**: shift exists but no correlation

## Handoff
Measure whether FFN compression makes attention heads more peaked (overconfident, high entropy decrease) or more diffuse (lost specificity, entropy increase). Correlate per-transition entropy changes to cascade amplification fractions from cycle 104. If both high correlation AND mean |shift| > 0.1 nats in most layers, confirms attention is *restructuring* under compression (not just amplifying raw V/QK error). Null result (no shift, no correlation) suggests cascade is purely V-space phenomenon.
