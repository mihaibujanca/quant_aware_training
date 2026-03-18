# Audit — Cycle 019

## Item
37 — Activation pattern entropy cascade — toy ReLU

## Phase 1: Verdict Check

Claimed outcome: **confirmed** — conditional entropy drops ≥50% for layers 2-4, and these layers have low single-affine residual.

Criteria: "Confirmed if: conditional entropy drops ≥50% for at least one layer AND low-conditional-entropy layers also have low single-affine residual (item 20 correlation)."

Evidence:
- L2 cH = 1.64 bits (drop ratio 0.259 from L1's 6.33 → 74% drop). ≥50% ✓
- L3 cH = 0.255 bits (drop ratio 0.040 → 96% drop). ≥50% ✓
- L4 cH = 0.271 bits (drop ratio 0.043 → 96% drop). ≥50% ✓
- r_affine from cycle 010: L2=0.050, L3=0.033, L4=0.024 — all < 0.15 ✓
- Correlation: low cH layers (L2-L4) have low r (< 0.05), high cH layer (L1) has high r (0.168/0.43)

**Verdict is correct.** Both criteria met clearly.

## Phase 2: Quality Checks

### G1: Methodology
- Architecture 2→32×4→1 matches cycle 010 (which used 2→64×4→1). The difference in hidden width is noted but doesn't invalidate the comparison since the r values from the execution match the pattern from cycle 010.
- Shannon entropy computed correctly: H = -Σ p_i log2(p_i).
- Conditional entropy via joint pattern concatenation: H(Li|prev) = H(prev, Li) - H(prev). Standard chain rule.
- Test set of 1000 points — sufficient for entropy estimation with ~136-375 patterns.

### G2: Reproducibility
Single seed (42). The effect size is massive (96% conditional entropy drop). Robust.

### G3: Tautology Check
**Partial concern**: If deep layers have many dead/always-active neurons (from cycle 010: 17% dead, 27% always-active at L4), then by construction their patterns have low entropy relative to L1. The conditional entropy drop is partly a consequence of neuron stability, not an independent signal.

However, the correlation with single-affine residual is the key finding — it's not tautological that low conditional entropy implies low affine residual. The two signals come from different computations and their alignment is informative.

### G4: Scope
Within scope. Directly addresses partition redundancy through depth.

## Status: PASS

original_outcome: confirmed
audited_outcome: confirmed
guardrail_flags: G3 (partial tautology — deep layer entropy drop partly from neuron death, but correlation with r is genuine)
reason: Clear confirmation with massive effect size. Deep layers add essentially zero new partition information, and this co-locates with affine compressibility. The entropy cascade provides an information-theoretic explanation for why cycle 010's single-affine collapse works.
recommendations: This confirms the toy pattern: L1 does the work, L2-L4 are near-redundant. But cycle 011 showed this doesn't transfer to GPT-2. The next question is whether GPT-2 has ANY layers with low conditional information gain — this would be a GPT-2 analog experiment. Also connects to cycle 018's Procrustes finding: high Procrustes linearity should correspond to low conditional entropy in the residual stream.
