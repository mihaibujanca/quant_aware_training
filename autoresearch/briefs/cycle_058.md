# Cycle 058 Brief

## Item
Item 75 — Multi-block stacking error budget: additive vs compounding degradation — GPT-2 FFN

## Context
**Cycle 057 finding**: Single-block 6× compression on OpenWebText delivers −3 to −5% PPL (within noise), but 3-block stacking costs +20.2% PPL — **stacking is the sole failure mode**, not per-block quality.

**Cycle 054 finding**: Same 6× ratio fails dramatically on OpenWebText (+48%), contradicting early per-block promise. But cycle 057 shows single-block actually works on OWT.

## Research Question
**Mechanism of error compounding?** If errors were additive through the residual stream:
- sum(single-block PPL deltas) ≈ stacked PPL delta
- Then per-block method quality is binding (no stacking problem)

If errors compound super-additively:
- stacked PPL delta >> sum of singles
- Then residual-stream interference dominates (new bottleneck)

## Task
Run k=1 through k=4 on OpenWebText (blocks [7,5,4,1] in easiest-first order):
1. **Per-block marginal PPL cost** (PPL_k − PPL_{k−1})
2. **Sum-of-singles vs actual stacked cost**
3. **Residual-stream MSE** at each insertion point

## Success Criteria
- **Confirmed**: stacked cost ≤ 1.5× sum of singles (sub- or mildly super-additive)
- **Refuted**: stacked cost > 2× sum of singles (strong compounding)
- **Inconclusive**: 1.5–2× ratio (ambiguous)

## Key Hypothesis
Residual attention architecture creates feed-forward cumulative error. Testing whether this is fundamentally compounding or just a linear cascade of independent per-block failures.
