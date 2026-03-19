# Cycle 043 Brief — Item 61

## Task
**Multi-point residual correction at every compressed block boundary — GPT-2 FFN**

Cycle 039 applied a single lstsq correction after the last compressed block and recovered 3.6pp (k=5: +16.1%->+12.5%) and 1.2pp (k=4: +9.6%->+8.4%). This experiment places N corrections — one after each compressed block — fit sequentially so each corrects the residual left by prior corrections.

## Hypothesis
Error accumulates progressively through compressed blocks. Intercepting at each boundary captures block-local error before it compounds, yielding better recovery than a single endpoint correction.

## Success Criteria
- **Confirmed**: k=5 multi-point <= +12% PPL OR k=4 improves >= 1.5pp over single-point (+8.35%).
- **Refuted**: multi-point does not improve over single-point for either k.
- **Inconclusive**: k=4 improves but k=5 does not, or gains < 0.5pp.

## Dependency
- Item 58 (cycle 039): CONFIRMED — single-point correction baseline established.

## Key Numbers
- Baseline PPL: 120.32
- k=4 single-point corrected: +8.35% (130.37)
- k=5 single-point corrected: +12.53% (135.39)
