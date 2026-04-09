# Cycle 378 Brief — Held-Out Verification of Adapter Finding on Wikitext-2-Test (Item 386)

## Context
- c377 trained float+adapter (h=64, 297K params) on wikitext-2-val, evaluated on wikitext-2-val: PPL 38.94.
- Compressed+adapter reference (h=128, 595K params): PPL 47.43 on same val split.
- Float baseline (no adapter): PPL 53.89 on val.
- The c377 audit flagged that both train and eval come from the same wikitext-2-validation split. While sequences don't overlap (200 train, 100 eval), they share distributional properties. The adapter may be domain-adapting to val-specific patterns.
- The gap (float+adapter 38.94 vs compressed+adapter 47.43 = 8.5 PPL) is the basis for the claim "adapter capacity > compression benefit." If this gap shrinks on held-out test data, the claim weakens.

## Hypothesis
The relative ordering (float+adapter < compressed+adapter < float baseline) holds on wikitext-2-test, with the gap between float+adapter and compressed+adapter remaining >= 6 PPL.

## Design
- Train both float+adapter and compressed+adapter on wikitext-2-val (identical to c377 and prior cycles).
- Evaluate both on wikitext-2-TEST (never seen during training).
- Also evaluate float baseline on test for reference.
- 3 seeds each (42, 137, 2024).

## Success Criteria
- **Confirmed** (gap >= 6 PPL): held-out gap is robust, compression genuinely hurts adapter performance.
- **Refuted** (gap < 2 PPL): val split was unreliable, gap was domain-adaptation artifact.
- **Inconclusive**: gap in [2, 6).

## Key References
- c377 results: `autoresearch/results/cycle_377_adapter_only.json` — float+adapter PPL 38.94
- c377 template: `experiments/gpt2_adapter_only_cycle377.py`
- Compressed+adapter: cycle 367, h=128, PPL 47.43
- Agenda item 386 thresholds: gap >= 6 CONFIRMED, gap < 2 REFUTED
