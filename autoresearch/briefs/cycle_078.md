# Cycle 078: Mixed-corpus lstsq refit with matched sample sizes — GPT-2 FFN

## Item
Item 90: Rerun cycle 076 with matched n=2048 for both conditions. Cycle 076 was confounded: Shakespeare had n=512 (underdetermined lstsq, p=513) while mixed had n=2048. This isolates corpus diversity from sample size.

## Context
- Cycle 076 (item 66): INCONCLUSIVE. Condition A (Shakespeare 512 tok) catastrophically degenerate (PPL ~3.6M) due to n<p lstsq. Condition C (mixed 2048) gave PPL ~113 (+61%). Cannot attribute improvement to corpus diversity vs sample size.
- Cycle 044: Shakespeare-only at 2048 tokens → WikiText-2 eval: PPL ~135 (+94% degradation). This is the expected baseline for condition A' here.
- Cycle 077 audit: single-block PPL misleading. This uses multi-block (4-block) swap, which is more honest.
- Channel selection fixed from Shakespeare for both conditions (same as cycle 076).

## Dependencies
- Item 66 (cycle 076): INCONCLUSIVE (sample size confound). This cycle directly fixes that confound.

## Key Files
- `experiments/mixed_corpus_lstsq_refit_gpt2.py` — cycle 076 script (adapt for 2 conditions at n=2048)
- `autoresearch/results/cycle_076_mixed_refit.json` — prior results showing confound
- `aleph/lm_geometric_compression.py` — compression pipeline

## Constraints
- Two conditions: A' = Shakespeare 2048 tok, C = mixed 2048 tok (512 each from 4 corpora). Both well overdetermined (p=513, n=2048, ratio ~4:1).
- Blocks [7, 5, 4, 1] at 6x. 3 seeds [42, 123, 7]. Eval on WikiText-2 validation.
- Confirmed if: mixed (C) outperforms Shakespeare-only (A') by >= 5% relative PPL.
- Refuted if: Shakespeare-only (A') matches mixed within 3% relative PPL.
