# Cycle 087 — Item 96 Context

## Task
Mixed-corpus stacking evaluation on GPT-2 FFN. Test whether mixed-corpus refit (confirmed in cycle 078 for single blocks) transfers to multi-block stacking.

## Dependency Status
**Item 90** (cycle 078) — CONFIRMED. Mixed-corpus probe (512 tokens each: Shakespeare, WikiText-2, OWT, C4) outperforms Shakespeare-only by +19.1% PPL on WikiText-2 validation.

## Key Setup
- **Blocks**: [7, 5, 4, 1] (easiest-first stacking order, from cycle 061 baseline)
- **Compression**: 6× per block (k=64, baseline n=384 for 4096-dim FFN)
- **Refit**: lstsq only, then +500 Adam steps (lr=1e-4, W_down only)
- **Eval**: Mixed-corpus refit vs OWT-only refit, both on OWT and WikiText-2
- **Success threshold**: mixed-corpus reduces WikiText-2 PPL by ≥10% relative vs OWT-only, no degradation >3% on OWT

## Code Assets
- Single-block baseline: `experiments/mixed_corpus_matched_refit_gpt2.py` (cycle 078)
- Stacking baseline: `experiments/gradient_polish_scaling_gpt2.py` (blocks [7,5,4,1], 500 Adam steps)
- Need: adapt mixed_corpus probe into stacking pipeline

## Context
- Cycle 078 found lstsq conditioning dramatically improves with corpus diversity (19.1% relative gain)
- Cycles 061, 071, 077 confirmed: basis irrelevant, refit dominates
- Recipe is now: mixed-corpus random projection + lstsq + gradient polish
- This item validates whether diversity helps in the full stacking regime (not just single blocks)
