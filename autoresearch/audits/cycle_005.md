status: PASS
original_outcome: confirmed
audited_outcome: confirmed
guardrail_flags: proxy-only,recovered-artifact
reason: The item-10 verdict is recomputed directly from the raw proxy sweep in `runs/nanochat_single_block_localization_overnight_main/results.jsonl` using the latest successful record for each `(layer, hidden_fraction, method)` pair. Of 24 geometric swaps at `>=4x` FFN compression, 22 satisfy both agenda conditions (`<=10%` relative perplexity increase and full-model perplexity within `1.25x` of the distilled baseline). Only layer `0` fails, at both tested budgets, so the proxy item is clearly confirmed.
recommendations: Treat layer `0` as the canonical hard block for item 12 and later analytic-method follow-ups. Keep the proxy label explicit, and rerun representative pairs with a stronger distillation control if later comparisons depend heavily on the baseline quality.
