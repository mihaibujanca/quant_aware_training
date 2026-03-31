status=CORRECTED
old_outcome=inconclusive
new_outcome=refuted
reason=PPL 73.0 >= 69.3 (the stated REFUTED threshold of "within 2% of topk+polish 70.7"). The code's verdict logic (line 292) would produce REFUTED for these numbers, but the committed JSON says INCONCLUSIVE — the verdict logic was likely changed post-run. Warm-start joint training does not beat topk+polish; this is a clear refutation, not an ambiguous result.
guardrail_flags=G3_verdict_mismatch
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Joint training from any init (random, warm, higher-rank) consistently fails to beat topk+lstsq+polish. Five consecutive cycles (153-158) refuted or inconclusive on joint training variants. This line of inquiry is exhausted — move to item 169 (correction-first compression) or other non-joint approaches.
interesting=false
