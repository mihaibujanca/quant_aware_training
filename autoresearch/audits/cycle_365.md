status=PASS
old_outcome=REFUTED
new_outcome=REFUTED
reason=Adaptive k proportional to per-block Frobenius error (PPL 67.63 +/- 0.03) is genuinely worse than uniform k=512 (PPL 66.34 +/- 0.13). Sqrt-proportional fallback also worse (67.52 +/- 0.25). Budget constraint verified (sum=3072 for all modes), k allocation independently reproduced, 3 seeds with deterministic topk+lstsq compression. Eval on held-out 100-sequence split. The mechanism is clear: starving early blocks (b1=256, b2=273 vs uniform 512) increases their frob recon error by 10%, and this propagates through subsequent blocks, outweighing the 15-20% frob improvement in late blocks (b10, b11).
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Item 377 (mixed compression) should NOT use Frobenius-proportional k. Instead, combine with item 376 (block importance pruning) to identify blocks where FFN can be zeroed entirely, then give their budget to remaining blocks uniformly. The finding that early-block error propagation dominates late-block local error is consistent with the composition barrier and argues for prioritizing early-block fidelity.
interesting=true
interesting_reason=Adaptive k allocation that provably reduces total Frobenius reconstruction error still produces worse PPL than uniform allocation. This demonstrates that Frobenius error is a poor proxy for task-relevant error when composition is involved: early-block errors propagate and amplify while late-block errors are terminal. Static per-block budget reallocation cannot overcome the composition barrier because the propagation sensitivity is not captured by per-block Frobenius norms.
