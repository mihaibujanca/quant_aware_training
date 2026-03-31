status=PASS
old_outcome=INCONCLUSIVE
new_outcome=INCONCLUSIVE
reason=Best correction-first PPL 60.0 vs Phase A 59.2 (delta -0.81, within noise at std=7.4). Verdict INCONCLUSIVE per criteria [57,62]. However, the experiment has limited informativeness: (a) first-order approximation (I+C)^{-1} ~ I-C is invalid at ||C_b||~1.0, (b) Y_target uses float activations while C_b was trained on compressed activations (domain mismatch), (c) transferred corrections degrade PPL (75.3 vs 80.2 uncorrected), confirming non-transferability.
guardrail_flags=G4,G6
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The correction-first pipeline has a fundamental chicken-and-egg problem: corrections depend on the compressed model, but correction-first compression depends on the corrections. An iterative alternating scheme (compress -> correct -> recompress -> re-correct) might converge but adds complexity for marginal gain. Recommend deprioritizing this direction. If revisiting, use the exact inverse (I+C_b)^{-1} instead of first-order approximation, and compute Y_target from compressed-model activations (not float).
interesting=true
interesting_reason=Transferred corrections degrade PPL worse than no corrections (75.3 > 80.2 uncorrected), meaning corrections trained on one compressed model are actively harmful on a differently-compressed model. This confirms correction subspaces are tightly coupled to the specific compression, not generic error-correction directions.
