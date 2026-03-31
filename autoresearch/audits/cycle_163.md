status=PASS
old_outcome=REFUTED
new_outcome=REFUTED
reason=Implementation is correct: progressive stacking compresses blocks {0..i} sequentially, marginals computed as PPL[i]-PPL[i-1], top-3 identified by sorting. Low-rank correction (additive U@V) applied at block output via forward hooks. 3 seeds all agree: top-3 fraction = 0.36 (need >0.60), max single-block = 0.13 (need >0.15). Concentrated corrections (60.3 PPL) are 1.7 PPL worse than uniform (58.6 PPL). Minor design note: conditions A/B compress only 3 blocks while uniform compresses 4, biasing in favor of concentration — yet concentration still loses, strengthening the refutation. Degradation follows a mild monotonic ramp (blocks 0-6: 3-10%, blocks 7-11: 10-12%) with no Pareto structure.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The uniform degradation profile confirms that topk+lstsq at a fixed hidden_fraction produces no block-level outliers in GPT-2. Correction budget should remain distributed. Future items should not revisit per-block concentration unless the compression ratio varies across blocks.
interesting=false
interesting_reason=
