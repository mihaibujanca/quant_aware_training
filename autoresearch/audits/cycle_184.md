status=PASS
old_outcome=refuted
new_outcome=refuted
reason=Diffuse divergence (rank_90=300-434 out of 768) correctly triggers the rank>256 refutation criterion. Exponential growth (R2=0.972) and smooth alpha interpolation (monotone PPL, sharp_ratio=2.17) are genuine but insufficient — the hypothesis required ALL three conditions for confirmation, and rank fails decisively. Code correctly hooks block outputs for residual capture, c_fc outputs for z, and applies iterative bias with proper sign and ordering.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Diffuse divergence rules out low-rank targeted GELU-to-ReLU correction in the residual stream. The exponential cascade rate (a=0.223/block) combined with block-11 self-correction (rank drops to 215) suggests the LM head / final LayerNorm partially absorbs diffuse error. Item 194 (compression ratio sweep) is the right next step — characterize the PPL-vs-block-count tradeoff curve since block count is confirmed as the primary lever.
interesting=true
interesting_reason=Block 11 shows anomalous self-correction: rank_90 drops from 390-434 (blocks 8-10) to 215, and divergence norm decreases from 0.34 to 0.27. This suggests the final transformer block or subsequent LM head partially absorbs diffuse GELU-ReLU divergence — worth investigating whether this is a general property of autoregressive LMs.
