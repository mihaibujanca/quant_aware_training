status=PASS
old_outcome=REFUTED
new_outcome=REFUTED
reason=Item 179 asks: does b_leak shift under compression, and does fixing it help? The experiment correctly compresses k=4 blocks at 6x (topk+lstsq), computes b_leak in output space (768-dim, after W_down projection) on both float and compressed models, and measures relative shift per block. Results: uncompressed blocks shift <10% (mean 5.1%), well below the 20% confirmed threshold. Compressed blocks shift 59-93% but this is expected — they have entirely different output projections from lstsq fitting. Fixing b_leak at compressed blocks to float values makes PPL dramatically worse (52.3 -> 119.5), confirming the compressed model's new b_leak is correct for its new weights. Code correctly handles both HuggingFace Conv1D (original MLP) and nn.Linear (CompressedMLP) weight layouts. Baseline PPL=35.2 is within sanity bounds. REFUTED: b_leak is stable where it matters (uncompressed blocks), cascade is not caused by bias drift.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The b_leak stability at uncompressed blocks (even those downstream of 4 compressed blocks showing only 7-10% shift) confirms the cascade damage is in the linear signal, not the bias. Combined with cycle 166 (b_leak is a constant bias) and cycle 163 (error is diffuse), the cascade mechanism is: compression distorts the active linear map W_down @ diag(gelu'(z)) @ W_up, and this distortion propagates through attention and residual connections. The bias is a spectator.
interesting=false
interesting_reason=Negative result confirming bias stability. Expected given that b_leak is only 8-10% of the total signal (cycle 166) — even if it shifted, the effect would be small.
