# Audit — Cycle 129

item=100
status=PASS
old_outcome=confirmed
new_outcome=confirmed
reason=Script verdict was REFUTED due to inverted decision logic (computed factor=random/trained=0.26, compared to >2.0 threshold). Correct ratio is trained/random=3.8-7.8x across blocks 4-7, clearly confirming isotropy is learned. Manifest and agenda already carry the corrected CONFIRMED outcome.
guardrail_flags=none
recommendations=Fix the decision logic in the experiment template (factor should be trained/random, threshold should be >2x for confirmation). Block 1 anomaly (1.5x) worth revisiting — may indicate partial architectural isotropy bias in early layers.
interesting=true
interesting_reason=Clean separation: random-init isotropy is uniformly low (~0.002) regardless of block or seed, while trained isotropy is 7-8x higher in blocks 4-7 but only 1.5x in block 1. This confirms the eigenspectrum structure is information-bearing (learned), not a consequence of weight initialization or activation function. Supports using spectral-aware channel selection over naive topk.
