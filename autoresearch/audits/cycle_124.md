status=PASS
old_outcome=INCONCLUSIVE
new_outcome=INCONCLUSIVE
reason=Verdict is correct: only 2/12 layers exceed 0.1 nats (not CONFIRMED), but 10/12 exceed 0.02 nats (not REFUTED). Implementation is clean — entropy formula correct, manual QKV hook avoids sdpa issues, layers 0-1 being exactly zero is expected (no upstream compressed FFN). 3 seeds with cross-seed std=0.010 (very stable). One caveat: the Spearman rho=-0.55 reported in the agenda is inflated by including layer 1 (trivially 0.0, paired with the highest attn_frac=1.64). Excluding this trivial point, rho drops to -0.39, p=0.26 (nonsignificant). The correlation claim in the writeup should not be relied upon.
guardrail_flags=G3-fragile-correlation (rho=-0.55 driven by trivial zero at layer 1; rho=-0.39/p=0.26 without it)
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The signed delta_H shows most layers have entropy DECREASE (attention sharpening) under compression, with Spearman vs attn_frac = +0.72, p=0.013 — this may be the real signal worth investigating. If pursuing attention-compression interaction further, (a) exclude trivially-zero layers from correlation, (b) analyze signed delta_H, and (c) decompose per-head vs per-layer effects.
interesting=true
interesting_reason=Attention entropy predominantly decreases (sharpening) under FFN compression in middle/late layers. The signed correlation with cascade amplification is positive and significant (rho=+0.72, p=0.013), suggesting high-cascade-fraction layers resist entropy change while low-cascade layers sharpen. This was invisible in the unsigned analysis.
