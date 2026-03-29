# Audit — Cycle 085

- **Item**: 97 — GELU activation pattern diversity — GPT-2 FFN
- **Cycle**: 085
- **Auditor verdict**:

```
status=PASS
old_outcome=refuted
new_outcome=refuted
reason=Implementation is correct: consistent linearization d_c = GELU'(z_c) via autograd, Jensen gap validated, hooks on mlp.c_fc (pre-GELU). Refutation is genuine — error saturates around k=128-256 with 5/12 blocks showing error INCREASE at k=256 due to degenerate clusters, and log-linear extrapolation puts the 5% threshold at astronomically large k (10^14+). The 5% threshold is not at issue: even at 20% the errors (0.69-1.95 at k=256) are nowhere close.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Item 98 (centroid affine compression) should be marked blocked/skipped — this route is dead for GPT-2 FFN. Consider pivoting to: (a) input-space clustering (cluster by h not z), (b) low-rank GELU Jacobian approximation instead of piecewise-constant, (c) focus on attention blocks where patterns may be less diverse.
interesting=true
interesting_reason=Error plateau/reversal at k=128-256 (5/12 blocks get WORSE) reveals a fundamental limit: with 2048 tokens and 3072-dim patterns, k=256 hits the regime where clusters are too small for reliable centroid estimation. The problem is not just high diversity — it is that clustering in GELU derivative space has essentially zero useful structure for compression. The 2045/2048 unique binary patterns (at 0.5 threshold) combined with low Hamming distances (0.13-0.29) means patterns are highly individuated but close together — a worst case for clustering (everything is slightly different from everything else).
```

## Guardrail Details

**G1 (Implementation)**: PASS. Consistent linearization d_c = GELU'(z_c) via autograd (line 250-253), not mean(GELU'(z_i)) which was the cycle 084 rejection reason. Hook placement on `model.transformer.h[i].mlp.c_fc` captures pre-GELU activations correctly. Reconstruction formula: y_approx = (GELU(z_c) + d_c * (z - z_c)) @ W_out + b — standard first-order Taylor. Jensen gap tracked and flagged but not used in verdict (correctly: it's a diagnostic).

**G2 (Reproducibility)**: PASS. K-means uses `random_state=42`. Hamming pairs use `default_rng(42)`. No stochastic training. All hyperparams recorded in JSON metadata. Probe corpus from 4 deterministic text sources. The only non-deterministic element is MiniBatchKMeans which has some ordering sensitivity, but results are so far from threshold it doesn't matter.

**G3 (Results interpretation)**: PASS, with nuance. The verdict "K* > 256 for all 12/12 blocks" is correct. More importantly, the error decay is sublinear — the 32x increase from k=8 to k=256 only reduces error by 10-50%, and at k=256 errors are 0.69-1.95 (vs 0.05 threshold = 14-39x above). Log-linear extrapolation gives absurd k values (10^14 to 10^97). Even relaxing to 10% or 20% threshold wouldn't change the verdict — lowest block error at k=256 is 0.69 (690%).

**G4 (Counterintuitive?)**: Not counterintuitive once you think about it. GELU is smooth, so binarizing at 0.5 yields near-unique patterns (each token activates a slightly different set of neurons above 0.5). The low Hamming distance (13-29%) confirms this: patterns differ in relatively few of 3072 dimensions, but just enough to be unique. This is structurally different from ReLU where patterns are exactly binary and many tokens share identical patterns. The smoothness of GELU means "activation pattern" is a continuous 3072-dim vector, not a discrete code — clustering is the wrong abstraction.

**G5 (Learning quality)**: This CLOSES item 97 (GELU patterns are too diverse for k-means clustering) and CLOSES item 98 (centroid affine compression is blocked — the prerequisite failed). It OPENS the question of whether alternative compression strategies (not based on piecewise-constant approximation in GELU derivative space) could work for GPT-2 FFN.

**G6 (Method failure vs genuine refutation)**: Genuine refutation. The reconstruction formula is correct (Jensen gap < 0.05 at k >= 64 for most blocks, confirming linearization itself is sound). The 5% threshold is not the issue — errors are 14-39x above it. Even a 20% threshold would leave 12/12 blocks failing. The error plateau at k=128-256 (with reversal in 5/12 blocks) shows the problem is fundamental: with only 2048 tokens and 3072 dimensions, you run out of data before you run out of pattern diversity. But even with more tokens, the near-maximum binary uniqueness (2045/2048) strongly suggests the piecewise-constant model is wrong for smooth activations.
