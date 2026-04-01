# Cycle 172 Brief — Item 182

## Item
182. Lstsq regression target: residual-stream delta vs raw output — GPT-2 FFN k=4

## Context
Current lstsq fits `W_compressed @ x ≈ W_float @ x` (raw FFN block output).
But FFN output enters a residual stream: `y = x + FFN(x)`. The compositionally
relevant quantity is the residual-stream state, not the FFN output alone.

## Baseline
4 blocks [1,4,7,11] @ 6x (hidden=512), topk+lstsq, 500 Adam polish.
PPL ~74 unpolished, ~59 polished (cycle 171 numbers).

## Alternatives to test
- **(a) norm-weighted**: fit `FFN(x) / ||x||` — scale-invariant residual delta
- **(b) SVD-subspace**: project targets onto top-k SVD directions of `X @ W_float.T`
  (proxy for Jacobian top-k; full Jacobian too expensive per block)
- **(c) token-frequency-weighted lstsq**: weight rows by unigram frequency

## Verdict criteria
- CONFIRMED: any alt improves stacked PPL >= 2 over raw lstsq
- REFUTED: all alts within 1 PPL of raw lstsq or worse
- INCONCLUSIVE: improvement 1-2 PPL
