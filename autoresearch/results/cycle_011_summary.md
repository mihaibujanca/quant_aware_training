# Cycle 011 Summary — Single-Affine Collapse Quality: GPT-2 FFN Blocks

**Item:** 21
**Verdict:** REFUTED
**Date:** 2026-03-18

## What was tested

For each of 12 GPT-2 FFN blocks, a single affine map A, b was fit from 1024 probe tokens
(Shakespeare corpus) and evaluated on 512 held-out tokens. The question: is any FFN block
functionally linear on the probe corpus, enabling zero-shot replacement by a single matrix
multiply?

## Key Results

### Per-block normalized residual r_held

| Block | r_fit  | r_held | Cycle-005 damage |
|-------|--------|--------|-----------------|
| 0     | 0.073  | 1.526  | 0.331           |
| 1     | 0.045  | 0.622  | 0.007           |
| 2     | 0.018  | 0.178  | 0.011           |
| 3     | 0.189  | 1.411  | 0.010           |
| 4     | 0.218  | 1.467  | 0.006           |
| 5     | 0.267  | 1.669  | 0.001           |
| 6     | 0.282  | 1.730  | 0.010           |
| 7     | 0.280  | 1.759  | -0.006          |
| 8     | 0.272  | 1.736  | 0.020           |
| 9     | 0.254  | 1.636  | 0.016           |
| 10    | 0.202  | 1.344  | 0.018           |
| 11    | 0.140  | 1.002  | 0.065           |

- Best block: block 2, r_held = 0.178 (above the r < 0.15 threshold)
- All other blocks: r_held >> 0.5
- **Blocks below r=0.15: 0** (need 2 for confirmed)

### Perplexity swap test

- Baseline PPL (eval_val, Shakespeare): 122.3
- Block 2 swap: PPL = 163.6 (+33.7% relative)
- Block 1 swap: PPL = 233.1 (+90.6% relative)
- Both blocks swap: PPL = 543.4 (+344.1% relative)

The swap always causes > 20% PPL increase, even for the best block.

## Verdict: REFUTED

Criteria: refuted if "all blocks have r > 0.5 OR the swap always causes > 20% perplexity
increase regardless of r."

The swap condition is met: the best block (r_held=0.178) causes +33.7% PPL increase.
While block 2 comes closest to linearity, the residual generalizes poorly:
r_fit=0.018 vs r_held=0.178 — a 10x gap — indicating the fit severely overfits
(1024 training points for a 589,824-parameter affine map is heavily underdetermined).

## Observations

1. **Large r_fit/r_held gap:** The affine map is fit to a heavily underdetermined system
   (1024 tokens × 769 augmented dims fitting a 769×768 coefficient matrix with 589K params).
   This explains near-zero r_fit while r_held >> r_fit — the fit is memorizing rather than
   capturing the functional relationship.

2. **Consistency with cycle 009:** Blocks 1-11 have low Jacobian variance L but high r_held.
   This is consistent: low L means the Jacobian is approximately constant (the derivative
   doesn't vary much), but the nonlinear offset (bias from GELU curvature) is large enough
   to prevent a single affine map from generalizing.

3. **Block 2 is special:** It has both the lowest r_fit and r_held. This corresponds to an
   early, lower-capacity block that processes simpler representations. Despite this, the
   affine approximation still causes substantial PPL damage.

4. **Baseline PPL note:** Baseline PPL of 122 on Shakespeare is expected — GPT-2 was not
   trained on Shakespeare and this text is out-of-distribution. The cycle-009 estimate of
   30-40 was incorrect for this corpus.

## Implication for compression

Single-affine collapse is not viable for GPT-2 FFN blocks without significant retraining.
GELU activations produce too much nonlinear variation for a single affine map to capture,
even for the most linear-looking blocks. This motivates experiments 22-23 (piecewise affine
via activation pattern clustering) and experiments 24-25 (RTG spectral coarsening).
