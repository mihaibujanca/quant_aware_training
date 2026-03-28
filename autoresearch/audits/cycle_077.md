status=PASS
original_outcome=REFUTED
audited_outcome=REFUTED
guardrail_flags=G4
interesting=true

## Phase 1: Verdict Check

The experiment compares SVD+lstsq vs random+lstsq at k/n in {0.3, 0.2, 0.1} on 4 blocks with 3 seeds. The verdict criteria:
- Confirmed if: SVD+lstsq outperforms random+lstsq by >=5% relative PPL at k/n<=0.3 in >=3/4 blocks
- Refuted if: random+lstsq matches SVD+lstsq within 3% at all k/n in all 4 blocks

Results (mean PPL across 3 seeds):

| k/n | Block | SVD+lstsq | Random+lstsq | Rel diff |
|-----|-------|-----------|--------------|----------|
| 0.3 | 1     | 122.4     | 122.4        | +0.05%   |
| 0.3 | 4     | 122.1     | 122.6        | -0.37%   |
| 0.3 | 7     | 120.9     | 121.1        | -0.19%   |
| 0.3 | 11    | 124.0     | 125.0        | -0.76%   |
| 0.2 | 1     | 122.9     | 122.9        | +0.05%   |
| 0.2 | 4     | 122.4     | 123.5        | -0.88%   |
| 0.2 | 7     | 120.5     | 120.7        | -0.23%   |
| 0.2 | 11    | 125.1     | 126.4        | -1.06%   |
| 0.1 | 1     | 123.7     | 123.7        | +0.01%   |
| 0.1 | 4     | 122.7     | 122.6        | +0.04%   |
| 0.1 | 7     | 119.7     | 120.1        | -0.34%   |
| 0.1 | 11    | 127.4     | 128.1        | -0.56%   |

All relative differences are within 1.1%. The refutation criterion (within 3% at all conditions) is met. **Verdict REFUTED is correct.**

## Phase 2: Quality Checks

### G1: Is the compressed block actually swapped for eval? — PASS

For SVD+lstsq and random+lstsq: a `register_forward_hook` on `model.transformer.h[block_idx].mlp` returns `y_approx`, which PyTorch uses as the replacement output. The hook code is correct — it intercepts the MLP output, projects through the rank-k bottleneck, and returns the approximation.

Positive control: topk+lstsq uses `swap_compressed_ffns` (physical module replacement) and shows substantial PPL degradation (131-185 at k/n 0.1-0.2). If the eval were broken, topk would also show near-baseline. The topk degradation confirms the eval pipeline is sensitive.

### G2: >=3 seeds? — PASS

3 seeds (42, 123, 7) used consistently for all conditions.

### G3: Results interpretation — near-baseline PPL at k=77? — FLAG (see G4)

At k=77 (k/n=0.1), SVD+lstsq gives PPL 119.7-127.4 across blocks (baseline 121.5). This is plausible because:

1. **Single-block compression**: only one MLP out of 12 is modified. In the residual architecture `x = x + mlp(ln2(x))`, the MLP output is a perturbation to the residual stream. Even zeroing the entire MLP output would degrade PPL moderately, not catastrophically.

2. **lstsq is overdetermined**: 1024 probe tokens, 77 features per output dim — the fit generalizes.

3. **The topk method uses hidden_fraction = k/3072, not k/768.** At k=77, topk keeps 77/3072 = 2.5% of hidden neurons while SVD/random keep 77/768 = 10% of output dimensions. The topk comparison is architecturally different (hidden pruning vs output projection), which explains why topk is far worse. This is acknowledged in code comments but not in the verdict analysis.

### G4: Counterintuitive — near-baseline PPL at 10% retention — FLAG

Two counterintuitive observations:

**Observation 1: PPL below baseline.** Block 7 with SVD+lstsq consistently gives PPL *below* baseline at all k/n ratios (120.9, 120.5, 119.7 vs baseline 121.5). More compression gives *lower* PPL — this is backwards. The lstsq refit on shakespeare probe data acts as a domain-specific regularizer: the rank reduction removes output dimensions that hurt shakespeare-specific PPL. This makes the absolute PPL numbers partially misleading as a compression quality metric.

**Observation 2: Rank_99=434-588 vs k=77 working.** Cycle 068's Jacobian rank_99 measures the intrinsic dimensionality of the FFN *mapping* (input-to-output). The lstsq refit here is different — it optimizes `(y @ Q) @ B.T ≈ y` for the *output distribution*, not the mapping. If the output distribution is concentrated in fewer dimensions than the mapping's rank, a lower k suffices. Additionally, single-block PPL is a weak signal (see G3 point 1).

**Reconciliation**: The near-baseline PPL is genuine but the experiment doesn't test what it claims. It tests "does basis choice matter for single-block output projection with lstsq refit?" — and the answer is no. It does NOT test whether k=77 is sufficient for *lossless* compression, because the lstsq refit provides in-distribution adaptation that masks information loss.

### G5: Learning quality — N/A

No training involved. lstsq is a closed-form solution.

### G6: Method failure vs genuine finding — PASS

The SVD-vs-random comparison is fair and the refutation is genuine: lstsq dominates basis choice. The finding that random bases match SVD is informative. However, the near-baseline absolute PPL values should not be interpreted as "10% output dims suffice for compression" — the single-block setup and lstsq domain adaptation inflate this impression.

## Critical Audit Answer

**How can k=77 give near-baseline PPL when rank_99 is ~450?**

Three factors:
1. **Single-block modification**: GPT-2 has 12 blocks. Perturbing one MLP's output is a small change to the residual stream. Even random noise added at one block wouldn't destroy PPL.
2. **lstsq domain adaptation**: The refit optimizes a linear map from 77 features to 768 outputs on in-distribution probe data. This can actually *improve* shakespeare PPL (block 7 shows this), meaning the lstsq partially compensates for the projection loss.
3. **Rank_99 measures the mapping, not the output**: The Jacobian's effective rank captures input-output sensitivity dimensions. The output *distribution* may live in a much lower-dimensional manifold, requiring fewer dims for output reconstruction.

The experiment is NOT evidence of lstsq overfitting (the problem is well-conditioned: 1024 >> 77) and IS evaluating on held-out data (different from probe data). The PPL measurement is correct. But the experimental design (single-block, same-domain lstsq refit) makes absolute PPL a weak proxy for compression quality.
