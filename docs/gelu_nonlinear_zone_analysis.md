# GELU Nonlinear Zone: Structural Analysis of GPT-2 FFN Blocks

**Date**: 2026-03-30
**Cycles**: 103, 108, 109, 110
**Status**: Interpretability findings — not optimization results

---

## Summary

GPT-2 FFN blocks operate extensively in the GELU nonlinear zone (|z| < 1), but in a structured way that varies systematically with depth. The key findings are:

1. **Two channel populations** exist in every block: bump-dominated channels (small pre-activation variance, computation lives in the GELU knee) and compression-important channels (large variance, mostly linear operation). These are anticorrelated.

2. **The negative leak is a massive structured bias** that grows with depth. Positive and negative regime outputs partially cancel — the FFN computes a difference between large opposing terms, not a simple sum.

3. **Block 1 is structurally unique** — it has the tightest pre-activations, the most bump-dominated channels, and operates primarily in the nonlinear regime.

---

## 1. Pre-activation distribution across depth (cycle 108)

The pre-activation z = W_in @ x + b has per-channel variance σ² that varies across blocks. The mean σ per block determines how much of the computation falls in the GELU nonlinear zone.

| Block | mean(σ) | bump-dominated frac | excess over Gaussian at T=0.5 | ρ(σ, importance) |
|-------|---------|--------------------|-----------------------------|-----------------|
| 0     | 0.624   | 12.2%              | −4.6% (matches)             | −0.341          |
| 1     | **0.519** | **44.4%**          | **+23.8%** (excess)         | −0.594          |
| 2     | 0.701   | 13.3%              | +28.9% (excess)             | −0.479          |
| 3     | 0.762   | 4.9%               | +21.2% (excess)             | −0.464          |
| 4     | 0.722   | 7.4%               | +13.2%                      | −0.615          |
| 5     | 0.746   | 6.2%               | +9.6%                       | −0.645          |
| 6     | 0.727   | 4.5%               | +6.7%                       | −0.562          |
| 7     | 0.739   | 0.8%               | +3.5%                       | −0.485          |
| 8     | 0.757   | 0.2%               | +1.6%                       | −0.361          |
| 9     | 0.780   | 0.1%               | +1.6%                       | −0.207          |
| 10    | 0.797   | 0.1%               | +4.4%                       | −0.007          |
| 11    | 0.796   | 0.0%               | +5.3%                       | +0.046          |

"Bump-dominated" = channels with σ < 0.5 AND |μ| < 0.5 (computation almost entirely in the GELU knee).

**Key observations:**

- **Block 1 is an outlier**: σ = 0.519 (tightest), 44% bump-dominated channels, 24% excess above Gaussian prediction. The network places far more of block 1's computation in the nonlinear zone than variance alone would predict. This is active concentration.

- **Blocks 2-3 also show excess** (21-29%) — real concentration in the bump, declining with depth.

- **Blocks 7+ match Gaussian** (<5% excess). Late blocks operate "normally" — their pre-activation distributions are explained by variance, no active concentration.

- **ρ(σ, importance) is negative everywhere** (−0.007 to −0.645). Channels with tight pre-activations (bump-dominated) are systematically LOW importance for compression. The nonlinear computation and the compression-important computation happen in **different channels**.

This last point means that compression (which keeps high-importance channels) and nonlinear structure (which lives in low-σ channels) are largely orthogonal. Compression discards the bump channels first.

---

## 2. Negative leak decomposition (cycle 109)

For z < 0, GELU(z) produces a small negative output (the "leak") that ReLU would zero. We decompose this into a constant bias b_leak (mean over tokens) and a token-dependent signal s_leak.

| Block | b_ratio (||b_leak||/||output||) | s_ratio (||s_leak||/||output||) | b_energy | norm_pos | norm_neg |
|-------|------|---------|----------|----------|----------|
| 0     | 0.62 | 0.20    | 91.3%    | 29.0     | 16.8     |
| 1     | **1.08** | **0.48** | 87.1% | 15.7     | 13.2     |
| 2     | 1.86 | 0.41    | 95.8%    | 30.1     | 27.9     |
| 3     | **2.42** | 0.45 | 97.0%   | 34.3     | **34.8** |
| 4     | 2.37 | 0.44    | 97.1%    | 33.9     | 35.0     |
| 5     | 1.96 | 0.40    | 96.3%    | 35.3     | 34.3     |
| 6     | 1.51 | 0.37    | 94.7%    | 34.0     | 31.1     |
| 7     | 1.25 | 0.34    | 93.2%    | 36.0     | 30.8     |
| 8     | 0.98 | 0.32    | 90.2%    | 36.9     | 30.1     |
| 9     | 0.82 | 0.29    | 88.1%    | 43.6     | 32.3     |
| 10    | 0.79 | 0.18    | 94.4%    | 82.3     | 55.1     |
| 11    | 1.54 | 0.18    | 98.5%    | 164.7    | 124.3    |

b_ratio = ||b_leak|| / ||mean block output||. Values > 1.0 mean the leak bias is LARGER than the total output.

**Key observations:**

- **The leak is overwhelmingly a constant bias** (b_energy 87-98% across all blocks). Token-dependent signal accounts for only 1.5-13% of negative-regime energy.

- **The bias exceeds the output norm in blocks 1-7** (b_ratio 1.08 to 2.42). This means the FFN output is a relatively small difference between large positive-regime and negative-regime (leak) contributions. The positive output and the negative leak partially cancel.

- **Blocks 3-4 are at peak cancellation**: norm_pos ≈ norm_neg (34.3 vs 34.8), with the output norm only 14.2. The FFN is computing output ≈ positive_contribution − leak ≈ 34 − 35 ≈ −1. The output is a small residual between two large opposing terms.

- **Block 11 is extreme**: norm_pos = 164.7, norm_neg = 124.3, output = 80.5. Absolute norms are much larger, but the ratio is less extreme than blocks 3-4.

- **The depth profile is non-monotonic**: b_ratio rises from 0.62 (block 0) to 2.42 (block 3), then falls back to 0.79 (block 9), then rises again to 1.54 (block 11). Peak cancellation is in middle-early blocks (3-5).

**Geometric interpretation**: GPT-2 FFN blocks compute their output as a near-cancellation of two large terms: the contribution from positively-activated channels and a structured bias from negatively-activated channels. The leak is not a small perturbation on top of a ReLU-like computation — it is a major structural component that the positive side must compensate for. This explains why GELU→ReLU swap is catastrophic (+8320% PPL): zeroing the negative leak removes a term that is 1-2.4× the output magnitude.

---

## 3. Two channel populations (cycle 110)

Channels partition into bump-dominated (high fraction of activations in |z| < 1.0) and compression-important (high topk importance score). The correlation between these properties varies with depth.

| Block | ρ(bump, importance) | n bump-critical | agg. contribution | median variance |
|-------|--------------------:|----------------:|------------------:|----------------:|
| 0     | +0.027              | 355             | 22.1%             | 0.0179          |
| 1     | +0.243              | 299             | 10.7%             | 0.0136          |
| 2     | −0.030              | 295             | 13.0%             | 0.0204          |
| 3     | −0.360              | 170             | 8.1%              | 0.0240          |
| 4     | −0.260              | 202             | 9.6%              | 0.0287          |
| 5     | −0.267              | 170             | 8.1%              | 0.0364          |
| 6     | −0.215              | 158             | 7.1%              | 0.0372          |
| 7     | −0.367              | 109             | 4.7%              | 0.0432          |
| 8     | −0.490              | 49              | 2.7%              | 0.0444          |
| 9     | −0.599              | 38              | 1.9%              | 0.0573          |
| 10    | −0.541              | 56              | 2.7%              | 0.0555          |
| 11    | −0.519              | 58              | 2.7%              | 0.0813          |

ρ(bump, importance) = Spearman correlation between per-channel bump fraction and weight-only importance score (||W_out[:,j]||). "Bump-critical" = top quartile in both bump fraction and importance.

**Key observations:**

- **The correlation inverts with depth**: blocks 0-1 are mildly positive (bump channels are somewhat important), blocks 3-11 are increasingly negative (bump channels are explicitly UNimportant for compression).

- **Bump-critical channels decline from 355 to 38-58** between blocks 0 and 9. In early blocks, there is significant overlap between bump-dominated and important channels. By late blocks, the two populations are almost entirely separate.

- **Per-token variance of bump-critical channels increases with depth** (0.014 → 0.081). In early blocks, bump-critical channels compute near-constants. In late blocks (where few remain), the surviving bump-critical channels are more variable — they're doing something token-dependent.

- **Aggregate contribution of bump-critical channels drops** from 22% (block 0) to 2-3% (blocks 8-11). Compression would naturally discard these channels in later blocks.

---

## 4. Synthesis: the GELU operating geometry of GPT-2

### The two-regime picture

Each FFN block has two functional populations of hidden channels:

**Population A — "Linear channels"** (σ > 0.7, most activations in z > 0 or z < −2):
- GELU operates nearly linearly (GELU(z) ≈ z for z >> 0, GELU(z) ≈ 0 for z << 0)
- High weight norms → high compression importance
- These are the channels that topk selects and lstsq refits
- They carry the "backbone" computation

**Population B — "Bump channels"** (σ < 0.5, activations concentrated in |z| < 1):
- GELU operates in its nonlinear knee, providing smooth gating
- Low weight norms → low compression importance
- These channels compute subtle corrections via GELU's soft thresholding
- Their aggregate output is mostly a structured bias (87-98% constant)

### Depth-dependent structure

- **Block 0**: Moderate separation. Both populations present, somewhat intermixed.
- **Block 1**: Maximum nonlinear operation. 44% bump-dominated channels, tightest pre-activations (σ=0.52), 24% excess concentration. The network places block 1's computation maximally in the GELU knee.
- **Blocks 2-5**: Peak cancellation regime. Leak bias exceeds output (b_ratio 1.5-2.4). The FFN output is a small residual of two large opposing terms. Population separation begins.
- **Blocks 6-9**: Populations fully separate. Bump channels are explicitly anti-correlated with importance (ρ down to −0.6). Compression discards bump channels entirely.
- **Blocks 10-11**: Large absolute norms but moderate cancellation. The surviving bump-critical channels (58) are highly variable — the few remaining do token-dependent work.

### Implications for compression

1. **Current compression (topk+lstsq) naturally discards bump channels** because they have low weight norms. This explains why compression works at all — it keeps the linear backbone and discards the nonlinear corrections.

2. **The discarded bump channels contribute a structured bias** that lstsq partially absorbs. The lstsq refit on the remaining channels can capture some of the leaked bias as a shift in the output projection.

3. **The 500 Adam steps likely recover the bias** that lstsq cannot fully absorb. The polish primarily corrects for the missing negative-leak contribution.

4. **Block 1's sensitivity to compression** (it was the one outlier in early experiments) may be because 44% of its channels are bump-dominated — compression removes nearly half of its functional channels, not just redundant ones.

### Implications for the geometric thesis

The geometry of GELU FFNs is not piecewise-linear (as MASO theory predicts for ReLU). Instead, it is a **smooth two-regime geometry**: a high-variance linear backbone modulated by a low-variance nonlinear correction layer. The "correction" is not small — it can exceed the output in magnitude — but it is structured (mostly constant bias) and orthogonal to the compression-important directions.

This is a different geometry than piecewise-linear, but it IS geometry. The structure is: (a) a depth-dependent operating regime that shifts channels between linear and nonlinear modes, (b) a cancellation structure where positive and negative contributions nearly balance, producing small outputs from large intermediate terms, and (c) a systematic separation of linear/important and nonlinear/corrective channel populations that deepens with block index.

---

## Data sources

| Cycle | Item | Type | Key finding |
|-------|------|------|------------|
| 103   | 108  | GELU-ReLU anatomy | 94.6% activations in bump zone (T=2.5); bump carries 31.5% disagreement energy |
| 108   | 119  | Pre-activation distribution | Block 1 outlier (σ=0.52, 44% bump-dom); early blocks show 21-29% excess; ρ(σ, importance) negative |
| 109   | 120  | Negative leak decomposition | Leak is 87-98% constant bias; b_ratio peaks at 2.42 (block 3); massive cancellation |
| 110   | 121  | Bump-compression relevance | ρ(bump, importance) inverts with depth (+0.24 → −0.60); bump-critical channels decline 355→38 |
