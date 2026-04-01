# Cycle 198 Implementation Plan: Block 0 Criticality Analysis

## Overview

Compare block 0 vs blocks 1-11 on data-active geometry metrics to identify
what makes block 0 uniquely critical (1923 PPL when dropped vs <41 for others).

**Script**: `experiments/block0_criticality_cycle198.py`
**Output**: `autoresearch/results/cycle_198_block0_criticality.json`

---

## Config

```python
N_SEQS = 50
SEQ_LEN = 64
SEED = 42
DEVICE = "mps"
N_BLOCKS = 12
```

---

## Metrics

### 1. Hidden activation rank (rank_90 of covariance)
For each block b, collect hidden pre-activations z (BEFORE GELU) from c_fc:
- hook on block.mlp.c_fc to get z: (N_tokens, 3072)
- center and compute covariance C = z.T @ z / N
- SVD of C → singular values → rank at 90% energy
Low rank = concentrated activation pattern.

### 2. Channel utilization
Fraction of channels with mean |GELU(z_c)| > threshold (e.g., 0.1, 1.0):
- high utilization = more channels actively contribute

### 3. Block output magnitude
||residual change|| = ||block(x) - x||_2 per token:
- block 0 likely makes LARGER changes (transforming raw embeddings)

### 4. c_fc weight singular value concentration
SVD of c_fc.weight (768×3072) → Gini coefficient of singular values:
- high concentration = few dominant directions

### 5. Weight-activation alignment
How well do the top singular vectors of c_fc.weight align with top PCA
directions of actual input activations? If aligned, block is "matched" to input.

---

## Implementation

```python
# Hook to collect c_fc outputs (pre-GELU) and residual changes
for b in range(12):
    hook on block.mlp.c_fc: collect z (pre-GELU) = (N_tokens, 3072)
    hook on block: collect (input, output) pairs → compute ||output - input||

# After collection:
for b in range(12):
    # Rank of z covariance
    z = preact_samples[b]  # (N_tokens, 3072)
    z_centered = z - z.mean(0)
    U, S, V = torch.linalg.svd(z_centered, full_matrices=False)  # S: (3072,)
    # Or: use svdvals of z directly
    S = torch.linalg.svdvals(z_centered)  # (min(N,3072),)
    energy = (S**2).cumsum(0) / (S**2).sum()
    rank_90 = (energy < 0.90).sum() + 1

    # Channel utilization
    act = F.gelu(z)
    util_01 = (act.abs() > 0.1).float().mean()  # fraction of channel-tokens > 0.1
    util_10 = (act.abs() > 1.0).float().mean()

    # Output magnitude
    delta_norms = delta_per_token[b].norm(dim=1)  # (N_tokens,)
    mean_delta = delta_norms.mean()

    # c_fc weight singular values
    W = model.transformer.h[b].mlp.c_fc.weight.float()  # (768, 3072)
    S_w = torch.linalg.svdvals(W)  # (768,)
    gini = compute_gini(S_w)
```

---

## Output JSON

```json
{
  "cycle": 198, "item": 205,
  "verdict": "...",
  "per_block": [
    {
      "block": 0,
      "preact_rank_90": ...,
      "channel_util_01": ...,  # fraction with |GELU(z)| > 0.1
      "channel_util_10": ...,  # fraction with |GELU(z)| > 1.0
      "mean_delta_norm": ...,  # mean ||block(x) - x||
      "weight_sv_gini": ...,   # Gini coefficient of c_fc singular values
      "weight_sv_top1_frac": ...  # top-1 SV / sum(SV)
    },
    ...
  ],
  "config": {...}
}
```

---

## Verdict

```python
# Block 0 is "confirmed unique" if ≥3 metrics show it as an outlier vs blocks 1-11
# E.g., preact_rank_90 is 2x smaller, OR delta_norm is 2x larger
# Compare block 0 to median of blocks 1-11 on each metric

block0 = per_block[0]
others = per_block[1:]
outlier_count = 0
if block0["preact_rank_90"] < 0.5 * median([r["preact_rank_90"] for r in others]):
    outlier_count += 1
if block0["mean_delta_norm"] > 2.0 * median([r["mean_delta_norm"] for r in others]):
    outlier_count += 1
if block0["channel_util_01"] < 0.5 * median([r["channel_util_01"] for r in others]):
    outlier_count += 1

if outlier_count >= 2: verdict = "CONFIRMED"  # block 0 has unique geometry
else:                  verdict = "REFUTED"    # similar geometry to other blocks
```
