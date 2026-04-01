# Cycle 195 Implementation Plan: Jacobian Effective Rank Census

## Overview

Compute per-token FFN block Jacobians using torch.func.jacrev for GPT-2. Measure
effective rank distribution across tokens per block. This is the core R3/MASO measurement.

**Script**: `experiments/jacobian_rank_census_cycle195.py`
**Output**: `autoresearch/results/cycle_195_jacobian_rank.json`

---

## Config

Small config for compute budget:
```python
N_SEQS = 30          # 30 sequences
SEQ_LEN = 32         # 32 tokens each → 960 tokens per block
BATCH_SIZE = 1       # jacrev requires per-sample
SEED = 42
DEVICE = "cpu"       # MPS may not support jacrev; use CPU
N_BLOCKS = 12
D_MODEL = 768
ENERGY_THRESHOLD = 0.90
```

---

## Implementation

### Step 1: Extract per-block FFN function

For each block b, define the function to differentiate:
```python
def ffn_fn(x_single: torch.Tensor) -> torch.Tensor:
    # x_single: (768,) — single token residual
    # Returns: (768,) — block output (residual stream, after residual connection)
    # Use the existing block but only the FFN sublayer
    with torch.no_grad():  # don't need grads for the weights
        x_2d = x_single.unsqueeze(0)  # (1, 768)
        # FFN: LN2 → mlp → + residual
        residual = x_2d
        ln_out = block.ln_2(x_2d)
        mlp_out = block.mlp(ln_out)
        return (residual + mlp_out).squeeze(0)  # (768,)
```

Wait — jacrev will differentiate w.r.t. x_single, so gradients through block.ln_2 and
block.mlp will be computed. We DO need enable_grad for the input while keeping weights fixed.

```python
def make_ffn_fn(block):
    def fn(x):  # x: (768,)
        x2 = x.unsqueeze(0)
        return (x2 + block.mlp(block.ln_2(x2))).squeeze(0)
    return fn
```

### Step 2: Per-sample Jacobian

```python
from torch.func import jacrev, vmap

# For each block:
fn = make_ffn_fn(float_model.transformer.h[b])

# vmap over tokens:
J_batch = vmap(jacrev(fn))(token_residuals)  # (N_tokens, 768, 768)
```

Note: vmap + jacrev on CPU for a 768×768 Jacobian over 960 tokens is expensive.
Alternative: use functional jacobian via finite differences if jacrev is too slow.

### Step 3: Effective rank per token

```python
for i in range(N_tokens):
    J = J_batch[i]  # (768, 768)
    S = torch.linalg.svdvals(J)  # (768,)
    energy = (S**2).cumsum(0) / (S**2).sum()
    rank_90 = int((energy < 0.90).sum()) + 1
```

### Step 4: Stacked Jacobian rank

Flatten all token Jacobians into a (N_tokens * 768, 768) matrix.
SVD this stacked matrix to get the rank of the span of all Jacobians.
```python
J_stacked = J_batch.view(-1, 768)  # (N_tokens * 768, 768)
S_stacked = torch.linalg.svdvals(J_stacked)
energy_stacked = (S_stacked**2).cumsum(0) / (S_stacked**2).sum()
stacked_rank = int((energy_stacked < 0.90).sum()) + 1
```

---

## Output JSON

```json
{
  "cycle": 195, "item": 206,
  "verdict": "...",
  "per_block": [
    {
      "block": 0,
      "mean_rank_90": ...,
      "std_rank_90": ...,
      "min_rank_90": ...,
      "max_rank_90": ...,
      "stacked_jacobian_rank_90": ...
    },
    ...
  ],
  "config": {...}
}
```

---

## Verdict

```python
# Mean rank across all blocks
all_mean_ranks = [r["mean_rank_90"] for r in per_block]
overall_mean_rank = mean(all_mean_ranks)
n_low_rank = sum(1 for r in all_mean_ranks if r < 100)

if n_low_rank >= 6:    verdict = "CONFIRMED"  # low-rank: <100/768
elif all_mean_ranks[-1] > 300:  verdict = "REFUTED"  # full-rank
else:                  verdict = "INCONCLUSIVE"
```

---

## Compute Budget

30 seqs × 32 tokens = 960 tokens × 12 blocks × (768×768 Jacobian) on CPU.
Each Jacobian via jacrev: ~768 forward passes of 1 token. Very slow.

Alternative: Use manual Jacobian via perturbing each input dimension:
```python
J = torch.zeros(768, 768)
eps = 1e-4
for d in range(768):
    e_d = torch.zeros(768); e_d[d] = eps
    J[:, d] = (fn(x + e_d) - fn(x - e_d)) / (2 * eps)
```
This is 768 × 2 = 1536 forward passes per token. For 960 tokens × 12 blocks = 17M forward passes.
Too slow.

Better approach: Use `torch.func.jacfwd` (forward-mode) which is much faster than jacrev for
tall Jacobians (more outputs than inputs, but here 768×768 is square). Or use jvp batched.

Actually, `jacfwd` with vmap should compute the full Jacobian in ONE batched pass using
JVP with 768 tangent vectors simultaneously. This should be much faster.

```python
from torch.func import jacfwd, vmap
J_all = vmap(jacfwd(fn))(tokens)  # (N, 768, 768)
```

Use jacfwd (forward-mode AD) for efficiency on CPU.

Estimate: jacfwd for 768×768 Jacobian for 960 tokens × 12 blocks ≈ 960 × 12 × T_fwd
where T_fwd is the time for one forward pass through a single block (~ms).
~10-30 seconds total on CPU. Should be within budget.
