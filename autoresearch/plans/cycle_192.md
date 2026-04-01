# Cycle 192 Implementation Plan: Exact Dead-Channel Pruning Baseline

## Overview

Audit GPT-2 FFN channels for near-dead behavior (max GELU activation < epsilon over all
calibration tokens). Prune near-dead channels exactly from weight matrices. Measure
compression ratio and PPL at multiple thresholds (0.001, 0.01, 0.1, 1.0).

**Script**: `experiments/dead_channel_pruning_cycle192.py`
**Output**: `autoresearch/results/cycle_192_dead_channel_pruning.json`

---

## Config

```python
DEVICE = "mps" or "cpu"
N_SEQS_CALIB = 500      # more sequences for reliable dead-channel statistics
BATCH_SIZE = 4
SEQ_LEN = 128
SEED = 1337
N_BLOCKS = 12
N_CHANNELS = 3072       # FFN hidden size
EPSILONS = [0.001, 0.01, 0.1, 1.0]
```

---

## Step 1: Collect max per-channel GELU activation

Hook on `model.transformer.h[b].mlp.act` to capture inputs z (shape B*S, 3072).
For each channel c: track max_{tokens} GELU(z_c) over all calibration batches.

```python
max_gelu = {b: torch.zeros(N_CHANNELS) for b in range(N_BLOCKS)}

def make_hook(idx):
    def hook_fn(module, inp, out):
        gelu_out = out.reshape(-1, N_CHANNELS).abs().float()  # (B*S, 3072)
        max_gelu[idx] = torch.max(max_gelu[idx], gelu_out.max(dim=0).values.cpu())
    return hook_fn
```

---

## Step 2: Prune at each threshold

For each epsilon in EPSILONS:
1. Deepcopy float_model
2. For each block b:
   - dead_mask[b] = (max_gelu[b] < epsilon)  # (3072,) bool
   - Zero out rows of c_fc.weight and c_fc.bias where dead_mask is True
     (c_fc maps 768→3072; weight shape (768, 3072) for Conv1D)
     Actually: GPT-2 Conv1D weight shape is (in_features, out_features) = (768, 3072).
     Zero out columns of c_fc.weight[:, dead_mask] and c_fc.bias[dead_mask]
   - Zero out rows of c_proj.weight where dead_mask is True
     c_proj weight shape (3072, 768). Zero out rows: c_proj.weight[dead_mask, :] = 0
   - Zero out c_fc.bias[dead_mask]
3. Measure PPL

---

## Step 3: Compute statistics

For each epsilon:
- n_dead_per_block[b] = dead_mask[b].sum()
- pct_dead = n_dead_per_block.mean() / N_CHANNELS × 100
- ppl at this threshold

---

## Step 4: Always-active analysis (bonus)

For each channel c: check if z_c > 0 for ALL calibration tokens.
These channels operate in the linear regime of GELU (GELU ≈ identity for z >> 0).

---

## Output JSON

```json
{
  "cycle": 192, "item": 202,
  "verdict": "...",
  "max_gelu_stats": {...},
  "thresholds": [
    {
      "epsilon": 0.001,
      "n_dead_per_block": [...],
      "mean_pct_dead": ...,
      "ppl": ...
    },
    ...
  ],
  "always_active": {"n_per_block": [...], "mean_pct": ...},
  "config": {...}
}
```

---

## Verdict

```python
n_dead_001 = mean channels dead at epsilon=0.01
if n_dead_001 / N_CHANNELS > 0.05 and ppl_001 < 25:
    verdict = "CONFIRMED"
elif n_dead_001 / N_CHANNELS < 0.01:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
```

Note: PPL < 25 is relatively permissive (+10% from baseline ~22.7 at SEQ_LEN=128 ~35.9).
Use the same baseline as cycle 189/191 (SEQ_LEN=128 gives ~35.9).
So the threshold should be ppl < 38 (+6% from 35.9), meaning near-zero accuracy loss.
