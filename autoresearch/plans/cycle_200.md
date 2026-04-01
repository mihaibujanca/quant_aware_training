# Cycle 200 Implementation Plan: Residual Stream Rank Trajectory Under Compression

## Overview

Compare the input_rank_90 trajectory across all 12 blocks in:
(a) float GPT-2 and (b) compressed GPT-2 (BLOCKS=[1,4,7,11], K=512).

Cycle 198 found: float input ranks 269→207→25→1 by block 3.
Hypothesis: compressing block 1 (K=512, rank-1 output) disrupts this collapse.

**Script**: `experiments/rank_trajectory_cycle200.py`
**Output**: `autoresearch/results/cycle_200_rank_trajectory.json`

---

## Config

```python
BLOCKS_COMPRESSED = [1, 4, 7, 11]
K = 512
N_SEQS = 50
SEQ_LEN = 64
SEED = 42
ENERGY_THRESH = 0.90
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_PATH = "autoresearch/results/cycle_200_rank_trajectory.json"
```

---

## Implementation

### Step 1: Collect residual stream states at each block boundary

Hook on each block's input (before the block runs) to capture the residual stream state.
GPT-2 transformer block receives the residual stream as `inp[0]`.

```python
residual_states = {b: [] for b in range(12)}  # input to each block

def mk_hook(bb):
    def fn(m, inp, out):
        # inp[0]: residual stream input to block bb, shape (1, T, 768)
        residual_states[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
    return fn

hooks = [model.transformer.h[b].register_forward_hook(mk_hook(b)) for b in range(12)]
# Also hook after block 11 to get final residual state
# Use hook on model.transformer.ln_f (final LN) to get post-block-11 state
```

Actually, easier: use `register_forward_pre_hook` to capture block input = residual stream state.

```python
def mk_pre_hook(bb):
    def fn(m, inp):
        residual_states[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
    return fn
```

Or use the block forward hook's `inp[0]` — this is the residual stream entering that block.

### Step 2: Compute effective rank at each block

```python
def effective_rank(X, threshold=0.90):
    """rank_90 of the centered data matrix X: (N, 768)."""
    X_c = X - X.mean(0)
    S = torch.linalg.svdvals(X_c)  # (min(N, 768),)
    energy = (S**2).cumsum(0) / ((S**2).sum() + 1e-12)
    return int((energy < threshold).sum()) + 1
```

### Step 3: Build compressed model

Reuse collect_mlp_io and fit_block from cycle 199 (k_sweep_rank1_cycle199.py).
Collect MLP I/O from the float model first, then fit each block, then replace.

```python
# Load float model, collect calibration I/O
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()
ln_outs, proj_outs = collect_mlp_io(model, calib_ids, BLOCKS_COMPRESSED)

# Fit and replace
for b in BLOCKS_COMPRESSED:
    compressed = fit_block(model.transformer.h[b], ln_outs[b], proj_outs[b], K)
    model.transformer.h[b].mlp = compressed.to(DEVICE)
```

### Step 4: Measure rank trajectories for both models

```python
# Float model: load fresh
model_float = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model_float.eval()
float_states = collect_residual_states(model_float, rank_ids)
float_ranks = [effective_rank(float_states[b]) for b in range(12)]

# Compressed model: as built above
comp_states = collect_residual_states(model_comp, rank_ids)
comp_ranks = [effective_rank(comp_states[b]) for b in range(12)]
```

### Step 5: Also compute Frobenius divergence

```python
# Per-block: ||compressed_residual - float_residual|| / ||float_residual||
frob_div = [
    (comp_states[b] - float_states[b]).norm(dim=1).mean().item() /
    float_states[b].norm(dim=1).mean().item()
    for b in range(12)
]
```

---

## Verdict Logic

```python
# Confirmed if: compressed model's rank at blocks 2-3 is >10× float rank at same block
# (rank collapse disrupted)
float_rank_b3 = float_ranks[3]  # ~1 from cycle 198
comp_rank_b3 = comp_ranks[3]
rank_disruption_ratio = comp_rank_b3 / max(float_rank_b3, 1)

if rank_disruption_ratio >= 10:
    verdict = "CONFIRMED"   # compression disrupts rank collapse
elif rank_disruption_ratio <= 2:
    verdict = "REFUTED"     # rank collapse preserved
else:
    verdict = "INCONCLUSIVE"
```

Also check all 12 blocks: if any compressed block has >10× rank vs float at same position.

---

## Output JSON

```json
{
  "cycle": 200, "item": 210,
  "verdict": "...",
  "float_ranks": [rank_b0, rank_b1, ..., rank_b11],
  "compressed_ranks": [rank_b0, rank_b1, ..., rank_b11],
  "rank_ratios": [comp/float for each block],
  "frob_divergence": [...],
  "max_rank_disruption": ...,
  "config": {...}
}
```

---

## Notes

- Use N_SEQS=50, SEQ_LEN=64 to match cycle 198 (comparability of float ranks)
- Separate rank measurement data from calibration data: use different sequences for each
  Actually: simpler to use the SAME calibration data for rank measurement; changes nothing for analysis
- Float ranks from cycle 198: b0=269, b1=207, b2=25, b3+=1; verify these reproduce
- Pre-hook vs post-hook: hook on block.forward gives inp[0]=residual_in and out[0]=residual_out
  Use inp[0] to measure rank of what enters the block = "input rank" as in cycle 198
