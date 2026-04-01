# Cycle 197 Implementation Plan: Block-Pair Joint Compression

## Overview

Test whether sequential progressive fitting (fitting each block on the previous block's
COMPRESSED output rather than float output) reduces cascade PPL compared to all-float fitting.

**Script**: `experiments/joint_compression_cycle197.py`
**Output**: `autoresearch/results/cycle_197_joint_compression.json`

---

## Config

```python
BLOCKS = [1, 4, 7, 11]   # same as working recipe
K = 512                   # channels (3072/6, 6x compression)
N_SEQS_CALIB = 200
SEQ_LEN = 128
SEED = 42
DEVICE = "mps"
```

---

## Two Fitting Strategies

### Strategy A: All-float fitting (baseline — same as cycle 196)
For each block b in BLOCKS:
- Collect MLP I/O from FLOAT model via hooks
- Fit lstsq on float I/O

### Strategy B: Sequential progressive fitting
Process blocks in order:
1. Block 1: fit on float model I/O (same as A)
2. Replace block 1 in a working copy of the model
3. Block 4: collect I/O from the PARTIALLY compressed model (block 1 compressed, 2-3 float)
4. Replace block 4
5. Block 7: collect from model with blocks 1,4 compressed
6. Block 11: collect from model with blocks 1,4,7 compressed

This way, each block's student is trained on the actual inputs it will receive.

---

## Implementation

```python
# Strategy A: collect all at once from float model
def fit_all_float(model, calib_ids, BLOCKS, K):
    mlp_in, mlp_out = collect_mlp_io(model, calib_ids, BLOCKS)
    fits = {}
    for b in BLOCKS:
        fits[b] = fit_compressed_mlp(model.transformer.h[b], mlp_in[b], mlp_out[b], K)
    return fits

# Strategy B: sequential progressive
def fit_sequential(model, calib_ids, BLOCKS, K):
    fits = {}
    working_model = copy.deepcopy(model)
    for b in BLOCKS:
        # Collect from the CURRENT working model (which has previous blocks compressed)
        mlp_in, mlp_out = collect_mlp_io(working_model, calib_ids, [b])
        fits[b] = fit_compressed_mlp(model.transformer.h[b], mlp_in[b], mlp_out[b], K)
        # Replace this block in working model
        apply_compression(working_model, b, fits[b], K)
    return fits
```

---

## Fitting Function

Same as cycle 196 (channel selection by std × output_norms × mean_act, lstsq with bias):

```python
def fit_compressed_mlp(block, ln_out, proj_out, K):
    # Channel selection
    fc_full = ln_out @ block.mlp.c_fc.weight + block.mlp.c_fc.bias
    act = F.gelu(fc_full)
    scores = act.std(0) * block.mlp.c_proj.weight.norm(dim=1) * (act.mean(0).abs() + 1e-6)
    idx = scores.topk(K).indices

    # Subsetted weights
    fc_w = block.mlp.c_fc.weight[:, idx]  # (768, K)
    fc_b = block.mlp.c_fc.bias[idx]       # (K,)

    # lstsq with bias
    act_sub = act[:, idx]  # (N, K)
    design = torch.cat([act_sub, ones], dim=1)  # (N, K+1)
    sol, *_ = torch.linalg.lstsq(design.cpu(), proj_out.cpu(), rcond=None)
    W_out = sol[:K].T  # (768, K)
    b_out = sol[K]     # (768,)

    return fc_w, fc_b, idx, W_out, b_out
```

---

## Output JSON

```json
{
  "cycle": 197, "item": 208,
  "verdict": "...",
  "float_ppl": ...,
  "strategy_a_ppl": ...,  # all-float fitting
  "strategy_b_ppl": ...,  # sequential progressive
  "improvement_b_over_a": ...,
  "config": {...}
}
```

---

## Verdict

```python
improvement = strategy_a_ppl - strategy_b_ppl
if improvement > 2:   verdict = "CONFIRMED"  # sequential fitting saves >2 PPL
elif improvement < 0: verdict = "REFUTED"   # sequential is worse
else:                 verdict = "INCONCLUSIVE"
```

---

## Compute Budget

- Strategy A: ~5 min (collect once, fit 4 blocks)
- Strategy B: ~20 min (4 sequential collection passes + 4 fits)
- PPL evaluation: ~5 min each
- Total: ~35 min (borderline; reduce N_SEQS if needed)
