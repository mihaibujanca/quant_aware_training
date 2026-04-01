# Cycle 191 Implementation Plan: Mean Residual Error Correction

## Overview

Build full-swap ReLU+bias GPT-2 model (all 12 blocks), calibrate per-layer mean error
correction vectors (12×768=9216 params), apply as residual stream hooks, and measure PPL.

**Script**: `experiments/mean_error_correction_cycle191.py`
**Output**: `autoresearch/results/cycle_191_mean_error_correction.json`

---

## Config

Match cycle 189 for comparability:
```python
DEVICE = "mps" or "cpu"
N_SEQS_CALIB = 200;  SEED_CALIB = 1337
N_SEQS_ANALYSIS = 100;  SEED_ANALYSIS = 42
BATCH_SIZE = 4;  SEQ_LEN = 128
N_BLOCKS = 12;  N_CHANNELS = 3072
```

---

## Step 1: Build swapped model

Same as cycle 189 `build_relu_bias_model`:
1. Compute bias_vectors[b] (768,) = E[GELU(z)-ReLU(z)] @ c_proj.weight from calib data
2. Deepcopy float_model; for each block b: replace mlp.act=ReLU(), add bias_vectors[b] to c_proj.bias

---

## Step 2: Calibrate mean correction

Run BOTH float_model and swapped_model on calib_seqs with ResidualCollector hooks.
For each layer l:
```python
e_l = res_float[l] - res_swapped[l]  # (N_tokens, 768)
mean_correction[l] = e_l.mean(dim=0)  # (768,) — first moment of error
```

---

## Step 3: Apply correction as hook

Register forward hook on `swapped.transformer.h[l]` that adds mean_correction[l] to output:
```python
def make_correction_hook(correction_vec):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        corrected = h + correction_vec.to(h.device)
        return (corrected,) + out[1:] if isinstance(out, tuple) else corrected
    return hook

correction_hooks = []
for l in range(N_BLOCKS):
    h = swapped.transformer.h[l].register_forward_hook(
        make_correction_hook(mean_correction[l]))
    correction_hooks.append(h)
```

---

## Step 4: Evaluate

1. `ppl_baseline` — float model on analysis_seqs (SEQ_LEN=128)
2. `ppl_uncorrected` — swapped model WITHOUT correction hooks on analysis_seqs
3. `ppl_corrected` — swapped model WITH correction hooks on analysis_seqs

---

## Step 5: Diagnostics

Also report:
- `mean_correction_norms[l]` = ||mean_correction[l]||₂ for each layer
- `fraction_of_frobenius[l]` = ||mean_correction[l]||₂ / (||e_l||_F / sqrt(N_tokens))
  → what fraction of per-token error is accounted for by the mean

---

## Verdict

```python
delta = ppl_uncorrected - ppl_corrected
if delta > 6.0:   verdict = "CONFIRMED"
elif delta < 1.0: verdict = "REFUTED"
else:             verdict = "INCONCLUSIVE"
```

---

## Output JSON

```json
{
  "cycle": 191, "item": 201,
  "verdict": "...",
  "baseline_ppl": ...,
  "uncorrected_ppl": ...,
  "corrected_ppl": ...,
  "ppl_improvement": ...,
  "correction_norms": [...],
  "fraction_of_frobenius": [...],
  "config": {...}
}
```
