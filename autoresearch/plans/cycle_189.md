# Cycle 189 Implementation Plan: Cascade Error Direction Analysis

## Overview

Implement experiment to measure directional alignment of errors across layers when GELU is progressively replaced with ReLU+bias. Collect residual streams from float and ReLU+bias models, compute per-layer error metrics (cosine similarity, magnitude, effective rank), and analyze whether errors compound coherently.

**Output**: `autoresearch/results/cycle_189_cascade_error_direction.json`
**Experiment script**: `experiments/cascade_error_direction_cycle189.py`

---

## Configuration

```python
# Hard-coded at top of experiment script
CONFIG = {
    'N_SEQS_CALIB': 200,          # Calibration seqs for mu_neg estimation
    'N_SEQS_ANALYSIS': 100,       # Analysis seqs for residual collection (12,800 tokens)
    'BATCH_SIZE': 4,
    'SEQ_LEN': 128,
    'SEED': 42,
    'DEVICE': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'DATA_PATH': 'data/openwebtext_sample.txt',
    'CALIB_SEED': 1337,
    'ENERGY_THRESHOLD': 0.90,     # For effective rank computation
}
```

---

## Step 1: Load Data and Model

1. Load GPT-2 model (pretrained, float32)
2. Load tokenizer
3. Load calibration data (200 seqs, random seed 1337)
4. Load analysis data (100 seqs, random seed 42, different split from calib)
5. Move model to device (MPS/CPU)

---

## Step 2: Estimate mu_neg (Per-Channel Bias)

For each transformer block l ∈ [0, 11]:

1. Run calibration data through float model with forward hooks on `model.transformer.h[l].mlp.act` (GELU input/output)
2. Collect activations **after** GELU (pre-ReLU distribution)
3. For each channel c:
   - Compute per-channel mean of GELU outputs: `mean_gelu[c]`
   - Compute per-channel mean of ReLU(GELU input): `mean_relu[c]`
   - **mu_neg[c]** = mean_relu[c] - mean_gelu[c] (bias to add after ReLU to match GELU mean)

**Alternative simpler approach** (if above is unstable):
- mu_neg[c] = mean(max(0, x[c])) - mean(GELU(x[c])) where x is GELU input
- This directly captures how ReLU differs from GELU in expectation

Store mu_neg as dict: `{block_idx: tensor(shape=[3072])}` (3072 = FFN hidden size)

---

## Step 3: Build Progressive ReLU+bias Models

For each k ∈ [0, 11]:

1. **Deepcopy** the float model
2. For blocks b in [0, k]:
   - Replace `model.transformer.h[b].mlp.act` with `nn.ReLU()`
   - Add mu_neg[b] to `model.transformer.h[b].mlp.c_proj.bias` **in-place**:
     ```python
     model.transformer.h[b].mlp.c_proj.bias.data += mu_neg[b]
     ```
3. Move modified model to device
4. Set to eval mode

Rationale: Replacing GELU with ReLU changes the activation function. mu_neg is applied at the MLP output (after projection) to shift the residual stream by the expected difference between ReLU and GELU.

---

## Step 4: Residual Collection via Forward Hooks

Create a **hook manager** that captures residuals from both float and ReLU+bias models simultaneously:

```python
class ResidualCollector:
    def __init__(self, model, layers_to_hook=[0..11]):
        self.residuals = {l: [] for l in layers_to_hook}
        self.hooks = []

        for l in layers_to_hook:
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output is the residual stream (full block output)
                    self.residuals[layer_idx].append(output.detach().cpu())
                return hook

            h = model.transformer.h[l].register_forward_hook(make_hook(l))
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def get_residuals(self):
        # Concatenate all collected residuals along token dimension
        return {l: torch.cat(self.residuals[l], dim=0) for l in self.residuals}

    def reset(self):
        self.residuals = {l: [] for l in self.residuals}
```

---

## Step 5: Analysis Loop

For each k ∈ [0, 11]:

1. **Forward pass on float model**:
   - Attach ResidualCollector to float model
   - Process all 100 analysis seqs in batches (BATCH_SIZE=4)
   - Collect residuals: `residuals_float = {l: tensor(N_tokens, 768)}`
   - Remove hooks, reset collector

2. **Forward pass on progressive ReLU+bias model** (blocks 0..k swapped):
   - Attach ResidualCollector to ReLU+bias model
   - Process **same** 100 analysis seqs in **same order**
   - Collect residuals: `residuals_swapped = {l: tensor(N_tokens, 768)}`
   - Remove hooks

3. **Compute error metrics**:
   - `e_l = residuals_float[l] - residuals_swapped[l]` for each layer l
   - **Error magnitude**: `mag_l = ||e_l||_F = sqrt(sum(e_l^2))`
   - **Cosine similarity**:
     ```python
     def mean_cosine_similarity(e_l, e_prev):
         # Average cos(e_l[t], e_prev[t]) over tokens t
         norms_l = torch.norm(e_l, dim=1, keepdim=True) + 1e-8
         norms_prev = torch.norm(e_prev, dim=1, keepdim=True) + 1e-8
         cos_sim = (e_l * e_prev).sum(dim=1) / (norms_l.squeeze() * norms_prev.squeeze())
         return cos_sim.mean().item()
     ```
   - **Effective rank** (SVD at 90% energy):
     ```python
     def effective_rank(e_l, threshold=0.90):
         U, S, Vt = torch.svd(e_l)
         energy = (S ** 2).cumsum(dim=0) / (S ** 2).sum()
         rank = (energy < threshold).sum().item() + 1
         return min(rank, e_l.shape[1])  # cap at 768
     ```

4. **Record results**:
   ```python
   results[k] = {
       'k': k,
       'ppl': compute_ppl(swapped_model, analysis_data),  # Run full forward pass
       'layers': [
           {
               'layer': l,
               'error_magnitude': mag_l,
               'cos_prev_layer': cos_sim_l_prev if l > 0 else None,
               'effective_rank': eff_rank_l,
           }
           for l in range(12)
       ]
   }
   ```

5. **Cleanup**: Delete swapped model, clear residuals, move to next k

---

## Step 6: Aggregate Statistics

```python
summary = {
    'mean_cos_aligned': mean([cos for k in results for layer in results[k]['layers']
                               if layer['cos_prev_layer'] is not None and layer['cos_prev_layer'] > 0.2]),
    'mean_cos_all': mean([layer['cos_prev_layer'] for k in results for layer in results[k]['layers']
                          if layer['cos_prev_layer'] is not None]),
    'rank_trend': 'increasing' if ranks[-1] > ranks[0] else 'decreasing' or 'stable',
    'magnitude_trend': 'exponential' or 'linear' or 'plateau',  # inferred from curve
}
```

---

## Step 7: Write Output JSON

```json
{
  "config": {
    "n_seqs_calib": 200,
    "n_seqs_analysis": 100,
    "batch_size": 4,
    "seq_len": 128,
    "seed": 42,
    "device": "mps",
    "data_path": "data/openwebtext_sample.txt"
  },
  "progressive_configs": [
    {
      "k": 0,
      "ppl": 15.234,
      "layers": [
        {
          "layer": 0,
          "error_magnitude": 0.456,
          "cos_prev_layer": null,
          "effective_rank": 12
        },
        ...
      ]
    },
    ...
  ],
  "summary": {
    "mean_cos_aligned": 0.28,
    "mean_cos_all": 0.18,
    "rank_trend": "increasing",
    "magnitude_trend": "exponential"
  },
  "verdict": "CONFIRMED",
  "notes": "..."
}
```

---

## Code Structure

```
experiments/cascade_error_direction_cycle189.py
├── CONFIG (top of file)
├── Helper functions
│   ├── load_data(path, n_seqs, seed)
│   ├── estimate_mu_neg(model, calib_data, batch_size)
│   ├── build_relu_bias_model(float_model, mu_neg_dict, k)
│   ├── ResidualCollector class
│   ├── compute_error_metrics(e_l, e_prev, threshold=0.90)
│   └── compute_ppl(model, data, batch_size)
├── Main function
│   ├── Load float model
│   ├── Load data
│   ├── Estimate mu_neg
│   ├── for k in [0..11]:
│   │   ├── Build ReLU+bias model
│   │   ├── Collect residuals (float & swapped)
│   │   ├── Compute error metrics
│   │   ├── Append to results
│   │   └── Cleanup
│   ├── Aggregate summary stats
│   └── Write JSON
└── if __name__ == '__main__': main()
```

---

## Verification Checklist

- [ ] mu_neg shapes match FFN hidden sizes (3072 for GPT-2)
- [ ] Progressive PPL closely matches cycle 181 table (within ±2%)
- [ ] All residuals have shape (N_tokens, 768)
- [ ] Error metrics computed for all 12 layers, all k values
- [ ] No NaN/Inf in cosine similarities or effective ranks
- [ ] JSON fully written and valid (test with `json.load()`)
- [ ] Run on sample (k=0,1,11 only) to check for crashes before full sweep
- [ ] Execution time < 30 sec total

---

## Notes

1. **Residual stream definition**: Use output of `model.transformer.h[l]`, which is the full block output including residual connection. This is (batch_size, seq_len, 768).

2. **Calibration vs analysis data**: Use different random splits to avoid overfitting mu_neg to analysis data.

3. **Device handling**: Test both MPS and CPU; provide fallback.

4. **Hook cleanup**: Always remove hooks after use to avoid memory leaks and double-counting in nested runs.

5. **Progressive model building**: Deepcopy ensures no interference between iterations. In-place bias addition is efficient.

6. **Cosine similarity edge case**: If error vector is zero (very rare), add 1e-8 to norms to avoid division by zero.

7. **Effective rank**: Use SVD energy threshold (90%) rather than hard rank cutoff; more interpretable for error analysis.

8. **PPL computation**: Recompute PPL for each swapped model to match cycle 181 reference; validates model construction.
