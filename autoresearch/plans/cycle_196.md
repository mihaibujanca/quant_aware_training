# Cycle 196 Implementation Plan: LayerNorm Recalibration

## Overview

Test whether recalibrating LayerNorm parameters (γ, β) after all-12-block topk+lstsq
compression reduces the cascade PPL cost. Cost: 18,432 params, no training.

**Script**: `experiments/ln_recalibration_cycle196.py`
**Output**: `autoresearch/results/cycle_196_ln_recal.json`

---

## Config

```python
N_SEQS_CALIB = 100   # calibration sequences
SEQ_LEN = 64         # sequence length
SEED = 42
DEVICE = "mps"       # use MPS for speed
K_TOPK = 4           # topk channels (best recipe: PPL ~59 at k=4)
COMPRESSION_RATIO = 6  # 6x
```

---

## Step 1: Build compressed model (all-12 topk+lstsq)

Re-use the existing topk+lstsq recipe from cycle 181/185.
The compressed FFN for each block replaces c_fc with topk k=4 from 3072 channels (512 = 3072/6).

```python
# For each block b:
# 1. Select top-k channels by activation magnitude on calib data
# 2. Fit c_proj_compressed via lstsq: y ≈ X_topk @ W_out_topk^T
# 3. Replace block's mlp with compressed version
```

Critical: the existing topk+lstsq recipe needs to be re-implemented here cleanly.
Look at `experiments/cascade_correction_k5_k8_gpt2.py` or similar for reference.

Actually, the best approach: import the topk+lstsq logic from prior experiments.
Check: `experiments/higher_rank_joint_cycle157.py` and similar files for the recipe.

Simpler self-contained plan: implement topk+lstsq from scratch:

```python
def build_compressed_block(block, calib_activations, k=512):
    """
    calib_activations: (N, 768) residual stream inputs to block's FFN.
    k: number of channels to keep (3072/6 = 512 for 6x compression).
    """
    # Run through ln_2 and c_fc to get pre-GELU activations
    with torch.no_grad():
        ln_out = block.ln_2(calib_activations)  # (N, 768)
        fc_out = F.linear(ln_out, block.mlp.c_fc.weight.T, block.mlp.c_fc.bias)  # (N, 3072)
        # Note: GPT-2 uses Conv1D with weight shape (out, in), but Linear needs (in, out)
        # Check actual Conv1D weight layout: weight is (in_features, out_features) = (768, 3072)
        # So F.linear(x, w.T, b) where w is (768, 3072)
        act_out = F.gelu(fc_out)  # (N, 3072)
        channel_norms = act_out.abs().mean(0)  # (3072,) — channel importance
        topk_idx = channel_norms.topk(k).indices  # (k,)
        act_topk = act_out[:, topk_idx]  # (N, k)

        # float block output (residual + mlp)
        mlp_out = F.linear(act_out, block.mlp.c_proj.weight.T, block.mlp.c_proj.bias)
        target = mlp_out  # (N, 768) — what c_proj should produce

        # Fit lstsq: target ≈ act_topk @ W_compressed^T + bias_compressed
        # Solve: W_compressed = lstsq(act_topk, target)
        W_comp, _, _, _ = torch.linalg.lstsq(act_topk, target, rcond=None)  # (k, 768)

    return topk_idx, W_comp, block.mlp.c_fc.weight, block.mlp.c_fc.bias
```

**GPT-2 Conv1D weight layout**: Conv1D weight is (in_features, out_features).
- `c_fc.weight`: (768, 3072) — transpose for nn.Linear
- `c_proj.weight`: (3072, 768) — transpose for nn.Linear

---

## Step 2: Forward pass with compressed model

```python
class CompressedMLP(nn.Module):
    def __init__(self, block, topk_idx, W_comp, fc_weight, fc_bias, gelu_bias):
        # Store topk_idx, W_comp (k, 768), fc_weight (768, 3072), fc_bias (3072,)

    def forward(self, x):  # x: (B, T, 768)
        # Run c_fc on ALL channels (need full fc for proper topk selection):
        fc_out = x @ self.fc_weight + self.fc_bias  # (B, T, 3072)
        act = F.gelu(fc_out)
        act_topk = act[:, :, self.topk_idx]  # (B, T, k)
        out = act_topk @ self.W_comp  # (B, T, 768)
        return out
```

---

## Step 3: Collect LN input statistics

Run calibration data through float model and compressed model, collecting
the residual stream BEFORE each block's ln_2 (i.e., after attention).

```python
# Hook on block.ln_2 to capture input
float_stats = {}  # b -> (mean, std) per channel
comp_stats = {}

for b in range(12):
    # float_model forward with hook on h[b].ln_2
    # collect all inputs → (N_calib_tokens, 768)
    # compute per-channel mean and std
    float_stats[b] = (inputs.mean(0), inputs.std(0))
    # same for compressed model
    comp_stats[b] = (inputs_comp.mean(0), inputs_comp.std(0))
```

---

## Step 4: Recalibrate LN parameters

```python
eps = 1e-5
for b in range(12):
    # Get current LN-2 parameters
    gamma = compressed_model.transformer.h[b].ln_2.weight  # (768,)
    beta = compressed_model.transformer.h[b].ln_2.bias  # (768,)

    float_mean, float_std = float_stats[b]
    comp_mean, comp_std = comp_stats[b]

    # Recalibrate so that LN(x_comp) ≈ LN_float(x_float)
    # LN normalizes: (x - mean_x) / std_x * gamma + beta
    # We want: ((x_comp - comp_mean) / comp_std) * gamma' + beta' ≈
    #           ((x_float - float_mean) / float_std) * gamma + beta
    # Solution:
    gamma_new = gamma * (float_std / (comp_std + eps))
    beta_new = beta + gamma * (float_mean - comp_mean) / (comp_std + eps)

    compressed_model.transformer.h[b].ln_2.weight.data = gamma_new
    compressed_model.transformer.h[b].ln_2.bias.data = beta_new

    # Optionally also recalibrate ln_1 (before attention):
    # float_stats_ln1[b], comp_stats_ln1[b] similarly
```

---

## Step 5: Evaluate PPL

Measure PPL on wikitext-2 validation set (standard stride-1024, max_length=1024).

---

## Output JSON

```json
{
  "cycle": 196, "item": 207,
  "verdict": "...",
  "baseline_ppl": 22.70,
  "compressed_ppl": 59.0,  # before recalibration
  "recal_ln2_ppl": ...,    # after ln_2 recalibration only
  "recal_ln12_ppl": ...,   # after ln_1 + ln_2 recalibration
  "n_recal_params": 18432,
  "config": {...}
}
```

---

## Verdict

```python
improvement = compressed_ppl - recal_ppl
if improvement > 9:  verdict = "CONFIRMED"  # saves >9 PPL
elif improvement < 2: verdict = "REFUTED"   # near zero benefit
else:                 verdict = "INCONCLUSIVE"
```

---

## Compute Budget

- topk+lstsq fitting: ~5 min for 12 blocks
- LN statistics collection: ~2 min
- PPL evaluation: ~3 min
- Total: ~10-15 min, within budget

---

## Key Implementation Note

The Conv1D weight layout in GPT-2 is TRANSPOSED relative to nn.Linear:
- `c_fc.weight.shape = (768, 3072)` → `F.linear(x, c_fc.weight.T)` for correct matmul
- `c_proj.weight.shape = (3072, 768)` → `F.linear(x, c_proj.weight.T)` for correct matmul

Double-check this by running a forward pass and comparing to the model's mlp forward.
