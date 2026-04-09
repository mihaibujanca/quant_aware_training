# Cycle 375 — Item 383: Adapter dead-neuron census and exact pruning — ReLU MLP adapter (R2)

## Pre-flight

### P1: Duplicate check — PASS

No prior cycle has examined adapter neuron liveness. c128/c162/c192 studied GPT-2 FFN GELU channels (0 dead). This targets the ReLU adapter's 64-dim hidden layer.

### P2: Dependency check — CONDITIONAL PASS

Item says "Depends on: trained 6-block adapter from c357 or c359." No saved `.pt` files exist — prior experiments train in-memory only. The plan includes training as step 1. Uses the same recipe as c374 (joint training, k=512, h=64, 6 blocks).

### P3: Falsifiability — PASS

- Confirmed: >= 10% adapter neurons prunable AND pruning PPL delta < 0.1
- Refuted: < 5% prunable OR pruning PPL delta > 0.5
- Inconclusive: 5-10% prunable or 0.1-0.5 PPL delta

### P4: Cost estimate — PASS

One 6-block training run (1500 steps, seed 42) ~10 min on MPS. Census is pure inference (deterministic, fast). Total ~12 min.

## Plan

### Script: `experiments/gpt2_adapter_census_cycle375.py`

Fork `experiments/gpt2_adaptive_k_cycle374.py` with these changes:

1. **Config**: Single seed (42). `ALL_BLOCKS = [2, 1, 10, 4, 3, 11]`, `K = 512`, `ADAPTER_HIDDEN = 64`, `N_STEPS = 1500`, `LR = 1e-4`. Remove adaptive-k machinery. `OUTPUT_PATH = "autoresearch/results/cycle_375_adapter_census.json"`.

2. **Training phase**: Reuse `compress_blocks`, `install_compressed_blocks`, `train_adapters_joint` from c374 (uniform k=512). Train once with seed 42. Record post-training PPL as `baseline_ppl`.

3. **Census phase** (new code, after training):
   - For each compressed block `b` in ALL_BLOCKS:
     - Access `model.transformer.h[b].mlp` (the `CompressedGPT2MLPWithAdapter`).
     - Run all calibration sequences through the full model, hooking the adapter's pre-ReLU activations: `pre_relu = F.linear(h, adapter_W1, adapter_b1)`.
     - Collect pre-ReLU activations across all calibration tokens (shape: [N_tokens, 64]).
     - Classify each neuron j (0..63):
       - **Dead**: `pre_relu[:, j].max() <= 0` (never activates)
       - **Always-active**: `pre_relu[:, j].min() > 0` (always in linear regime)
       - **Mixed**: otherwise (crosses ReLU boundary)
     - Report counts per block.

4. **Pruning phase**:
   - **Prune dead neurons**: For each dead neuron j, delete row j from W1/b1, column j from W2. This is exact (zero output contribution).
   - **Fold always-active neurons**: For always-active neuron j, the path is linear: contribution = `W2[:, j] * (W1[j, :] @ h + b1[j])`. Fold into a linear term `W_linear += outer(W2[:, j], W1[j, :])`, `b_linear += W2[:, j] * b1[j]`. Then delete neuron j from W1/b1/W2.
   - Add the folded linear term to `proj_weight` and `proj_bias` (since adapter output is summed with proj output).
   - Eval PPL after pruning → `pruned_ppl`. Compute delta = `pruned_ppl - baseline_ppl`.

5. **Output JSON**:
   ```json
   {
     "baseline_ppl": ...,
     "census": {
       "block_2": {"dead": N, "always_active": N, "mixed": N, "total": 64},
       ...
     },
     "total_neurons": 384,
     "total_dead": ..., "total_always_active": ..., "total_prunable": ...,
     "pct_prunable": ...,
     "pruned_ppl": ...,
     "ppl_delta": ...,
     "verdict": "CONFIRMED|REFUTED|INCONCLUSIVE"
   }
   ```

6. **Verdict logic**:
   - `pct_prunable >= 10 and ppl_delta < 0.1` → CONFIRMED
   - `pct_prunable < 5 or ppl_delta > 0.5` → REFUTED
   - else → INCONCLUSIVE

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_adapter_census_cycle375.py
```

### Verification

- Baseline PPL must be in [63, 68] range (consistent with prior 6-block results).
- Dead neuron pruning must be exactly lossless (PPL delta = 0.000 to float precision).
- Always-active folding must be exactly lossless (PPL delta = 0.000 to float precision).
- Any non-zero delta indicates a bug in the pruning/folding code, not a real effect.
