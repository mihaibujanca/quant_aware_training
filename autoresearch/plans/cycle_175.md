# Cycle 175 Plan — Item 186

## Pre-flights
1. Verify `data/openwebtext_sample.txt` exists
2. Verify GPT-2 loads and baseline PPL ∈ [15, 65]
3. Verify MPS device available

## Approach
Single script `experiments/prediction_impact_gelu_relu_cycle175.py`:

### Per-block loop (b=0..11)
1. Load fresh GPT-2 model
2. **Float forward**: run full model on 2048 tokens (16 seqs × 128 tokens), collect logits → P_float = softmax(logits, dim=-1)
3. **Hook block b**: register forward hook on `model.transformer.h[b].mlp` that intercepts the GELU activation and replaces with ReLU: `output = relu(c_fc_output) @ c_proj.weight + c_proj.bias` instead of `gelu(c_fc_output) @ c_proj.weight + c_proj.bias`
   - Actually simpler: hook the activation function. GPT-2 MLP computes `h = act(c_fc(x))`, then `c_proj(h)`. We replace `act` from GELU to ReLU by hooking c_proj input.
   - Implementation: hook on c_fc output, apply ReLU instead of GELU, pass to c_proj.
4. **Modified forward**: get logits → P_relu = softmax(logits, dim=-1)
5. **KL computation**: per-token KL(P_float || P_relu) = sum(P_float * (log P_float - log P_relu))
6. **Statistics**:
   - mean_KL, median_KL, max_KL
   - top-50 tokens by KL: decode token, 5-token context window
   - fraction where P_relu(correct) > P_float(correct)
7. **Signal norms**: compute ||y_neg|| and ||s_neg|| per block using cycle-174 methodology (hook c_proj input, mask negative GELU outputs, project through c_proj)
8. **Correlation**: Pearson r(mean_KL_per_block, mean_||y_neg||_per_block) across 12 blocks

### Output
- JSON: `autoresearch/results/cycle_175_prediction_impact.json`
- PNG: `autoresearch/results/cycle_175_prediction_impact.png` (2-panel: KL per block, correlation scatter)
- Console: top-50 token text dump per block (scan for semantic patterns)

## Risks
- MPS memory for 12 separate model loads → use hooks, reload model once, swap hooks
- Softmax precision: use log_softmax to avoid numerical issues in KL
- Token decoding: handle BPE fragments correctly

## Verdict logic
```
if mean_KL > 0.1 in >=8 blocks AND top-50 shows semantic pattern:
    CONFIRMED
elif mean_KL < 0.01 in >=8 blocks:
    REFUTED
else:
    INCONCLUSIVE
```

## Time budget: ~10 min (forward passes only, no training)
