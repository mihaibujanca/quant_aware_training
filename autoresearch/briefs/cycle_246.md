# Item
256 (P0): gelu_noneg + μ + LoRA r=8 + K=128 W_out compression — does the bump-removed activation absorb channel compression?
Combine cycle 239's recipe (gelu_noneg + per-channel μ + LoRA r=8 on c_proj, all 12 blocks) with K=128 W_out compression on BLOCKS=[1,4,7].

# Context
- Cycle 223 (CONFIRMED): K=128 W_out only, BLOCKS=[1,4,7] → 48.989.
- Cycle 239 (CONFIRMED): gelu_noneg + μ + LoRA r=8 (no K compression), all 12 blocks → 47.692 ± 0.143.
- Open question: do the two mechanisms compose, trade, or conflict?

# Dependencies
- 228 / cycle 228 (μ_l recipe).
- 234 / cycle 234 (LoraConv1D wrapper).
- 239 / cycle 239 (disjoint splits + LoRA framework).

# Key Files
- experiments/gelu_noneg_lora_disjoint_cycle239.py — direct template.
- experiments/adam_lower_k_cycle223.py — K=128 topk+lstsq init.

# Constraints
- All 12 blocks: relu(gelu(z)) + μ_l (cycle 228 recipe).
- BLOCKS=[1,4,7] additionally compressed: K=128 topk+lstsq W_out replacing the original c_proj. The COMPRESSED c_proj has shape (K=128, 768) instead of (3072, 768).
- LoRA r=8 on EVERY block's c_proj — for [1,4,7] this is on the COMPRESSED (K=128) c_proj; for the other 9 blocks it's on the float c_proj.
- LoRA dimensions:
  - For float c_proj (shape 3072, 768): A ∈ R^(3072, 8), B ∈ R^(8, 768).
  - For compressed c_proj at K=128 (shape 128, 768): A ∈ R^(128, 8), B ∈ R^(8, 768).
- Trainable params: 9 × (3072×8 + 8×768) + 3 × (128×8 + 8×768) + 3 × (W_out (128×768) + b_out 768) = 9 × 30720 + 3 × (1024 + 6144) + 3 × 99072 = 276480 + 21504 + 297216 = 595,200 ≈ 595K.
- Adam LR=1e-4, 2000 steps, EVAL_EVERY=200, best-state restore.
- Disjoint calib/eval (cycle 239 splits).
- 3 seeds (42, 137, 2024) if budget allows; else 1.
- Confirmed if best_mean ≤ 47.0.
- Refuted if best_mean ≥ 49.5.
- Inconclusive if 47-49.5.

# Handoff
Plan agent: combine cycle 239 + cycle 223 carefully. Install order:
1. gelu_noneg + μ patches (12 blocks)
2. K=128 topk+lstsq compression on [1,4,7]'s c_proj
3. LoRA r=8 on every block's c_proj (compressed or float)
4. Adam trains LoRA + compressed W_out + b_out

New script `experiments/gelu_noneg_lora_k128_cycle246.py`. Output JSON + summary.
