# Cycle 340 Brief

## Item
Item 349: Re-run cycle 332 eff_rank with exp(entropy) formula — canonical cross-activation comparison

## Context
- Cycle 332 trained 3 tiny GPTs (GELU/ReLU²/LeakyReLU², 4L 256d) across 3 seeds
- Measured eff_rank using participation ratio: (Σλ)² / Σλ²
- Results (seed 42 only): GELU [90,108,114,89], ReLU² [109,138,138,111], LeakyReLU² [107,152,149,119]
- Project standard is exp(entropy): exp(-Σ p_i log p_i) where p_i = λ_i / Σλ
- Need to recompute with correct formula for cross-cycle comparability

## Key Question
Does exp(entropy) eff_rank still show squared activations > GELU by ≥10% relative at ≥2 blocks?

## Key Files
- `experiments/tiny_gpt_act_compare_cycle331.py` — TinyGPT model class
- `experiments/tiny_gpt_act_compare_checkpoints/{gelu,relu2,lrelu2}_s42.pt` — checkpoints (seed 42)
- `autoresearch/results/cycle_332_eff_rank.json` — participation ratio results
