# Cycle 168 Plan: Training Dynamics — Scale Comparison (Item 178)

## Task
Compare z_mean (pre-GELU FFN activations) across GPT-2 small (12L/768d), medium (24L/1024d) and large (36L/1280d). Cycle 165 already proved negativity is learned (random-init ≈ 0, trained ≈ -1). This cycle tests scale-universality.

## Hypothesis
z_mean ≈ -1 is a universal property of GELU transformers, not specific to GPT-2 small. All three scales should show similar z_mean magnitude, with the anti-alignment pattern emerging regardless of width/depth.

## Pre-flights
- P1 (duplicate): cycle 165 tested GPT-2 small only. No prior cycle compared across model scales.
- P2 (deps): none. Standalone analysis.
- P3 (scope): 3 forward passes, ~10 min. Single cycle.
- P4 (program.md): no modifications needed.

## Novelty assessment
Cycle 165 answered parts (a) and (c) of item 178 (random-init z_mean ≈ 0, confirming negativity is learned; no training checkpoints available). Part (b) — scale comparison — is genuinely novel. This is a thin but clean cycle: one clear testable prediction (scale-invariance of z_mean).

## Steps
1. Load GPT-2 small, medium, large from HuggingFace.
2. Tokenize 4096 OWT tokens (from `data/openwebtext_sample.txt`).
3. For each model, hook `mlp.c_fc` output (pre-GELU z) at all FFN blocks.
4. Forward pass, collect z per block.
5. Per block: compute z_mean, decompose into Wx_mean and b_mean.
6. Summary statistics across scales: mean z_mean, std across blocks, range.
7. Save JSON, plot comparison (z_mean per block for all 3 models on one chart).

## Verdict logic
```
z_mean_small = mean across blocks of z_mean (from cycle 165: ≈ -0.93)
z_mean_medium = mean across blocks of z_mean
z_mean_large = mean across blocks of z_mean

max_ratio = max(|z_mean_X|) / min(|z_mean_X|) across {small, medium, large}
if max_ratio < 2.0 AND all three have z_mean < -0.3:
    CONFIRMED — negativity is universal across scales
elif max_ratio > 2.0:
    REFUTED — z_mean magnitude scales with model size
else:
    INCONCLUSIVE
```

## Validation
- GPT-2 small z_mean should match cycle 165 results (≈ -0.93 mean across blocks).
- All models should produce sensible PPL on OWT (small: 20-25, medium: 16-20, large: 14-18).

## Artifacts
- `experiments/zmean_scale_comparison_cycle168.py`
- `autoresearch/results/cycle_168_scale_comparison.json`
- `autoresearch/results/cycle_168_scale_comparison.png`

## Fallbacks
- If GPT-2 large doesn't fit in memory, use CPU inference or reduce to 2048 tokens.
- If medium/large download is slow, proceed with small+medium only and note.
