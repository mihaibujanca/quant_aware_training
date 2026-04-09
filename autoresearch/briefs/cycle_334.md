# Cycle 334 Brief

## Item
Item 328: Extended budget rank-1 retrain at b11 — 8000 steps to settle cycle 311's borderline pass

## Context
- Cycle 311 tested GPT-2 b11 rank-1 LASER retrain (4000 steps, 3 seeds). Result: best_mean_ppl = 59.7074 (passed the 60 bar by only 0.29 PPL). Verdict was CONFIRMED (4/4 blocks passed).
- b11 trajectory from cycle 311: all 3 seeds still slowly descending at step 4000, best_steps = {3600, 3600, 3900}. Slopes at steps 3000-4000: seed 42 = -0.09 ppl/1000 steps, seed 137 = -0.10, seed 2024 = -0.02. Very shallow descent.
- b11 r=1 init (no-retrain) PPL = 60.6103. The retrain only recovered 0.9 PPL over 4000 steps.
- This cycle extends to 8000 steps with an LR step decay (1e-4 for 0-4000, 5e-5 for 4000-8000) to see if PPL can break below 58.
- Infrastructure: LowRankCProj class and training loop from `experiments/rank1_all_concentrators_cycle311.py`. HF Conv1D layout: weight (3072, 768), forward = x @ W + b. A = U[:,:1]*S[:1] frozen (3072,1), B = Vh[:1,:] trainable (1,768), bias trainable (768,). Total trainable = 1,536.

## Dependencies
- Cycle 311 JSON results: `autoresearch/results/cycle_311_rank1_all_concentrators.json` (for cross-validation of init PPL and step-4000 values)
- Cycle 311 script: `experiments/rank1_all_concentrators_cycle311.py` (LowRankCProj class, data loading, eval_ppl)
- GPT-2 small pretrained weights (HuggingFace)
- wikitext-2-raw-v1 validation split

## Key Files
- Template: `experiments/rank1_all_concentrators_cycle311.py`
- Prior results: `autoresearch/results/cycle_311_rank1_all_concentrators.json`
- Output script: `experiments/b11_extended_budget_cycle334.py`
- Output JSON: `autoresearch/results/cycle_334_b11_extended.json`

## Constraints
- Single block (b11 only), 3 seeds, 8000 steps. ~30 min wall time on MPS.
- LR schedule: 1e-4 for steps 0-4000, 5e-5 for steps 4000-8000 (step decay at midpoint).
- Must cross-validate: init PPL = 60.6103 (cycle 311 value), step-4000 best_ppl within 0.05 of cycle 311 per-seed values.
- Verdict bars: best_mean_ppl ≤ 58 → CONFIRMED-robust-r1; ≥ 60 → REFUTED-borderline-artifact; else INCONCLUSIVE.

## Risk Assessment
- Slope analysis from cycle 311 suggests the trajectory is nearly flat at step 4000. Best-case slope (-0.10/1000) extrapolated over 4000 more steps at half LR would give ~0.2-0.4 ppl improvement → final ~59.3-59.5. This is well above the 58 bar. INCONCLUSIVE is the likely outcome.
- This is still useful: confirms whether b11's 59.7 is a true plateau vs a still-descending curve.

## Handoff
Write single-file script based on cycle 311 template. Block 11 only. Add LR scheduler (step decay at 4000). Report trajectory_last_10_evals for plateau analysis. Cross-validate against cycle 311 values at init and step 4000.
