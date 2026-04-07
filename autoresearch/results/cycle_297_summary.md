# Cycle 297 — eff_rank trajectory on spirals (item 307)

**Verdict:** CONFIRMED-init-rank

## Config
- MLP(2, 32, 1, depth=12, ReLU); Adam lr=0.001; full-batch BCEWithLogits
- Spirals N=2000 (1000 train / 1000 held), noise=0.5, n_turns=3, input_scale=2.0
- Snapshot every 100 epochs; max 12000; early stop at train_acc >= 0.99
- Seeds: [42, 137, 2024]
- Trainable params/net = 11745

## Verdict logic (item 307 verbatim)
- CONFIRMED gradual: per-layer eff_rank monotone-decreasing AND no >=50% drop in any 100-step window
- CONFIRMED phase-transition: any layer drops by >=50% within a 100-step window
- CONFIRMED init-rank: initial eff_rank <= 5 for >= 6 layers
- REFUTED: eff_rank doesn't collapse at all
- Multiple CONFIRMED-* clauses can apply simultaneously.

## Per-seed results

### seed=42
- final_epoch=12000  train_acc=0.9800  converged=True  snapshots=121
- init eff_rank: [2.777, 3.026, 1.695, 1.13, 1.046, 1.006, 1.002, 1.0, 1.0, 1.0, 1.0, 1.0]
- final eff_rank: [3.129, 3.583, 2.51, 1.798, 1.352, 1.165, 1.125, 1.222, 1.392, 1.156, 1.123, 1.16]
- max single-window relative drop per layer: [0.002, 0.196, 0.05, 0.125, 0.283, 0.325, 0.323, 0.303, 0.123, 0.252, 0.186, 0.252]
- monotone non-increasing: False
- n_init_le5: 12/12  (CONFIRMED-init-rank iff >=6)
- any 50% drop: False  (CONFIRMED-phase iff true)
- max relative drop: 0.3246
- epoch at 1.5x final per layer: [None, None, None, None, None, None, None, None, None, None, None, None]
- flags: ['CONFIRMED-init-rank']

### seed=137
- final_epoch=12000  train_acc=0.9700  converged=True  snapshots=121
- init eff_rank: [2.861, 2.315, 1.69, 1.291, 1.086, 1.011, 1.002, 1.001, 1.0, 1.0, 1.0, 1.0]
- final eff_rank: [2.947, 2.668, 2.338, 2.074, 1.957, 1.839, 2.456, 2.842, 2.035, 2.223, 2.038, 2.121]
- max single-window relative drop per layer: [0.013, 0.039, 0.006, 0.009, 0.021, 0.202, 0.059, 0.092, 0.135, 0.07, 0.131, 0.078]
- monotone non-increasing: False
- n_init_le5: 12/12  (CONFIRMED-init-rank iff >=6)
- any 50% drop: False  (CONFIRMED-phase iff true)
- max relative drop: 0.2021
- epoch at 1.5x final per layer: [None, None, None, None, None, None, None, None, None, None, None, None]
- flags: ['CONFIRMED-init-rank']

### seed=2024
- final_epoch=12000  train_acc=0.9870  converged=True  snapshots=121
- init eff_rank: [3.296, 2.7, 2.196, 1.129, 1.038, 1.013, 1.002, 1.001, 1.0, 1.0, 1.0, 1.0]
- final eff_rank: [3.584, 2.917, 2.848, 2.746, 2.11, 2.251, 1.897, 1.955, 1.611, 1.297, 1.17, 1.151]
- max single-window relative drop per layer: [0.006, 0.306, 0.231, 0.007, 0.044, 0.054, 0.033, 0.053, 0.153, 0.204, 0.105, 0.103]
- monotone non-increasing: False
- n_init_le5: 12/12  (CONFIRMED-init-rank iff >=6)
- any 50% drop: False  (CONFIRMED-phase iff true)
- max relative drop: 0.3063
- epoch at 1.5x final per layer: [None, None, None, None, None, None, None, None, None, None, None, None]
- flags: ['CONFIRMED-init-rank']

## Overall verdict
- CONFIRMED-init-rank (flag counts across 3 seeds: {'CONFIRMED-init-rank': 3})

Total wall: 28.8s
Plot: /Users/mihai/Projects/quant_aware_training/autoresearch/results/cycle_297_trajectory.png
