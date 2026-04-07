# Cycle 297 Plan — Item 307

## Dependencies
Item 305 (cycle 295) infrastructure: `aleph.models.MLP`, `aleph.datasets.make_spirals`, `eff_rank_from_Y` formula. The 2D-input bottleneck flagged in cycle 295 is expected to drive a CONFIRMED-init-rank verdict per item 307.

## Item 307 (verbatim Hypothesis)
eff_rank collapse is gradual and tracks train loss monotonically. No phase transition.

## Approach
New script `experiments/eff_rank_trajectory_cycle297.py`. Train fresh spirals 32×12 net (cycle 295 recipe: input_scale=2.0, lr=1e-3, full-batch BCEWithLogits) for 3 seeds {42, 137, 2024}. Snapshot every 100 training steps: per-layer post-ReLU eff_rank (12 layers), train loss, held-out accuracy. Cap epochs at 12000 (covers cycle 295 max 11325). 1000/1000 train/held split per seed (cycle 296 convention). Aggregate per-layer trajectories across seeds. Plot per-layer eff_rank vs step plus loss/acc curves. Compute (i) init eff_rank per layer (step 0), (ii) final eff_rank per layer (last snapshot), (iii) max single-window collapse rate (max drop within any 100-step window per layer), (iv) step at which each layer first reaches 1.5× its final value (note: with cycle 295 evidence that init ≤ final for most layers, this may be undefined for layers that monotone-increase).

## Files
- `experiments/eff_rank_trajectory_cycle297.py` (new)
- `autoresearch/results/cycle_297_eff_rank_trajectory.json`
- `autoresearch/results/cycle_297_summary.md`
- `autoresearch/results/cycle_297_trajectory.png` (optional plot)
- `autoresearch/results/cycle_297_manifest.txt`
- `autoresearch/results/cycle_297_exec_result.txt`
- `autoresearch/logs/cycle_297_run.log`
- `autoresearch/audits/cycle_297.md`
- `autoresearch/results/cycle_297_audit.txt`

## Config
HIDDEN=32, DEPTH=12, N=2000 (1000 train / 1000 held per seed via random split), SPIRALS_NOISE=0.5, N_TURNS=3, INPUT_SCALE=2.0 (MEMORY.md depth-12 recipe), LR=1e-3, MAX_EPOCHS=12000, SNAPSHOT_EVERY=100, TARGET_ACC=0.95 (logged but not used to early-stop; we keep training to capture full trajectory until either MAX_EPOCHS or train_acc≥0.99 to avoid runaway). Adam, BCEWithLogits, full-batch, CPU. Seeds {42, 137, 2024}. Trainable per net = 11,745 (asserted at runtime).

## Verification
- Per-snapshot per-layer eff_rank ∈ [1.0, 32.0] sanity bound asserted.
- Snapshot t=0 must equal the init eff_rank used in cycle 295 within 1e-5 (cross-check).
- All 3 seeds must reach train_acc ≥ 0.95 by MAX_EPOCHS or be flagged "did not converge" (interpretation guarded).
- Total wall ≤ ~10 minutes (cycle 295 spirals: 14–26s/seed at convergence; collecting eff_rank every 100 steps adds eigvalsh on (32,32) matrices ×12 ×N_snapshots, trivial).

## Verdict logic (item 307 verbatim)
- **CONFIRMED gradual** if per-layer eff_rank curves are monotone-decreasing across the entire training run AND no layer drops by ≥50% within any 100-step window.
- **CONFIRMED phase-transition** if any layer's eff_rank drops by ≥50% within a 100-step window.
- **CONFIRMED init-rank** if initial eff_rank is already ≤5 for ≥6 layers.
- **REFUTED** if eff_rank doesn't collapse at all.
Apply per-seed; report all three. Overall verdict by majority across seeds; tie → INCONCLUSIVE. Multiple CONFIRMED-* clauses can apply (e.g. CONFIRMED-init-rank AND CONFIRMED-gradual).

## Baselines
Cycle 295 spirals init_mean per layer (3 seeds): [2.99, 2.71, 1.87, 1.19, 1.06, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]. Cycle 295 spirals post_mean: [3.14, 2.86, 2.11, 2.09, 2.15, 1.94, 1.99, 2.07, 2.05, 2.03, 1.76, 1.74]. Random init expected ≈32 (only meaningful for high-dim input lift).
