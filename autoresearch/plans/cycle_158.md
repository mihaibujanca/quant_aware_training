# Task
Item 168: Initialize joint rank-64 training from topk+lstsq solution instead of random init.

# Hypothesis
Topk+lstsq gives a reliable compression (PPL ~95 stacked). Using it to warm-start joint training avoids bad optimization basins seen with random init (cycle 155: 91.1±1.0).

**Criteria (verbatim from agenda):**
- Confirmed if: topk-init joint PPL < 68 AND std < 2.0 across seeds (reliable AND good).
- Refuted if: PPL within 2% of topk+lstsq+polish (~70.7) — warm start doesn't help joint training.
- Inconclusive if: better than random-init but worse than topk+corrections (~59).

# Steps
1. Create `experiments/topk_init_joint_cycle158.py` based on `experiments/joint_multiseed_cycle155.py`.
2. Add initialization function:
   - Run topk selection + lstsq fit per block (reuse `derive_ffn_compression` or replicate logic).
   - Build encoder: rows for selected k=4 indices get identity-like weights; remaining 60 rows zero.
   - Build decoder: first k columns from lstsq W_out; remaining 60 columns zero.
   - **Noise injection**: after setting active dims, add Gaussian noise to ALL encoder rows and decoder columns with scale `1e-3 * std(active_encoder_weights)`. This preserves the warm start to ~3 decimal places while giving all 64 dims nonzero gradient signal from step 1.
   - Copy bias from lstsq fit.
3. Run 3 seeds (42, 7, 137), same hyperparams as cycle 155: 2000 steps, lr=1e-4, cosine schedule.
4. Evaluate stacked PPL with 200 OWT sequences. Record per-seed and summary stats.
5. **Post-training effective rank diagnostic**: for each block's encoder, compute SVD and count singular values > 1% of max. Record per-block effective rank in results JSON. If median rank ≤ 5 after training, flag that joint training failed to expand beyond the init subspace — result is "topk+lstsq with extra Adam steps," not true joint training from warm start.
6. Save results JSON to `autoresearch/results/cycle_158_topk_init_joint.json` (include per-seed PPL, summary stats, per-block effective rank before and after training).
7. Compare: topk-init joint vs random-init (91.1) vs topk+lstsq+polish (70.7).
8. Update `autoresearch/agenda.md`: move item 168 to PAST EXPERIMENTS with commit hash and artifact paths.
9. Commit: `[autoresearch] cycle 158: topk+lstsq init for joint training`

# Validation
- Verify topk-init model evaluates to ~95 PPL *before* joint training (matching known stacked baseline).
- Verify training loss decreases over 2000 steps (not stuck at init).
- Verify all 3 seeds produce valid PPL (not NaN/inf).
- Verify effective rank > 4 after training (confirms dims beyond init activated). If rank stays ≤ 4, the noise was too small or training too short — see fallbacks.
- Cross-check: float baseline PPL matches prior cycles (~30).

# Artifacts
- `experiments/topk_init_joint_cycle158.py`
- `autoresearch/results/cycle_158_topk_init_joint.json`

# Fallbacks
- If effective rank stays ≤ 5 after training: increase noise scale to 1e-2 * std and retry 1 seed. If still stuck, training dynamics are the bottleneck, not init.
- If topk-init PPL before training is far from 95, debug the initialization mapping before joint training.
- If training diverges, halve lr to 5e-5 and retry.
