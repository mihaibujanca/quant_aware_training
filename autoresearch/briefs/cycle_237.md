# Item
247 (P0): T3 — Bump subspace ablation vs random subspace baseline.
For each block, take top-10 right singular vectors of centred Δ̃ from cycle 235's t1_features.pt. Define projector P_l = V_l[:,:10] @ V_l[:,:10].T. After block l's FFN output is added to residual stream, replace x_l ← x_l - P_l x_l. Measure full-model PPL on held-out wikitext-2 val.

# Context
- Cycle 235 (T1, CONFIRMED with G3): top-10 right SVs of centred Δ̃ per block. Saved as V_l in t1_features.pt.
- Cycle 236 (T2, INCONCLUSIVE): bump directions are causally active (max 29× random) but T1's wte projection is wrong probe.
- T3 tests whether ablating the bump SUBSPACE (not just along directions) damages the model — and whether this damage is bigger than ablating a random 10-dim subspace.

# Dependencies
- 245 / cycle 235 (done) — provides V_l (768, 32), use first 10.

# Key Files
- autoresearch/results/cycle_235_t1_features.pt — V_l per block.
- experiments/bump_t1_features_cycle235.py — for the data loader pattern.
- experiments/surgical_bump_removal_cycle220.py — for the eval_ppl pattern (200 wikitext-2 val seqs × 128).
- notebooks/bump_features_explore.py — loader contract.

# Constraints
- Held-out: wikitext-2 val, N_SEQS=200, SEQ_LEN=512 per agenda.
- For each block l: P_l = V_l[:, :10] @ V_l[:, :10].T  (768, 768)
- Ablation: after block l's residual update, x_l = x_l - P_l @ x_l.
- This is a forward hook AFTER block l's forward (i.e., on transformer.h[l].register_forward_hook that modifies the output).
- Comparisons:
  - baseline (no ablation)
  - joint bump ablation (all 12 blocks simultaneously)
  - per-block ablation (12 single-block runs, one block at a time)
  - random k=10 baseline (10 seeds Gaussian QR-orthonormal subspaces in R^768), joint and per-block (10 random seeds each)
  - top-10 right SVs of FULL FFN OUTPUT (not just Δ̃) — needs computing on calibration corpus
  - no ablation baseline
- Output filenames per item 247:
  - experiments/bump_t3_ablation_cycle237.py
  - autoresearch/results/cycle_237_t3_ablation_ppl.json
  - autoresearch/results/cycle_237_t3_summary.md
- Confirmed if joint bump ≥ 1.5× random AND ≥3 single-block bump ≥ 2× random.
- Refuted if bump ≤ 1.1× random.
- Inconclusive otherwise.

# Handoff
Plan agent: design the script. Tricky parts:
1. Hook on h[l] to modify the output: forward_hook receives output as (residual_stream, present_key_value, ...) tuple — need to modify the first element.
2. P_l ablation: x_modified = x - x @ P_l (since P_l is symmetric, x @ P_l = (P_l x)^T, but use the right convention based on shape).
   For x ∈ R^(B, T, 768) and P_l ∈ R^(768, 768): x_proj = x @ P_l. x_modified = x - x_proj.
3. Top-10 right SVs of FULL FFN output: need a separate forward pass + SVD on the FFN OUTPUT (not Δ̃) per block.
4. Wall time: baseline + 12 single-block bump + joint bump + 10 random seeds × 13 ablations + full-FFN baseline = ~15 + 130 = 145 PPL evals. At ~30s each = 70 min — TIGHT. Use single-seed random baseline if needed.

New script `experiments/bump_t3_ablation_cycle237.py`. Output 3 files per the spec.
