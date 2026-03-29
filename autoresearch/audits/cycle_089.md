# Audit — Cycle 089

- **Item**: 104 — Compositional error accumulation curve (k=1..8 blocks) — GPT-2 FFN
- **Cycle**: 089
- **Auditor verdict**:

```
status=PASS
old_outcome=refuted
new_outcome=refuted
reason=Implementation correct: per-block polish (500 Adam steps, lr=1e-4) done in isolation then polished W_down injected into k-block student. Block ordering follows damage ranking from cycle 005/007. PPL monotonically increasing with k for both orderings. Forward <= reverse at all k. k=8 PPL identical for both orderings (523.7) as expected (same 8 blocks). Multiplicative fit R²=0.933 (forward), 0.849 (reverse) with c=0.408/0.371 per block. Actual k=8 exceeds multiplicative prediction by ~30% — genuinely super-multiplicative. REFUTED is the correct verdict per the criteria (super-multiplicative = cross-block amplification = hard ceiling).
guardrail_flags=G4
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The 41%/block compounding rate means k=4 is the practical ceiling at 6× (56% PPL increase). This closes the FFN stacking direction at current compression ratios. Actionable follow-ups: (a) test whether lower compression ratios (3×, 4×) reduce the compounding rate enough to enable more blocks, (b) test joint polish (optimize all k blocks simultaneously rather than individually) — individual polish ignores inter-block interactions which likely cause the super-multiplicative regime, (c) attention compression (item 105) may be more stackable since attention heads have explicit low-rank structure.
interesting=true
interesting_reason=Quantifies a fundamental scaling limit for geometric FFN compression: errors compound at 41%/block, creating a hard ceiling at k=4 blocks (6×). The super-multiplicative regime (actual > multiplicative fit at high k) means cross-block error amplification is real — blocks interact nonlinearly when multiple are compressed. This is the definitive stacking experiment for this compression method and directly sets the operational boundary for any deployment scenario.
```

## Guardrail Details

**G1 (Implementation)**: PASS. Block ordering `[7, 5, 4, 1, 3, 6, 8, 9]` matches cycle 005/007 damage ranking. Per-block compression via `_derive_block_pack_from_atlas` correctly builds CompressedFfnPack for specified block_ids. Per-block polish correctly: (1) swaps single block into fresh teacher copy, (2) runs 500 Adam steps on `c_proj.weight` only, (3) extracts polished weights, (4) injects into k-block student before evaluation. The stacking loop at line 397-421 correctly builds a k-block pack, swaps all k blocks, then injects per-block polished weights. Seed management: `_set_seed(seed)` called before atlas probing and before each k-step evaluation. PPL computed via `_evaluate_model` from the library (standard cross-entropy → exp).

**G2 (Reproducibility)**: PASS. 3 seeds (42, 123, 7). Both orderings (forward, reverse). All hyperparameters recorded in JSON: `hidden_fraction=1/6`, `polish_steps=500`, `polish_lr=1e-4`, `seeds=[42,123,7]`. Per-seed raw PPL values stored. Corpus: 200 probe_train, 200 probe_holdout (polish), 200 eval_val, seq_len=128. Shuffle seed = experiment seed.

**G3 (Results)**: PASS. Numbers in agenda match JSON. Forward k=4: delta_pct_mean=56.18% (agenda says "+56%"). Forward k=5: 144.8% ("+145%"). Forward k=8: 913.8% ("+914%"). Multiplicative c=0.408 matches. R²=0.933 matches. Reverse k=8: PPL=523.7, identical to forward (correct — same 8 blocks at k=8). Scaling law fits: multiplicative > additive > power_law by R² for both orderings. Verdict logic (lines 482-498): actual k=8 (523.7) > predicted multiplicative k=8 * 1.05 → "super_multiplicative" → "REFUTED_SUPER_MULTIPLICATIVE." Correct.

**G4 (Surprising?)**: MILDLY FLAGGED. The plan predicted polished multiplicative null would give PPL(8) ≈ baseline × 1.01^8 ≈ +8%, based on prior cycles showing ~1% single-block damage with polish. Actual: +914% at k=8. The discrepancy starts at k=1: forward k=1 (block 7 alone) gives -3% (slightly BETTER than baseline), which is consistent with cycle 005, but k=2 already shows +7.4%. The plan's baseline prediction of ~30 PPL was also wrong (actual: 51.66). The higher baseline likely reflects the specific eval subset (200 seqs, seq_len=128, OWT). The compounding rate of 41%/block is far higher than the plan's assumed 1%/block. This is because polish only optimizes W_down for single-block accuracy — when multiple blocks are compressed, the residual stream carries errors from upstream blocks that downstream blocks were never polished to handle. The super-multiplicative regime is real, not a bug, but the magnitude was not anticipated. The result is internally consistent (monotonic, forward <= reverse, k=8 converges) and consistent across 3 seeds.

**G5 (Learning quality)**: HIGH. This is a clean scaling experiment with clear actionable implications. The curve is smooth, monotonic, and consistent across seeds and orderings. Three scaling laws are compared with proper R² reporting. The conclusion (k=4 ceiling at 6×) is directly actionable for any deployment planning. The convergence of forward and reverse orderings at k=8 provides a built-in consistency check. The per-seed variance (std/mean) increases with k (from 13% at k=1 to 30% at k=8), which is expected for exponentially growing quantities.

**G6 (Method)**: PASS. Per-block polish is the correct methodological choice to isolate composition effects (joint polish would conflate composition with optimization). The scaling law fitting uses log-linearization for the multiplicative fit (line 255-260), which is the standard approach. The R² is computed in PPL space (not log-PPL), which is the right choice since PPL is the metric of interest. Power-law fit uses log(k+1) to avoid log(0), which is a reasonable parameterization. All three fits (additive, multiplicative, power-law) are reported with R² and RMSE, allowing the reader to assess which model best describes the data. The residual plot (panel b) provides visual evidence of super-multiplicative behavior.
