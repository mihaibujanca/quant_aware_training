# Cycle 264 Audit — Item 274

## Verdict
INCONCLUSIVE.

## What was claimed
For each k ∈ {0..11}, install ReLU+b_leak ONLY at GPT-2 small block k, compute pullback κ profile (layers 1..11) under G = U^T U on the 200×128 wikitext-2 calib slice (seed 42), build the 12×11 Δκ matrix vs the all-GELU baseline, rank swaps by ‖Δκ‖₁, test the hypothesis "diagonal band ≥ 3× off-diagonal AND late-layer monotone amplification as k decreases".

## What was actually done
- `experiments/per_block_swap_curvature_cycle264.py` — fresh script, reuses cycle 258's `ReluPlusBleak` module verbatim and the cycle 258 b_leak precompute hook recipe (E_x[gelu(z) − relu(z)] in 3072-d intermediate space, per block per channel).
- 200×128 wikitext-2 val calib slice, seed 42, MPS — same loader as cycle 258.
- Phase 1: hook all 12 c_fc inputs once, compute b_leak per block. Sanity: b_leak[k] mean ranges from −0.0916 (block 1) to −0.1113 (block 11) — matches cycle 258 manifest exactly.
- Phase 2: baseline κ via `collect_residual_trajectory` + `compute_curvature(reduce='mean')`. Result: κ_baseline = [5.87e-5, 2.23e-3, 2.22e-3, 1.85e-3, 1.09e-3, 6.67e-4, 3.36e-4, 1.89e-4, 1.04e-4, 2.34e-5, 3.84e-6] — **matches cycle 258 GELU κ to all reported digits** (cross-check passes).
- Phase 3: per k ∈ {0..11}, patch block k with `ReluPlusBleak(b_leak[k])`, assert exactly 11 of 12 blocks remain at original GELU module identity, eval PPL, collect trajectory, compute κ, restore. Post-restore PPL = 45.0637 = baseline (round-trip clean, asserted < 1e-4).
- Phase 4: build Δκ matrix (12, 11), compute L1 norms, diagonal band stats, late-layer monotonicity, verdict.
- Phase 5: heatmap with diagonal band overlay rectangles.

## Verdict logic check
- **Clause 1 (diagonal dominance)**: max|Δκ|_diag = 8.00e-4 (at k=2, l=2). median|Δκ|_offdiag = 0 (because 54/100 off-diagonal entries are *causally* bit-zero — swap_at_k cannot affect κ_l for l < k − 1). The implementation uses median, which makes the ratio inf and trivially passes. The substantive check (mean diag / nonzero offdiag mean) = 1.15e-4 / 2.15e-5 ≈ 5.4×. Either way clause 1 passes.
- **Clause 2 (late-layer monotone amplification as k decreases)**: 0/4 of l ∈ {7..10} columns are non-decreasing as k goes from 6 down to 0. l=7 column over k=0..6: [1.0e-6, 1.4e-6, 2.8e-6, 4.7e-6, 1.9e-6, 9.7e-6, 1.9e-6] — non-monotonic. Clause 2 FAILS cleanly.
- **REFUTED clause** (`diag/offdiag<1.0` OR `max_diag<median_offdiag`): does not trigger. Verdict INCONCLUSIVE is the correct branch.

## Sanity checks
1. **Causal lower triangle**: 86/132 = 65% of entries are *bit-identical* zero. For swap_at_k=11 only κ index 10 is nonzero (the only κ index that depends on block 11's output, since κ_l is a 3-point central difference using x^l−1, x^l, x^l+1). Verified manually from the JSON. The lower triangle pattern is exactly what the architecture forces.
2. **Baseline κ matches cycle 258**: bit-identical at the 5-digit precision recorded in cycle 258 manifest.
3. **Post-restore PPL = baseline PPL** (assert passes; no module leakage).
4. **L1 ranking sanity**: top swap is k=2 (1.78e-3), bottom is k=10 (1.04e-6) — 1700× dynamic range, clean ordering. The 5 most disruptive (k=2,3,1,4,0) are the 5 earliest non-trivial blocks, peaking at k=2, which is exactly where cycle 258's baseline κ_GELU peaks (κ at l=2, l=3 ≈ 2.23e-3).
5. **PPL deltas per swap range +0.022 (k=0) to +1.048 (k=4)** — modest, indicating the b_leak first-moment correction does its job locally; cycle 258's full all-12 swap pushed PPL from 45.06 to 60.08 (+15 PPL or +33%), so a single block swap should be roughly 15/12 ≈ 1.25 PPL on average if losses were additive — observed range [0.02, 1.05] is consistent (sub-additive for early blocks where the GELU gradient is most concentrated).

## Issues / caveats
1. **Diag/offdiag ratio is inf because of zero off-diagonal median**. The agenda's verdict criterion (max diag ≥ 3× off-diag median) is poorly defined for a strictly causal architecture where over half the off-diagonal is bit-zero. The substantive 5.4× ratio (vs nonzero offdiag mean) clears the 3× bar honestly. The trivial inf-pass should be flagged in any future re-run as a verdict-criterion bug, not a real result. The brief should have specified "median over the non-causal-zero entries". Recorded in the JSON's `verdict_logic.median_abs_delta_offdiag = 0.0` so it is auditable.
2. **Late-layer monotonicity criterion** (clause 2) is the half of the hypothesis that genuinely fails: the propagation pattern is *not* depth-monotone in k. The agenda hypothesis was "early swaps have more downstream amplification", but in fact swap_at_k=4 propagates more to layer 10 than swap_at_k=0 does — late-middle swaps have more downstream impact than the very earliest swap. This is the real, geometrically meaningful finding of the cycle.
3. **The κ ranking does NOT track the cycle 173 PPL ranking** (block 4 max PPL but rank 4 in κ; block 11 second-highest PPL but rank 11 in κ). This is the second meaningful finding: κ measures trajectory bending, PPL measures output distribution shift. They are separate properties of a swap. Cycle 173's finding that block 11 is the second-most damaging in PPL despite being the *least* disruptive in κ means late-block damage is from output-channel structure, not trajectory geometry.
4. **HF download progress bar clobbered the live tee output**, but the saved log file is clean (the tqdm carriage returns get overwritten in stdout but appended in the log file via `tee`). All numbers are reproducible from the log.

## Conclusion
- The script is correct (causal pattern, baseline match, restore round-trip).
- The verdict (INCONCLUSIVE) is the correct branch given the agenda criteria.
- The interesting *substantive* findings — (a) the κ load-bearing zone is at blocks 1-4 with k=2 dominant, (b) late-layer propagation is NOT monotone in k, (c) κ-ranking diverges from PPL-ranking — are recorded in the agenda PAST entry and the manifest.
- The verdict criterion bug (clause 1 trivially passes) is documented; the substantive 5.4× ratio also passes clause 1, so the verdict is robust to the bug.
- No follow-up recommended unless the user wants to test the hypothesis with the un-corrected ReLU (no b_leak) to see whether the b_leak smoothing kills downstream propagation, or to add a wave-equation analog using ‖Δx^l‖_G instead of κ_l.
