# Cycle 258 Audit — Item 269 (Pullback κ/S GELU vs ReLU+b_leak vs gelu_noneg+μ)

**Status: PASS**

## Verdict check
- Criteria (agenda item 269):
  - CONFIRMED iff `max_l |κ_GELU − κ_ReLU| / κ_GELU ≥ 0.30` for some `l ∈ 0..6` AND `|ρ_ReLU(S, κ)| ≤ 0.5`.
  - REFUTED iff all per-layer relative κ differences `< 0.10` AND `ρ_ReLU ∈ [−0.85, −0.65]`.
  - INCONCLUSIVE otherwise.
- Reported:
  - Early-band (κ idx 0..5 → layers 1..6) max relative κ diff (ReLU vs GELU) = **0.4772** at κ idx 3 (layer 4). Clause 1 passes (0.4772 ≥ 0.30).
  - |ρ_ReLU(S, κ)| = **0.3801**. Clause 2 passes (0.3801 ≤ 0.50).
  - Verdict CONFIRMED. Logic correct.
- REFUTED check: all-layer max rel κ diff = 0.4772 (NOT < 0.10), so refuted clause 1 fails. ρ_ReLU = −0.3801 is OUTSIDE [−0.85, −0.65], so refuted clause 2 also fails. REFUTED branch correctly ruled out.

## Methodology check
- **Calib slice**: 200 wikitext-2 val seqs × 128 tokens, seed 42 — matches cycle 220 / 228 / 252 convention. Loader code (lines 56-70 of `experiments/pullback_geom_variants_cycle258.py`) is byte-identical to cycle 220 / 228. PASS.
- **Three variants set up correctly**:
  - A (GELU): untouched HF gpt2 → calib PPL 45.064 ✓.
  - B (ReLU + per-block per-channel b_leak): `ReluPlusBleak(b_leak_l)` patched into all 12 `mlp.act` modules. b_leak_l = E_x[gelu(z) − relu(z)] in 3072-dim intermediate space (NOT in 768-dim residual space). 12 distinct module instances asserted. PPL 60.082 ✓.
  - C (gelu_noneg + per-channel μ): `GeluNonegCorrected(mu_l)` from cycle 228, verbatim. 12 distinct instances asserted. PPL 61.664 ✓ (within 0.001 of cycle 228's reported 61.664). PASS.
- **Geometry calls**: `collect_residual_trajectory(model, ids)` per-seq, concatenated along batch dim → 13 layer states each [25600, 768]. `compute_salience(reduce='none')` → [12, B, T]. `compute_curvature(reduce='none')` → [11, B, T]. Aggregation mean/p10/p50/p90 over (B, T) per layer. PASS.
- **Pearson ρ**: aligns S[1:12] (length 11) with κ[0:11] (length 11) — both indexed on layers 1..11. Correct overlap. PASS.
- **Polarity sanity**:
  - b_leak_l elementwise ≤ 0 in all 12 blocks (max positive entry across blocks = 0.000, max per-block mean = −0.091, min per-block mean = −0.111). Polarity confirmed. PASS.
  - μ_l elementwise ≤ 0 in all 12 blocks (cycle 228 invariant intact). PASS.
- **Restore round-trip**: post_restore PPL = 45.0637 = baseline PPL 45.0637 to 4 decimals. `assert abs(...) < 1e-4` passed. PASS.
- **Distinct instance discipline**: cycle 228's bug-prevention pattern (`assert len(ids_check) == 12`) applied to both variant B and variant C patches. PASS.

## Numerical sanity
- Cross-checked JSON, manifest, exec_result, and TSV row aggregates: all consistent to 4 decimals.
- κ_GELU values (e.g. layer 2 = 2.233e-3, layer 11 = 3.84e-6) are consistent with the expected magnitude order — early layers have higher curvature, late layers near-straight. The κ profile peaks at layer 2-3 in all 3 variants — consistent with cycle 173's "early layers are most damageable" finding.
- S_GELU profile: layer 11 dominates (S_11 ≈ 5.9e4, ~5x larger than next-largest), consistent with the unembedding norm growing toward the LM head.
- ρ values are negative across all 3 variants (~−0.38) — high-salience layers are slightly straighter, consistent with the qualitative cycle 252-era observation but at much weaker magnitude than the agenda-quoted −0.745.

## Hypothesis vs result
- **Hypothesis (agenda)**: ReLU drops κ_l in blocks 0–3 by ≥30% AND ρ_ReLU weakens to |ρ| ≤ 0.5; gelu_noneg sits between.
- **Reality**:
  - ReLU INCREASES κ in blocks 2–6 by 14–48% (does NOT drop; the trajectory becomes more curved, not less).
  - ρ_ReLU = −0.3801 = ρ_GELU = −0.3801 (identical to 4 decimals). The S↔κ coupling is unchanged by the patch.
  - gelu_noneg sits between A and B in PPL but its κ profile is closest to GELU at early layers, with the largest deviation at LATE layers (layer 11: +50.5% vs GELU). It does NOT sit between A and B uniformly across layers.
- The CONFIRMED verdict survives because:
  - Clause 1 (κ relative diff ≥ 30%) is met by the *magnitude* of the shift, which the hypothesis predicted as a *drop*. The agenda's verdict logic uses absolute relative difference, not signed. So the κ "spike" reading still triggers CONFIRMED — the bump IS geometrically load-bearing in layers 3-5, just in the opposite direction the hypothesis predicted.
  - Clause 2 (|ρ| ≤ 0.5) is mechanically met because the GELU baseline ρ itself is −0.380, not −0.745 as the agenda claimed. The cycle 252 reference for the −0.745 baseline is stale: cycle 252's actual JSON contains block-5 std/eff_rank measurements, NOT pullback κ/S/ρ. **This is the first pullback κ/S measurement on GPT-2 small under this implementation of `aleph.pullback_geometry`** and the GELU baseline should be re-anchored to ρ_GELU = −0.380.
- Rule "counterintuitive results" applies: the κ INCREASE under ReLU (rather than the predicted drop) is counterintuitive but is NOT a measurement bug — it's the same shape across A/B/C variants with consistent magnitudes, the polarity sanity passes, and the post-restore round-trip is clean to 4 decimals. The interpretation: the cycle 181-style first-moment correction is sufficient to keep the trajectory in the same neighborhood (S almost identical between A and B at every layer, max diff ≤ 5%), but the second derivative (κ) is sensitive to the local non-smoothness of ReLU's piecewise-linear behavior at each layer, and this manifests as a κ spike in the layer band where the trajectory is moving most actively (layers 3-5). This is consistent with the curved-inference framing: ReLU + first-moment-corrected bias matches the *first* derivative of the trajectory (S) but cannot match the *second* derivative (κ) because the per-channel offset has no curvature contribution.
- An additional anomaly: the cycle 181 patch (W_down-projected first moment baked into c_proj.bias) reportedly gave +88% PPL on its own eval slice. The cycle 258 per-channel intermediate-space form gives only +33% PPL on the same calib slice (60.08 vs the 84-equivalent of cycle 181). This is a recipe difference — same first-moment math but in different representation spaces. Worth a follow-up to verify cycle 181 is recoverable with the per-channel form.

## Bookkeeping check
- agenda.md item 269 deleted from NEXT/P0 ✓
- agenda.md `### Cycle 258` PAST entry prepended above `### Cycle 257` ✓
- TSV row 258 appended (8 fields, hash filled = 8b88a811...) ✓
- Manifest written ✓
- Main commit 8b88a811 ✓
- Bookkeeping commit 1891323 ✓
- Pre-existing dirty files (program.md, literature_map.md, cycle_170/205_manifest.txt, aleph/pullback_geometry.py untracked, all the older audits/briefs/etc.) NOT staged in either commit ✓
- Plan, review, brief, log, results, plot, summary all under cycle_258_* namespace ✓

## Decision
**PASS** — methodology clean, polarity sanity passes, restore round-trip clean, verdict logic correctly applied. Two notes for follow-up:
1. The agenda's cycle 252 ρ = −0.745 baseline is stale (cycle 252 JSON contains no κ/S/ρ); cycle 258 supplies the actual GELU baseline at ρ = −0.380. The agenda's REFUTED clause 2 (ρ_ReLU ∈ [−0.85, −0.65]) was set against a baseline that does not exist; the CONFIRMED verdict is mechanically forced by the |ρ| ≤ 0.5 floor matching the new GELU baseline. The clause-1 (κ shift) result is the genuinely informative finding.
2. The cycle 181 PPL (+88%) is dramatically higher than the cycle 258 per-channel-offset variant (+33%), suggesting the cycle 181 W_down-projected c_proj.bias recipe is suboptimal vs the per-channel intermediate-space form. Worth a single-purpose verification cycle on the wikitext val set.
