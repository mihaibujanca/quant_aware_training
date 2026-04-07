# Cycle 258 Plan — Item 269
## Item (verbatim)
269 (P0): Pullback curvature & salience — GELU vs ReLU+bias trajectories. For (a) baseline GELU, (b) ReLU+precomputed b_leak (cycle 181 patch), (c) gelu_noneg+μ (cycle 228 patch): collect residual trajectory via aleph.pullback_geometry, compute κ_l and S_l per layer, plot, report ρ(S, κ) per variant.
## Hypothesis
ReLU+bias drops κ_l in blocks 0–3 by ≥30% relative to GELU baseline AND ρ_ReLU(S, κ) weakens to |ρ| ≤ 0.5; gelu_noneg+μ sits between the two.
## Baselines
Cycle 252 GELU κ profile (ρ = −0.745). Cycle 181 ReLU+bias all-12 PPL +88% (42.6). Cycle 228 gelu_noneg+μ PPL +37% (61.66). Cycle 189 cos(e_l, e_{l−1}) = 0.72.
## Setup
Create `experiments/pullback_geom_variants_cycle258.py`. Reuse cycle 220/228 data loader (200 wikitext-2 val seqs × 128 tokens, seed 42). Build 3 variants on the same `gpt2` model:
- **A — GELU baseline**: untouched HF gpt2.
- **B — ReLU + per-block b_leak**: replace each `mlp.act` with a `ReluPlusBleak(b_leak_l)` module returning `relu(z) + b_leak_l`, where `b_leak_l ∈ R^3072` is precomputed on the calib slice as `mean_x[gelu(z) - relu(z)] = mean_x[gelu(z) ⊙ 1[z<=0]]` (per-channel intermediate-space first-moment correction; same calib pass as cycle 228 but RELU baseline instead of relu(gelu)).
- **C — gelu_noneg + per-channel μ**: cycle 228's `GeluNonegCorrected` patched on each block, μ_l = mean_x[gelu(z) - relu(gelu(z))] per channel.
For each variant: `collect_residual_trajectory(model, calib_ids)` over the full 200 seqs (concatenated to ~25.6k tokens). `compute_salience(reduce='none')` and `compute_curvature(reduce='none')` with `U = get_unembedding(model)`. Aggregate per-layer mean / p10 / p90 over (B, T). Compute Pearson ρ(S_l, κ_l) per variant using overlapping layer indices (κ has L−1=11 entries on layers 1..11, S has L=12; correlate S[1:12] vs κ).
## Verdict
**CONFIRMED** if `max_l |κ_GELU − κ_ReLU| / κ_GELU ≥ 0.30` for some `l ∈ 0..6` AND `|ρ_ReLU(S, κ)| ≤ 0.5`.
**REFUTED** if all per-layer relative κ differences `< 0.10` AND `ρ_ReLU ∈ [−0.85, −0.65]`.
**INCONCLUSIVE** otherwise.
## Execute
`cd /Users/mihai/Projects/quant_aware_training && MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/pullback_geom_variants_cycle258.py 2>&1 | tee autoresearch/logs/cycle_258_run.log`
Output JSON: `autoresearch/results/cycle_258_pullback_variants.json` (per-variant κ_l, S_l mean/p10/p90, ρ, config). Plot: `autoresearch/results/cycle_258_kappa_salience.png` (κ profiles in left panel, S profiles in right panel, 3 variants overlaid). Summary md: `autoresearch/results/cycle_258_summary.md`.
## Bookkeeping
Stage cycle 258 files only (NOT pre-existing dirty files). Commit `[autoresearch] cycle 258: pullback κ/S GELU vs ReLU vs gelu_noneg — <verdict>`. Update agenda.md (delete item 269 from NEXT/P0; prepend `### Cycle 258` PAST entry). Append TSV row. Manifest, bookkeeping commit, exec_result.
## Risks
Memory: collecting 200×128 tokens × 13 trajectories × 768 dims × float32 = ~10 MB per variant — fine on MPS. Curvature has L−1=11 layers (curvature defined on l=1..11); salience on l=0..11 (12 layers). For ρ correlate the overlapping range. b_leak per channel sign sanity: should be ≥0 elementwise (relu zeroes negatives that gelu kept slightly negative AND zeroes the negative tail of gelu in the positive range — net `gelu(z) - relu(z)` is positive on z<0 by ~0.17 and small/zero on z>0). Run cap ~5 min on MPS.
