# Cycle 302 Plan — Item 312: Unified geometric curvature residual framework

## Item 312 (verbatim Hypothesis)
On `(GELU, ReLU, GPT-2 small, all blocks)` the framework reproduces (a) cycle 217 rank_95 ≤ 64 in 9/12 blocks within ±5%, (b) cycle 218 c_fc top-5 cos sum ≥ 2.0 in ≥6 blocks, (c) cycle 258 κ shift ≥ 35% in at least one of layers 3-5. Reproduction within rounding validates the framework before applying to other activation pairs.

## Dependencies
- `experiments/delta_per_token_cycle217.py` — raw Δy SVD / rank_95 reference: `[10,132,1,1,1,1,1,8,73,110,1,1]` (9/12 ≤ 64).
- `experiments/delta_centred_alignment_cycle218.py` — centred Δ̃, c_fc top-5 reader alignment reference: all 11 pairs ≥ 2.49.
- `experiments/bump_t1_features_cycle235.py` — top-k vocab decoding of right SVs via `wte @ v`.
- `experiments/pullback_geom_variants_cycle258.py` — κ with `aleph.pullback_geometry.{collect_residual_trajectory,compute_curvature,compute_salience,get_unembedding}`; layer-4 κ rel shift (ReLU+b_leak vs GELU) = 0.477 (index 3 of 11, reference).
- `experiments/curvature_residual_A_proj_cycle301.py` — A-projection panel, E_kept reference block 1 = 0.7463.

## Approach (no training)
1. New module `experiments/curvature_residual_geometric.py` providing a reusable `UnifiedCurvatureFramework` class parameterised by `(model, sigma, sigma_lin, blocks, calib_loader, A_factors=None)`. For each block `l`:
   - **Capture**: forward pre-hook on `model.transformer.h[l].mlp.c_fc` (HF Conv1D) to stash `ln_2(x)` over the calib corpus. Concat to `x_l ∈ R^(N, 768)`.
   - **Compute Δy**: chunked on CPU float32. `preact = x_l @ W_fc + b_fc ∈ R^(N, 3072)`, `delta = sigma(preact) - sigma_lin(preact)`, `Dy = delta @ W_proj ∈ R^(N, 768)`. Also the centred variant `Dy_c = Dy - Dy.mean(0)`.
   - **Panel 1 (cycle 217 rank_95)**: SVD of raw `Dy` → `rank_95`, `rank_99`, top-5 σ, top-3 right SVs.
   - **Panel 2 (cycle 218 c_fc top-5 cos sum)**: SVD of centred `Dy_c` → top-20 right SVs stacked as `(20, 768)`. For block `l+1`, compute `U_cfc = top_20 left SVs(W_fc_{l+1})`, then `M = V20 @ U_cfc` → `svdvals(M)`. Report top-5 cos sum. Also compute `Q`, `K`, `V` reader alignments from `h[l+1].attn.c_attn.weight` sliced `[0:768,768:1536,1536:2304]`.
   - **Panel 3 (cycle 235 vocab projection)**: for each of top-5 right SVs `v` of centred Δ̃, `logit_impact = wte @ v` → top-20 tokens by absolute impact (boost and suppress) via tokenizer.decode.
   - **Panel 4 (cycle 301 A-projection)**: if `A_factors[l]` provided, compute SVD `W_proj = U diag(S) Vh` and `E_kept = ||Dy @ Vh[:2,:].T||^2 / ||Dy||^2`. Otherwise skip.
   - **Panel 5 (cycle 258 κ shift)**: patched variant — install `sigma_lin + b_leak` per block (where `b_leak_l = E[(sigma − sigma_lin)(preact)]` over calib, 3072-dim). Re-run `collect_residual_trajectory` → `compute_curvature` on (unpatched, patched). Report per-layer `|kappa_A − kappa_B| / kappa_A`. Report max in layers 3–5 (κ indices 2–4, corresponding to layers 3–5 of 11).
2. New driver `experiments/unified_curvature_cycle302.py` instantiates the framework with `sigma=lambda z: F.gelu(z, approximate='tanh')`, `sigma_lin=F.relu`, `blocks=range(12)`, calib 200 wikitext-2 val × 128 tokens, seed 42 (matches 217/218/258/301).
3. **Validation metrics** (item 312 verbatim):
   - (a) `n_blocks_rank95_le_64 ≥ 9` (matches cycle 217 count) AND per-block rank_95 within ±5% of cycle 217 reference `[10,132,1,1,1,1,1,8,73,110,1,1]` — but the real guard is count.
   - (b) `n_pairs_cfc_top5_ge_2 ≥ 6` (cycle 218 reported 11/11).
   - (c) `max_{l∈{3,4,5}} rel_κ_shift ≥ 0.35` (cycle 258 reported 0.477 at layer 4).
4. **Verdict**:
   - **CONFIRMED** iff (a) AND (b) AND (c).
   - **REFUTED** iff any reproduction off by >2× (e.g. `n_blocks_rank95_le_64 < 5` or `max_kappa_shift < 0.18`).
   - **INCONCLUSIVE** otherwise.

## Files
- New: `experiments/curvature_residual_geometric.py`, `experiments/unified_curvature_cycle302.py`
- New: `autoresearch/results/cycle_302_unified.json`, `cycle_302_summary.md`, `cycle_302_manifest.txt`, `cycle_302_exec_result.txt`, `cycle_302_audit.txt`, `autoresearch/audits/cycle_302.md`, `autoresearch/logs/cycle_302_run.log`
- Updated: `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, `autoresearch/STATUS.md`

## Config
N_CALIB=200, SEQ_LEN=128, SEED=42, BLOCKS=range(12), DEVICE="mps" if available else "cpu", linalg on CPU float32. N_TOKENS=25,600. Trainable params = 0.

## Verification
- Float PPL within 0.5 of 53.895 (asserted).
- `W_fc.shape == (768, 3072)`, `W_proj.shape == (3072, 768)` per block.
- Panel 5 round-trip: post-restore PPL matches baseline within 1e-4.
- Per-block `rank_95_raw` compared against cycle 217 reference within ±5%.

## Baselines
- Cycle 217: `rank_95 = [10,132,1,1,1,1,1,8,73,110,1,1]`, 9/12 ≤ 64.
- Cycle 218: c_fc top-5 cos sum per pair = `[2.78,2.71,2.59,2.70,3.32,3.23,3.33,3.36,3.39,2.76,2.49]`, 11/11 ≥ 2.0.
- Cycle 258: `rel_kappa_shift_ReLU_vs_GELU = [0.118,0.144,0.357,0.477,0.408,0.261,0.133,0.037,0.028,0.118,0.314]` (κ indices 0..10 → layers 1..11); layer 4 (idx 3) = 0.477.
- Cycle 301: block 1 E_kept_top2 = 0.7463.

# APPROVE
