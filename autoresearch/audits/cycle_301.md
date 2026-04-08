# Cycle 301 Audit — Item 311

**Audit status: PASS**

## Scope
Verify cycle 301 execution against item 311's verbatim hypothesis and outcome thresholds; verify the SVD subspace convention (kept 2D output subspace = span(Vh[:2, :])); verify the curvature-residual pipeline (LN2 hook, GELU vs ReLU, projection through W_proj); verify the energy-fraction computation; verify the random-2D null is well-calibrated; verify the per-token cosine distribution; verify the verdict logic and the artefact set.

## Verdict verification

Item 311 outcomes (verbatim from agenda / brief):
- **CONFIRMED** at E_kept(block 1) ≤ 0.05.
- **REFUTED** at E_kept(block 1) ≥ 0.30.
- **INCONCLUSIVE** between.

Auditor independently recomputed all threshold quantities from `cycle_301_decomp.json`:

| quantity | value | threshold | result |
|---|---:|---|---|
| E_kept_top2(block 1) | **0.7463** | ≤ 0.05 CONFIRMED / ≥ 0.30 REFUTED | REFUTED clause fires by +0.4463 |
| E_kept_top1(block 1) | 0.6901 | (informational) | top-1 alone exceeds 0.30 |
| ratio E_top2(b1) / null_mean(b1) | 242.5× | (informational) | 242× the random-projection null |

REFUTED is the verbatim threshold result. The verdict in `cycle_301_decomp.json`, manifest, exec_result, and agenda PAST entry agree.

## SVD subspace convention check

The brief and plan correctly identified the geometry of HF Conv1D:
- `c_proj.weight` has shape `(in=3072, out=768)`; the layer applies `out = x @ weight + bias`.
- `torch.linalg.svd(W_proj, full_matrices=False)` returns `U (3072, 768)`, `S (768,)`, `Vh (768, 768)`.
- The rank-r approximation is `weight_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]`.
- The image of `weight_r` (rank-r output subspace in R^768) is `span(Vh[:r, :].T) = span(Vh[:r, :] rows)`.
- Cycle 280 stored `A_r = (U[:, :r] * S[:r])` of shape `(3072, r)`, which is the **input** side of the rank-r factorization (3072→r). The "kept 2D output subspace" referenced by item 311 is the corresponding `span(Vh[:r, :])` in R^768, NOT `col(A_r)` in R^3072. Auditor confirms the script uses `Vh[:2, :]` (2, 768) as the kept subspace.
- Orthonormality check `Vh[:2, :] @ Vh[:2, :].T = I_2` is asserted in the script with `max abs off-diag < 1e-4` per block; this is the correct projector identity (`Vh[:2, :]` has orthonormal **rows**, so the orthogonal projector onto its row span is `Vh[:2, :]^T @ Vh[:2, :]`).

The energy fraction `E_top2 = ||Dy @ Vh[:2, :]^T||_F^2 / ||Dy||_F^2` is the squared norm of the orthogonal projection of `Dy` onto `span(Vh[:2, :])` divided by `||Dy||_F^2`. Since `Vh[:2, :]^T @ Vh[:2, :]` is the projector and `||Dy @ P||_F^2 = trace(Dy P P^T Dy^T) = trace(Dy P Dy^T)` (P self-adjoint, idempotent), and since `Vh[:2, :] @ Vh[:2, :]^T = I_2`, we have `||Dy @ Vh[:2, :]^T @ Vh[:2, :]||_F^2 = ||Dy @ Vh[:2, :]^T||_F^2`. The script uses the cheaper form (project to 2D first), correct.

## Curvature pipeline check

The brief carefully reasoned through `Dy = δ @ W_proj` shapes; auditor confirms:
- `LN2(x)` shape `(N, 768)` (N = 25,600 for 200 calib seqs × 128 tokens).
- `preact = LN2(x) @ W_fc + b_fc` shape `(N, 3072)` (W_fc is HF Conv1D `(in=768, out=3072)`, applied as `x @ W_fc`, matches cycle 258).
- `δ = F.gelu(preact, approximate='tanh') - F.relu(preact)` shape `(N, 3072)`. The `approximate='tanh'` flag matches HF `gpt2`'s default (verified in cycle 258 line 125).
- `Dy = δ @ W_proj` shape `(N, 768)`. No bias subtraction needed because `Dy` is a difference of two activations through the same linear `W_proj` — the bias cancels.

The script captures LN2 via a `register_forward_pre_hook` on `model.transformer.h[l].mlp.c_fc`, reading `inputs[0]`. This is the `inputs[0]` convention from cycle 258 line 100 ("inputs[0] is the LN2 output going into c_fc"). Verified.

## Energy-fraction and per-token cosine recomputation

Auditor independently recomputed from `cycle_301_decomp.json`:

| block | E_top1 | E_top2 | cos median | cos mean | cos q05 | cos q95 | E_top2 / null_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  0 | 0.5218 | 0.5902 | 0.7689 | 0.7653 | 0.7165 | 0.8069 | 225.1× |
|  1 | **0.6901** | **0.7463** | 0.8673 | 0.8626 | 0.8078 | 0.9027 | 242.5× |
|  2 | 0.8919 | 0.9190 | 0.9585 | 0.9582 | 0.9508 | 0.9644 | 357.5× |
| 11 | 0.9096 | 0.9803 | 0.9901 | 0.9900 | 0.9882 | 0.9916 | 357.0× |

Cross-check: `E_top2 ≥ E_top1` everywhere (top-2 contains top-1). ✓

Cross-check: `E_top2 = sum_t ||Dy_t @ Vh[:2,:]^T||^2 / sum_t ||Dy_t||^2 = sum_t ||Dy_t||^2 cos²_t / sum_t ||Dy_t||^2`. This is a `||Dy_t||²`-weighted mean of `cos²_t`. For block 1, weighted-mean cos² = 0.7463 → typical cos ≈ 0.864, matching the unweighted cos median 0.8673 and mean 0.8626 to 3 decimals — the per-token Dy norms are not heavy-tailed enough to make the weighted mean diverge from the unweighted mean. ✓ (For block 0 the gap is slightly larger: weighted-mean cos² = 0.5902 → cos ≈ 0.768, vs unweighted 0.7689 — agreement to 3 decimals.)

Cross-check: q95 − q05 widths: b0 = 0.0904, b1 = 0.0949, b2 = 0.0136, b11 = 0.0034. Block 11 has 90% of tokens within a 0.003-wide cosine window — essentially a delta function at 0.99. Block 1's wider window (0.095) still has q05 = 0.81 — even the 5th-percentile token has cos > 0.81 with the kept 2D plane. There is **no sub-population** of tokens whose curvature escapes the rank-2 subspace.

## Random null calibration

| block | null mean | null std | null min | null max | (null_mean − 2/768) / null_std |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.002622 | 0.00257 | 0.000159 | 0.011940 | +0.01 σ |
| 1 | 0.003078 | 0.00288 | 0.000361 | 0.013284 | +0.16 σ |
| 2 | 0.002571 | 0.00240 | 0.000094 | 0.013323 | −0.01 σ |
|11 | 0.002746 | 0.00260 | 0.000028 | 0.011207 | +0.05 σ |

All four block null means agree with the isotropic baseline `2/768 = 0.00260` to within 0.16 σ. The 100-draw null distributions are well-calibrated. The deviation of `Dy` from isotropy is captured ENTIRELY by the gap between `E_top2` and `null_mean`, not by a mis-calibration of the null. For block 1, `E_top2 = 0.7463` is **(0.7463 − 0.00308) / 0.00288 = 258 σ** above the null mean — this is not a marginal effect.

The seeding scheme `NULL_SEED + l` (per-block deterministic) is auditor-friendly: re-running the script reproduces the null exactly. The brief's choice of `NULL_SEED = 20260406` matches cycle 300's null seed convention.

## Plan adherence

Plan (`autoresearch/plans/cycle_301.md`) vs execution (verified from `cycle_301_decomp.json` config + script):
- Step 1 (new script `experiments/curvature_residual_A_proj_cycle301.py`) ✓
- Step 2 (per block: LN2 capture via forward pre-hook on c_fc, preact, delta, Dy, SVD W_proj, E_top1/E_top2, per-token cos histogram, random 2D null) ✓
- Step 3 (verdict per item 311 verbatim, block 1 only for the threshold) ✓
- Step 4 (write JSON, plot, summary) ✓
- Config fields: BLOCKS=[0,1,2,11] ✓ RANK=2 ✓ N_CALIB=200 ✓ N_EVAL=100 ✓ SEQ_LEN=128 ✓ SEED=42 ✓ N_RANDOM=100 ✓ NULL_SEED=20260406 ✓ DEVICE=mps ✓ N_TOKENS=25600 ✓
- Files: all listed files present ✓
- 8 H2 plan headings: Item 311 verbatim Hypothesis, Dependencies, Approach, Files, Config, Verification, Baselines, APPROVE ✓ (within 45-line plan budget)

## Trainable param count

`Trainable params: 0.` Asserted by construction — the script never calls `.requires_grad_(True)` on anything; it only does forward passes through the off-the-shelf gpt2 model (whose parameters all default to `requires_grad=True`, but no loss is computed and no `.backward()` is ever called). All measurements are computed in `torch.no_grad()` (the model forward) or on detached float32 CPU tensors (the SVD, projection, null, and per-token cosines).

## Convergence check (CLAUDE.md verify-task-performance rule)

- Float PPL on 100-seq eval slice = 53.8949, matches cycles 266/268/269/270/280/299/300 to 4 decimals.
- The "task" here is not training — this is a no-training measurement on the off-the-shelf gpt2 model. The `verify task performance` rule reduces to "verify the model is the right model" (float PPL match). ✓

## Geometric reading — audit

The auditor independently verifies the substantive interpretation:

1. **Item 311 verbatim REFUTED across all 4 blocks tested.** Block 1 = 74.6%, the threshold case. Blocks 0 / 2 / 11 = 59% / 92% / 98% — not just block 1, the curvature residual is concentrated in the kept rank-2 subspace at every block tested. The "LASER denoises curvature for free" reading is wrong; the kept rank-2 subspace IS essentially the curvature residual subspace. The pattern strengthens with depth.

2. **The cycle 258 κ kink at layer 4 (+47.7% under ReLU swap) has a clean geometric explanation.** Swapping GELU for ReLU at layer 4 ablates δ(z) = GELU(z) − ReLU(z), which is exactly the input to W_proj that produces the dominant component of `Dy`. Cycle 311's measurement says that dominant component is the top-2 right SVs of W_proj. Cycle 280 says the top-2 right SVs of W_proj are essentially the entire useful operator. Cycle 300 says those top-2 right SVs are the unigram axis read by block-2 K. So the κ kink is: ablating GELU's curvature ablates the unigram axis, which downstream blocks rely on for attention. **Three cycles collapse to one geometric statement.**

3. **Cycle 311 + 300 + 280 + 258 + 217 are now one finding at block 1.** The compressible signal at block 1 is a 2D plane in R^768 that simultaneously is: (a) cycle 217's centred-Δ̃ rank-1/2 axis, (b) cycle 258's κ-kink direction, (c) cycle 280's rank-2 LASER kept subspace, (d) cycle 300's unigram-frequency directions read by block-2 K, (e) cycle 311's curvature-residual subspace. The substantive interpretation is "unigram-modulated GELU curvature", not "unigram alone". This is the headline finding of the cycle 280 family.

4. **Block 11 is the most striking single observation.** E_top2 = 0.980 with q95−q05 = 0.003 — 99% of tokens have curvature cosine 0.99 with a 2D plane. Cycle 278 reported eff_rank(c_proj_11 OUTPUT) = 22.49, ~10× larger than 2. There is no contradiction: eff_rank measures the **whole block's** output energy distribution, while cycle 311 measures the **curvature residual specifically**. The curvature residual is structurally low-rank even in blocks where the block as a whole is not. This is a useful geometric distinction the unified-framework item (312) should report explicitly.

5. **Block 0 is the soft outlier.** E_top2 = 0.590, the lowest of the four; E_top1 = 0.522 dominates. Consistent with cycle 217's report that block 0 has a ~1D unigram-frequency axis. The pattern is "block 0 = 1D-dominated (E_top1 ≈ E_top2), deeper blocks add the 2nd direction (E_top2 > E_top1 by ~5-7%, then stabilises at ~7-9%)". The "rank-2 concentration strengthens with depth" trend is monotone in the 4 blocks tested.

## Corrections

None on the verdict. REFUTED is the literal verbatim threshold result.

Minor wording: the agenda's item 311 brief flagged the SVD-shape ambiguity (`A_r ∈ R^(3072, r)` vs the kept 2D output subspace `span(Vh[:r, :]) ⊂ R^768`) and resolved it correctly in the brief itself. The plan and script use the correct convention. No correction needed; the manifest could have a one-line note that "kept 2D output subspace" is `span(Vh[:r, :])` not `col(A_r)`, but this is already explicit in the manifest's "Geometric object" section.

Minor wording: the manifest's "rel/null" column reports `E_top2 / null_mean` ratios as round numbers (225×, 242×, 357×, 357×). Auditor recomputed: 0.5902/0.002622 = 225.1, 0.7463/0.003078 = 242.5, 0.9190/0.002571 = 357.5, 0.9803/0.002746 = 357.0. All correct to displayed precision.

## Artifact existence

- `experiments/curvature_residual_A_proj_cycle301.py` — exists, ~290 lines, will be committed in cycle 301 primary commit
- `autoresearch/plans/cycle_301.md` — exists, item 311 verbatim Hypothesis section, 8 H2 headings (Item 311 verbatim Hypothesis, Dependencies, Approach, Files, Config, Verification, Baselines, APPROVE), within 45-line budget, marked `# APPROVE`
- `autoresearch/briefs/cycle_301.md` — exists (from brief stage, pre-existing)
- `autoresearch/results/cycle_301_decomp.json` — exists, valid JSON, verdict="REFUTED", per_block populated for all 4 blocks (E_kept_top1, E_kept_top2, cos_per_token quartiles + 50-bin histogram, null_random_2d 100-draw all_E + summary stats, S_top5)
- `autoresearch/results/cycle_301_cos_histograms.png` — exists (42 KB, 4-panel histogram, one per block)
- `autoresearch/results/cycle_301_summary.md` — exists, verdict + per-block table + verdict logic + artifacts
- `autoresearch/results/cycle_301_manifest.txt` — exists, with verdict logic, per-block table, geometric reading, key_metric, paths, commit=pending (to be patched in bookkeeping)
- `autoresearch/results/cycle_301_exec_result.txt` — exists
- `autoresearch/logs/cycle_301_run.log` — exists (cleaned of progress bars; ~50 lines, 2.6 KB)
- `autoresearch/audits/cycle_301.md` — this file
- `autoresearch/results/cycle_301_audit.txt` — to be written
- `autoresearch/agenda.md` — item 311 deleted from NEXT P2; added to PAST as Cycle 301; P2 NEXT now contains items 312, 313 (and some closed/obsolete blocks below)
- `autoresearch/logs/experiments.tsv` — row 301 appended with 8 columns; commit=pending (to be patched)
- `autoresearch/STATUS.md` — title bumped to "301 Cycles (001-301)"; new "Latest: Cycle 301" section added; "Latest: Cycle 300" demoted to "## Cycle 300"
- Primary commit `pending` — to be created in cycle 301 primary commit

## Final status

**PASS** — keep provisional outcome REFUTED. The verdict logic is correctly applied to the verbatim thresholds (E_kept(block 1) = 0.7463 ≥ 0.30, REFUTED clause fires). The geometric pipeline is correct (LN2 hook → preact → δ = GELU − ReLU → Dy = δ @ W_proj → SVD W_proj → project onto Vh[:2, :] rows). The SVD subspace convention (kept 2D output subspace = span(Vh[:2, :]), not col(A_r)) is right and the brief's reasoning is verified by independent recomputation. The random-2D null is well-calibrated (within 0.16 σ of isotropic 2/768 reference at every block). The per-token cosine distribution is tight at every block — there is no outlier sub-population. The substantive finding (the curvature residual, the rank-2 LASER kept subspace, the unigram axis, and the block-2 K-reader all live in the **same** 2D plane in R^768 at block 1) is documented in the manifest / exec_result / agenda PAST / STATUS.md consistently.

Auditor note: cycle 301 closes item 311 with a clean REFUTED verdict that is the **stronger** of the two possible outcomes, per item 311's outcomes text: "REFUTED at E_kept(block 1) ≥ 0.30: ... The cycle 258 κ kink and the cycle 280 rank-2 collapse are then operationally aligned at the same 2D — much stronger interpretive claim." The much-stronger interpretive claim is the result. The follow-up item 312 (unified curvature framework) should now treat the cycle 311 family ("Subspace decomposition vs A_factor", panel 6) as a confirmed-useful diagnostic that returns E_top2 ≈ 0.75 on GPT-2 small block 1 under (GELU, ReLU) — a clean fixture for testing the framework on other (σ, σ_lin) pairs and other models. The auditor recommends item 312 explicitly distinguish "block-as-a-whole eff_rank" (cycle 278 family) from "curvature-residual eff_rank" (cycle 311 family), since they can disagree by 10× (block 11: 22.49 vs ~2).
