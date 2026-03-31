# Cycle 142 Plan: Correction Subspace Analysis (Item 154)

## Task
Characterize what LoRA corrections learn by decomposing the correction effect
into three orthogonal subspaces and measuring each component's contribution
to PPL recovery. This extends cycle 127's alignment analysis (which found
max cosine 0.46 but near-null overall) with a proper projection + ablation
framework.

## Hypothesis
Corrections primarily learn to compensate for topk-specific error patterns in
a "novel" subspace that is neither compression error nor attention amplification.
Evidence: (1) cycle 127 found near-null alignment with both reference subspaces,
(2) cycle 138 showed corrections trained on topk error actively worsen AE-based
compression (76.7 vs 74.4). The novel component should account for >=60% of
PPL recovery. If so, corrections are not "fixing error" — they are rerouting
the residual stream around the damaged subspace.

## Pre-flight
- P1 (duplicate): cycle 127 did alignment. This extends to projection + ablation — distinct.
- P2 (deps): no hard dependencies. Cycle 127 code patterns reused.
- P3 (scope): 3 seeds x [train corrections + extract + project + ablate] ~60 min total CPU.
- P4 (null baseline): corrected PPL ~69.4 (cycle 127). Uncorrected ~95.9. Float ~57.65.
  Abort if float PPL > 60 or corrected PPL > 75 (correction training failed).
- P5 (param accounting): LoRA r=8: 2 x 768 x 8 = 12,288 params per block.
  4 blocks = ~49K total. Print at init.

## Methods

### Step 1: Train LoRA corrections (3 seeds)
Reuse P1Model and LowRankCorrection from `experiments/svd_alignment_cycle127.py`.
Same protocol: topk k=4 compression on blocks [1, 4, 5, 7], lstsq + 500 Adam
polish, then 2000-step LoRA training (r=8, lr=1e-4, CE loss). Seeds [42, 137, 7].

Validation: corrected PPL must be within [60, 75] for each seed. If any seed
falls outside, extend training to 3000 steps or adjust lr.

### Step 2: Extract correction matrices
For each seed and block b: extract U_b (768x8) and V_b (8x768). Compute
effective correction C_b = U_b @ V_b (768x768, rank 8).

Also compute data-weighted correction: collect activations X_holdout (N x 768)
on 200 holdout sequences, compute C_b @ X_holdout^T. SVD of this gives the
directions actually used on real data (per cycle 127 reviewer suggestion).

### Step 3: Compute reference subspaces
For each block b:
- **Error subspace E_b**: run float and compressed model on holdout data, collect
  residual-stream difference at block b output (post-FFN, pre-LayerNorm of next
  block — per cycle 127 review fix). Compute top-8 PCs of the error matrix.
- **Attention amplification subspace A_b**: collect residual-stream error entering
  block b+1, then collect residual-stream error exiting block b+1's attention
  (after attn but before FFN). The difference isolates attention's amplification.
  Top-8 PCs.
- **Novel complement N_b**: the orthogonal complement of span(E_b) + span(A_b)
  within the rank-8 correction subspace. Computed via: project C_b's SVD vectors
  onto span(E_b, A_b), subtract, and take what remains.

### Step 4: Projection decomposition
For each block b, decompose C_b into three components:
1. C_error = proj_{E_b}(C_b) — projection of correction onto error subspace
2. C_attn = proj_{A_b}(C_b - C_error) — projection of residual onto attn subspace
   (orthogonalize against E_b first to avoid double-counting)
3. C_novel = C_b - C_error - C_attn — the remainder

Record Frobenius norm fraction for each component: ||C_x||_F / ||C_b||_F.

### Step 5: Ablation
For each block b and component, construct ablated correction:
- Zero-error: C_b - C_error (remove error component, keep attn + novel)
- Zero-attn: C_b - C_attn (remove attn component, keep error + novel)
- Zero-novel: C_b - C_novel (remove novel component, keep error + attn)
- Each-only: C_error, C_attn, C_novel separately

Replace the per-block LoRA matrices with rank-8 SVD approximation of the
ablated C_b (reconstruct U_ablated, V_ablated from SVD). Evaluate full-model
PPL for each ablation. This gives 6 ablation conditions per block (3 zero-out,
3 each-only), but run the critical ones first: the 3 each-only conditions tell
us which component is sufficient.

To keep computational cost manageable: run per-block ablation (ablate one block
at a time while others retain full correction). This isolates each block's
subspace contribution. Do NOT ablate all blocks simultaneously — that confounds.

### Step 6: Cross-block sharing test
Take block 4's trained correction and apply it to block 7 (and vice versa).
Measure PPL. If PPL recovers (close to the block's own correction), corrections
are generic. If PPL is worse than uncorrected, corrections are block-specific.

### Step 7: Record and visualize
- Per-block Frobenius norm fractions (stacked bar chart).
- Per-block ablation PPL table (seeds x blocks x 6 conditions).
- Cross-block sharing PPL.
- Summary: which subspace dominates PPL recovery?

## Steps (implementation)
1. Create `experiments/correction_subspace_cycle142.py`.
2. Load GPT-2 + OWT corpus. Verify float PPL ~57.65.
3. For each seed in [42, 137, 7]:
   a. Compress blocks [1,4,5,7] via topk k=4 + lstsq + 500 polish.
   b. Train LoRA corrections (2000 steps). Record corrected PPL.
   c. Extract C_b = U_b @ V_b for each block.
   d. Compute data-weighted SVD of C_b @ X_holdout^T.
   e. Compute error subspace E_b and attention subspace A_b from holdout.
   f. Project + decompose (step 4).
   g. Run ablation conditions (step 5). Record PPL for each.
   h. Cross-block sharing test (step 6). Record PPL.
4. Aggregate across seeds. Compute means/stds.
5. Save JSON + plots.

## Validation
- Float PPL: 57.65 +/- 1. Abort if violated.
- Corrected PPL (per seed): must be in [60, 75]. Cycle 127 got 69.4.
- Ablation sanity: C_error + C_attn + C_novel = C_b (check Frobenius norm,
  residual should be < 1e-6). If not, projection is wrong.
- Cross-block: uncorrected PPL ~95.9. Wrong-block correction should be > uncorrected
  or at least much worse than own-block correction.

## Criteria (verbatim from agenda)
- **Confirmed** if a single identifiable subspace (error, attention, or novel)
  accounts for >=60% of correction's PPL recovery when ablated.
- **Refuted** if all three subspaces contribute roughly equally (diffuse).
- **Inconclusive** if dominant subspace varies by block or seed.

## Artifacts
- `experiments/correction_subspace_cycle142.py`
- `autoresearch/results/cycle_142_subspace_decomposition.json`
- `autoresearch/results/cycle_142_norm_fractions.png`
- `autoresearch/results/cycle_142_ablation_table.png`

## Fallbacks
- If correction training diverges for a seed: extend to 3000 steps, reduce lr
  to 5e-5. If still fails, drop that seed and note it.
- If error/attention subspaces are nearly collinear (angle < 15 deg), merge them
  into a single "known error" subspace and compare "known" vs "novel" (2-way
  instead of 3-way). This simplifies the story without losing the core question.
- If per-block ablation is too slow (>2h total): skip each-only conditions,
  run only the 3 zero-out conditions. The zero-novel condition is the most
  informative (does removing the novel component kill correction?).

## Wall time estimate
- 3 seeds x [compression ~5 min + correction training ~15 min + analysis ~3 min] = ~70 min.
- Ablation: 3 seeds x 4 blocks x 6 conditions x ~30 sec eval = ~36 min.
- Cross-block: 3 seeds x 2 swaps x ~30 sec = ~3 min.
- Total: ~110 min. Within 2h budget.
