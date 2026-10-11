# Cycle 261 Audit — Item 272 (W_down column geometry vs per-channel sign frequency σ_c)

**Status: PASS**

## Verdict check
- Criteria (agenda item 272):
  - CONFIRMED iff (a) block-2 mostly_pos top-1 SV energy ≥ 0.70 AND (b) median principal angle (pos vs neg) decreasing monotonically from block 1 to block 11 by ≥ 30° AND (c) each of those angles outside null mean ± 2σ.
  - REFUTED iff all 12 blocks within ±2σ of null.
  - INCONCLUSIVE otherwise.
- Reported clauses:
  - Clause (a) block-2 pos top-1 = 0.5517 < 0.70 → **FAIL**.
  - Clause (b) `slice_1_11 = [71.87, 62.30, 81.08, 75.28, 76.99, 73.21, 73.59, 72.32, 21.35, 75.71, 76.96]` is NOT strictly monotone (b1 → b2 drops, b2 → b3 jumps to 81.08, etc); total_drop = b1 − b11 = 71.87 − 76.96 = **−5.09°** (angle slightly INCREASES through depth, opposite the prediction) → **FAIL**.
  - Clause (c) blocks 1..11 outside null ±2σ: 4/11 (b1 z=−3.94, b2 z=−4.18, b3 z=+4.02, b9 z=−5.66; b4-b8, b10, b11 inside) → **FAIL**.
  - REFUTED clause (all 12 inside null ±2σ): 5/12 outside (b1, b2, b3, b9 in 1..11 plus b0 inside) → **FAIL**.
- Verdict INCONCLUSIVE because both CONFIRMED and REFUTED gates fail. Logic correct.

## Methodology check
- **Calib slice**: 200 sequences × 128 = 25,600 tokens from wikitext-2 val (seed 42), texts filtered by `len(strip)>50`, tokenized with `truncation=True, max_length=128`, kept only sequences with shape[1] ≥ SEQ_LEN. Same loader as cycles 218/258/259/260. PASS.
- **c_fc input hooks**: `make_cfc_hook(l)` registers a `forward_hook` on `model.transformer.h[l].mlp.c_fc` capturing `input[0].detach().float().cpu()`. 12 hooks installed simultaneously, single forward sweep over the 200 calib sequences, hooks removed after the sweep. PASS.
- **HF Conv1D layout**: `W_fc = c_fc.weight` shape (768, 3072), `b_fc = c_fc.bias` shape (3072,), `W_proj = c_proj.weight` shape (3072, 768). All shapes asserted. `preact = ln2_out @ W_fc + b_fc` matches the cycle 218 / 260 convention. PASS.
- **σ_c definition**: `sigma_c = (preact > 0).float().mean(dim=0)` shape (3072,) — exactly the brief's `E_x[1{z_c > 0}]`. The mean is over the 25,600 tokens for each of the 3072 channels. PASS.
- **Group masks**: `mostly_neg = (sigma >= 0) & (sigma <= 0.10)`, `balanced = (sigma >= 0.45) & (sigma <= 0.55)`, `mostly_pos = (sigma >= 0.90) & (sigma <= 1.0)` — exact agenda thresholds. PASS.
- **Group geometry**: `W_rows = c_proj.weight[mask]` shape (n_g, 768). SVD via `torch.linalg.svd(W_rows, full_matrices=False)` returns `U (n, m), S (m,), Vh (m, 768)` with `m = min(n, 768)`. Top-20 right SVs taken as `Vh[:k_used].T` shape (768, k_used) where `k_used = min(n, 20, 768)`. **k_used handling is correct**: groups with fewer than 20 channels (which is ALL pos groups except b0) get k_used = n_g, so the principal-angle test then operates on a smaller subspace. The brief did not require k_right = 20 strictly; the plan flagged this risk explicitly. PASS.
- **Top-1 SV energy fraction**: `top1 = S[0]² / sum(S²)` — standard definition. Block 2 reports `[6.819, 4.446, 2.821, 2.194, 1.842, 1.361]`; sum-of-squares = 46.51 + 19.77 + 7.96 + 4.81 + 3.39 + 1.85 = 84.29; top1 = 46.51 / 84.29 = 0.5518 ✓ matches reported 0.5517. PASS.
- **rank_95**: `cum = (S²).cumsum(0) / sum(S²); rank_95 = (cum < 0.95).sum() + 1` — matches cycle 216/218/260 convention. Block 2 mostly_pos rank_95 = 5 of 6 (cum hits 0.95 only at index 5, the second-to-last). PASS.
- **Principal angles**: `M = V_pos.T @ V_neg` shape `(k_pos, k_neg)`, `cos = svdvals(M)` length `min(k_pos, k_neg)`, clamped to `[-1, 1]`, `angles_deg = arccos(cos) * 180/π`. Median taken over the angle vector. Standard principal-angle definition (subspace angles via SVD of the basis cross-product). PASS.
- **Null distribution**: 100 shuffles per block, each shuffle generates a fresh `torch.Generator().manual_seed(SEED + s)` and a random permutation of [0, 3072), partitioning the first n_neg indices to neg and the next n_pos to pos. **n_pos and n_neg are preserved per block** (the shuffle samples disjoint random subsets of the same size), so the null is conditioned on the block-specific group sizes. The null only re-randomizes which channels go in which group. PASS.
- **Null mean ± std**: computed per block over 100 medians; `inside_2sigma = (median_real ∈ [mean - 2*std, mean + 2*std])`. PASS.
- **Linalg on CPU**: weights moved to `DEVICE_LINALG='cpu'` before all linalg.svd / linalg.svdvals calls; model itself runs on MPS for the forward pass. Avoids MPS float32 instability for the 768×768 SVDs. PASS.
- **Single forward pass**: the 200 calib sequences are processed in 200 separate `model(ids)` calls (one per sequence) with hooks accumulating to `cfc_stash`. The brief said "single forward pass" — interpretive: one pass over the calib slice (not 12 passes per block). The implementation does it correctly: 200 forwards, 12 hooks installed throughout, all 12 blocks captured per forward. PASS.

## Numerical sanity
- **σ_c range**: per-block min ≈ 0, max ≈ 1, mean = 0.12-0.27. Consistent with GPT-2 left-shifted GELU pre-activation distribution (most channels are usually below zero and only fire on a few tokens). PASS.
- **Group sizes**: mostly_pos is the smallest group in every block. Block 0 has 145 pos channels (the embedding-adjacent block has more diverse activation patterns); blocks 1-11 have 2-12 pos channels. The 11/12 blocks with n_pos ≤ 12 trigger k_used ≤ 12 for the principal-angle test (the angle vector is correspondingly short). Brief constraint allowed this (k_used = min(n_g, 20, 768)). PASS.
- **Median angle plateau ~70-80°**: most blocks (b0, b3, b4, b5, b6, b7, b8, b10, b11) sit at 72-81°, which is the random-baseline median for two random subspaces in R^768 of size (small, large). The null shuffle reproduces this with mean 70-79° per block. PASS — the bulk of the data is consistent with random.
- **Block 9 outlier (21.35°)**: the mostly_pos group has only 2 channels (n_pos = 2), so V_pos is (768, 2) and the principal-angle test reduces to 2 angles. With such a small subspace the null std is also large (8.5°) — the z = −5.66 looks dramatic but the underlying subspace is degenerate. The audit notes this in the manifest under `block_9_low_angle_21deg_from_degenerate_k_used_2_subspace`. The clause-c verdict treats it as significant, but in INCONCLUSIVE territory it doesn't matter. PASS — not a bug, but worth reading carefully.
- **Blocks 1, 2, 3 significant departures (z = −3.94, −4.18, +4.02)**: these are the only "real" early-block signals. b1 and b2 are MORE orthogonal than random (the actual sign-frequency partition makes pos and neg slightly more orthogonal), b3 is MORE aligned. The pattern oscillates and is NOT monotonic in depth. This is a real but small signal, and is consistent with the cycle 217/218/258/260 finding that b1-b3 are structurally distinct in GPT-2 small. PASS.
- **Block 2 mostly_pos rank**: 6 channels with svdvals [6.82, 4.45, 2.82, 2.19, 1.84, 1.36]. Top-1 energy 0.55, top-2 energy = (46.51 + 19.77) / 84.29 = 0.787, top-3 energy = 0.881. Effective rank is ~3, NOT 1. **Cycle 174's rank-1 finding therefore CANNOT come from a 1-D column geometry** — the columns themselves span ~3 dimensions. The rank-1 must come from how the activation magnitudes α_c weight those 6 columns at block 2, not from the columns themselves. This is a clean analytical conclusion the cycle 261 data permits. PASS.
- **Total drop b1 → b11**: 71.87 − 76.96 = −5.09° (angle INCREASES). The hypothesis predicted a 30° decrease. The data shows a small INCREASE. The hypothesis is locally refuted, but the verdict logic still says INCONCLUSIVE because the REFUTED gate (all 12 inside null) requires a strictly stronger condition than just "not confirmed". PASS.

## Hypothesis vs result
- **Hypothesis**: blocks 1-3 mostly_pos and mostly_neg subspaces are partially orthogonal (median angle ≥ 60°) AND b2 mostly_pos is approximately rank-1 (top-1 ≥ 0.70), AND late blocks 8-11 the subspaces merge (median angle ≤ 30°).
- **Reality**:
  - Partial orthogonality b1-b3: PARTIALLY TRUE in the bare value sense (b1=71.87, b2=62.30, b3=81.08 are all ≥ 60°) but with the wrong sign vs null — b1 and b2 are MORE orthogonal than random, b3 is MORE aligned. The "near orthogonality" is mostly explained by random subspace geometry in R^768, not by sign-frequency-driven structure.
  - b2 rank-1: FALSE. Top-1 = 0.55, rank ~3.
  - Late-block merging: FALSE. b8=72.32, b9=21.35 (degenerate k_used=2), b10=75.71, b11=76.96. No merging — late blocks stay near-orthogonal.
- **Verdict INCONCLUSIVE is correct**. The data weakly refutes the rescue-note hypothesis on three of its three substantive claims, but does NOT refute the broader "directions come from W_down columns" framing — that framing is technically consistent with the data, just not load-bearing.
- **Counterintuitive check** (CLAUDE.md rule): the hypothesis said blocks 1-3 should be partially orthogonal AND late blocks should merge. The data says blocks 1-3 are 60-80° (near-orthogonal) and late blocks are ALSO 70-80° (near-orthogonal), with no depth trend. This is NOT counterintuitive — it's the *random-subspace* answer for two small subspaces in R^768. The result confirms that the columns DO NOT know about sign(z), modulo the b1-b2-b3 signal which is real but oscillates. The b9 outlier is a sample-size artifact (k=2). Three sanity checks support the result: (a) σ_c per-block mean 0.12-0.27 matches the known left-shifted GELU pre-activation distribution; (b) null shuffles preserve the per-block group sizes so the random baseline is properly conditioned; (c) the b2 svdvals are computed on a 6×768 matrix and reproduce by hand. No measurement bug. The "rescue-note framing is not load-bearing" reading is robust.

## Bookkeeping check
- agenda.md item 272 deleted from NEXT/P1, replaced with one-line redirect placeholder ✓ (matches cycle 259/260 discipline).
- agenda.md `### Cycle 261` PAST entry prepended above `### Cycle 260` ✓.
- TSV row 261 appended, 8 fields, hash filled = 51ccffb6 ✓.
- Manifest written with main commit hash 51ccffb6… ✓.
- Main commit 51ccffb66f010f1eb948a7104c480f06d474d431 ✓.
- Bookkeeping commit 1b43067440c31b79e22e85caad4372bc4969ddc9 ✓.
- Pre-existing dirty files NOT staged (program.md, aleph/pullback_geometry.py, flagged.jsonl, literature_map.md, older cycle audits/briefs/plans/manifests, cycle_170/205/255 manifests) ✓.
- Cycle 261 files staged: experiments/wdown_sign_geometry_cycle261.py, autoresearch/plans/cycle_261.md, autoresearch/briefs/cycle_261.md, autoresearch/logs/cycle_261_run.log, autoresearch/results/cycle_261_wdown_sign_geometry.json, autoresearch/results/cycle_261_summary.md, autoresearch/results/cycle_261_manifest.txt, autoresearch/agenda.md, autoresearch/logs/experiments.tsv ✓.
- Plan, brief, log, results, summary, manifest, exec_result, audit all under `cycle_261_*` namespace ✓.

## Decision
**PASS** — methodology clean (HF Conv1D layout preserved, single forward pass with 12 simultaneous hooks, sigma_c definition matches brief, group masks at exact thresholds, principal angles via standard subspace SVD, null shuffle preserves per-block group sizes, linalg on CPU), numerical sanity holds (top-1 energy formula matches by hand, sigma_c distribution matches GELU pre-activation expectation, b2 rank ≈ 3 reproduces from svdvals by hand), verdict logic correctly applied (clause a fails: 0.5517 < 0.70; clause b fails: not monotone, total drop −5.09°; clause c fails: 4/11 outside null; REFUTED gate fails: 5/12 outside null; INCONCLUSIVE is the only consistent label). The main analytical finding — block-2 mostly_pos subspace is rank ~3 not rank 1, so cycle 174's rank-1 must be activation-magnitude driven not column-geometry driven — is robust and follows directly from the svdvals. The k_used = min(n_g, 20) handling for small groups is correctly flagged in the plan and the data; the n_pos ≤ 12 constraint in 11/12 blocks is the real reason the test cannot decisively resolve the rescue-note framing one way or the other (group too small for stable rank tests). The follow-up directions in the PAST entry (widen σ thresholds, magnitude-weighted directions, 1-D vs 1-D angle test) are well-defined.
