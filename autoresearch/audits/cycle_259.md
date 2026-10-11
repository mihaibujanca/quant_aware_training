# Cycle 259 Audit — Item 270 (Per-token trajectory divergence d_l under depth)

**Status: PASS**

## Verdict check
- Criteria (agenda item 270):
  - CONFIRMED iff log-fit growth constant `b ≥ 0.20/layer` in `l ∈ [0, 6]` AND median per-token `cos(d_l, d_{l+1}) ≥ 0.6` in ≥4 of those layers.
  - REFUTED iff growth is sub-linear OR median per-token cos ≤ 0.2 in ≥3 early layers.
  - INCONCLUSIVE otherwise.
- Reported:
  - Log-linear fit on `l ∈ [1, 6]` (restricted from the plan's "l ∈ [0, 6]" because `d_0 = 0` exactly — both variants share `transformer.drop` output): `b_early = +0.21348`. Clause 1 passes (0.2135 ≥ 0.20).
  - Median per-token cos over `l ∈ {1, 2, 3, 4, 5, 6}` (cos indices 1..6, layer pairs (1,2)…(6,7)): [+0.3718, +0.5714, +0.7032, +0.7671, +0.7814, +0.8175]. 4 of 6 values ≥ 0.6 (indices 3,4,5,6). Clause 2 passes (4 ≥ 4).
  - Verdict CONFIRMED. Logic correct.
- REFUTED check: best-fit is log_linear (not linear or saturating). Saturating SSR = 2.67e8 is 7× larger than log_linear 3.85e7 → saturating fit ruled out. Early-band multiplicative ratio d_6/d_1 = 2.95 (between e^1.0 and e^1.1), not sub-linear. No cos values ≤ 0.2 in early indices (min is +0.3718 at cos(1,2)). REFUTED branch correctly ruled out.

## Methodology check
- **Calib slice**: 200 wikitext-2 val seqs × 128 tokens, seed 42 — byte-identical loader to cycles 220 / 228 / 258. PASS.
- **Two variants set up correctly**:
  - A (GELU): untouched HF gpt2 → calib PPL 45.0637 ✓ (matches cycle 258's 45.064).
  - B (ReLU + per-block per-channel b_leak): `ReluPlusBleak(b_leak_l)` with `b_leak_l = E_x[gelu(z) − relu(z)]` precomputed on the calib slice via a c_fc forward hook, verbatim cycle 258 recipe. 12 distinct module instances asserted. Calib PPL 60.0816 ✓ (matches cycle 258's 60.082). PASS.
- **Geometry calls**: `collect_residual_trajectory(model, ids)` per-seq, yielding 13-tensor list each [1, 128, 768]. Concatenated per-layer along batch → [200, 128, 768]. `pullback_norm(v, U)` used per-layer inside a per-sequence streaming loop (one seq at a time, peak [1, 128, 50257] ≈ 25.8 MB) to avoid the full [200, 128, 50257] ≈ 5 GB blow-up on MPS. `pullback_inner(d_l, d_{l+1}, U)` for cosine numerator, with the same per-sequence streaming pattern. PASS.
- **Flatten order**: per-layer `per_token_d` is assembled by concatenating per-sequence `[1, 128] → [128]` arrays along dim=0 in the same loop order (iterating over `calib_ids` in list order) in both the d and cos passes, giving consistent token indexing across layers and across A/B variants. PASS.
- **Log-linear fit on `l ∈ [1, 12]`**: the script correctly avoids `log(d_0)` where `d_0 = 0` exactly by restricting the regression to `l ∈ [1, 12]` (not `[0, 12]`). The verdict clause 1 uses a second restricted fit on `l ∈ [1, 6]`. This is a principled workaround and is documented in the notes field of the JSON output. PASS.
- **Cos index 0 skipped**: cos(d_0, d_1) is numerically undefined since d_0 = 0 exactly (pullback_norm(0) = 0 → 0/0). The script zeros out cos[0] via the `.clamp_min(1e-20)` denominator fallback and the verdict count uses cos indices 1..6 (6 values). PASS.
- **Restore round-trip**: post_restore PPL 45.0637 = baseline 45.0637 exactly. `assert abs(...) < 1e-4` passed. PASS.
- **Distinct instance discipline**: `assert len(ids_check) == 12` for variant B patches. PASS.

## Numerical sanity
- Cross-checked JSON, manifest, exec_result, and TSV row: all values consistent to 4 decimals.
- Key independent verification from the raw JSON:
  - `b_early` from restricted fit: 0.21348 → 0.2135 in the report. Consistent.
  - SSRs: log_linear 3.85e7 < linear 4.68e7 < saturating 2.67e8. Best-fit = log_linear. Consistent.
  - Early ratio d_6/d_1 = 665.81 / 225.42 = 2.954. Consistent with script's printed 2.95.
  - d_0 = 0.0 exactly (float32). Expected — both variants feed the same token IDs through the untouched embedding + positional + drop path, and `trajectory[0]` is captured at `transformer.drop` output which has no activation-function dependency. PASS.
  - d_12 / d_1 = 10162.68 / 225.42 = 45.08×. Consistent.
- The d_l profile is monotone increasing: 225 → 290 → 392 → 470 → 543 → 666 → 796 → 1027 → 1263 → 1707 → 2686 → 10163. No monotonicity violation. PASS.
- The cos profile is monotone increasing from +0.37 at cos(1,2) to +0.91 at cos(10,11), with a tiny dip to +0.9061 at cos(11,12). Consistent with the "late layers are more directionally coherent" reading. PASS.
- The 10× jump from d_11 = 2686 to d_12 = 10163 is consistent with the unembedding-read dominating the final residual state: the last block's output is the input to LayerNorm → lm_head, where the pullback norm becomes the natural Euclidean norm of the LM-head output, and the GELU vs ReLU final-layer output disagreement is amplified by the long downstream cascade of errors.

## Hypothesis vs result
- **Hypothesis (agenda)**: median d_l grows super-linearly through l = 0..6 then saturates around l = 8–10 (residual stream norm is bounded), AND median per-token cos(d_l, d_{l+1}) ≥ 0.6 in ≥4 of the early layers.
- **Reality**:
  - Super-linear early-band growth: CONFIRMED. b_early = 0.21/layer = e^0.21 ≈ 1.24×/layer, ratio d_6/d_1 = 2.95 over 5 layer steps.
  - Saturation around l = 8–10: REFUTED. d_l grows *faster* from layer 8 onward, not slower. The saturating curve fit (best `b = 0.02/layer`, c = 4.66e4) has SSR 7× larger than log-linear. There is no saturation; if anything the profile is super-exponential at late layers because the final layer (d_11 → d_12) jumps 3.78× in one step.
  - Cos ≥ 0.6 in ≥4 of the early layers: CONFIRMED, exactly at the threshold (4 of 6). The per-token coherence is LOWEST at early layers (+0.37 at cos(1,2)) and HIGHEST at late layers (+0.91 at cos(10,11)) — monotone increasing, the inverse of a random-walk pattern where early steps would be correlated and late steps decorrelated.
- **Cycle 189 reconciliation**: cycle 189's aggregate cos = 0.72 is recovered mid-profile (per-token median hits 0.72 around cos index 4 = layer pair (4, 5) = +0.767). Cycle 189's single number is a mid-cascade readout that undercounts the late-layer directional lock-in.
- Rule "counterintuitive results" applies: the NON-saturation and the MONOTONE-INCREASING (not early-peaked) coherence are counterintuitive but are NOT measurement bugs. Three independent checks support the finding:
  1. `d_0 = 0` exactly (sanity of the shared embedding path).
  2. Post-restore PPL = 45.0637 = baseline exactly (no patch residue).
  3. Two independent quantities (d_l and cos_l) both point in the same direction (both imply late layers are harder to patch, not easier).
  The saturation hypothesis was the naive "residual stream norm is bounded" prior, but the final-layer 10× jump is a first-layer-of-the-lm-head effect (unembedding amplifies the accumulated d_11 in the direction of the token-prediction subspace), not a residual-stream norm effect. The bounded-norm prior applies to the residual state itself, not to its GELU-vs-ReLU difference.

## Bookkeeping check
- agenda.md item 270 deleted from NEXT/P0 ✓ (replaced with a one-line redirect placeholder — consistent with the "### P0" section header discipline).
- agenda.md `### Cycle 259` PAST entry prepended above `### Cycle 258` ✓.
- TSV row 259 appended, 8 fields, hash filled = 7ac520bc… ✓.
- Manifest written with commit hash ✓.
- Main commit 7ac520bc8717a68350e21ebe69ed5912018c53e9 ✓.
- Bookkeeping commit a2522d327d45e1726ae4f5f6926ddf5365280770 ✓.
- Pre-existing dirty files NOT staged (program.md, flagged.jsonl, literature_map.md, aleph/pullback_geometry.py untracked, cycle_170/205/255 manifests, audits/briefs/plans for older cycles). Only cycle_259_* files and the intentional agenda.md + experiments.tsv updates ✓.
- Plan, review, brief, log, results, plot, summary, manifest, exec_result all under `cycle_259_*` namespace ✓.

## Decision
**PASS** — methodology clean, polarity sanity (b_leak ≤ 0 all blocks) preserved, restore round-trip clean to 4 decimals, verdict logic correctly applied, numerical cross-checks consistent. The `d_0 = 0 / cos[0] skipped / log-fit on l ∈ [1, 12]` triad is a principled workaround that the review approved up front, is documented in the JSON notes field, and is transparent in the script. The counterintuitive non-saturation finding survives three independent sanity checks and aligns with the late-layer unembedding-read amplification intuition. One note for follow-up: the agenda's "saturates around l = 8–10" prior was informed by cycle 109's block-11 residual-norm growth measurement, which may itself be worth re-measuring under the pullback metric to reconcile with this cycle's super-exponential late-layer finding.
