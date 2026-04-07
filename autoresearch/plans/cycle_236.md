# Task

Cycle 236 / item 246 (P0, T2): Causal intervention on the top bump directions from cycle
235's T1. For each of 12 GPT-2 blocks, take the top-5 directions by variance_explained,
intervene on the residual stream entering the next block with α · std(U_l[:, i]) · v_{l,i}
for α ∈ {-3, -1, 0, +1, +3}, measure logit shifts against a random-direction baseline, and
cross-check each direction's top-20 boosted vocab against T1's predicted boost list
(Jaccard). Save artifacts loadable by `notebooks/bump_features_explore.py`.

# Hypothesis

Item 246 verbatim: "T2: Causal intervention on top bump directions. Take the top 5
directions per block from item 245 (skip blocks where 245 found <5 coherent directions).
On a held-out 1000-token batch (subset of held-out wikitext-2 val) … For each direction
v_{l,i} and α ∈ {-3, -1, 0, +1, +3} (units = std of U_l[:, i] from calibration) … Random
direction baseline: same procedure with 10 v_random ∈ R^768 unit-norm Gaussian seeds.
Confirmed if: ≥3 directions per block in ≥6 of 12 blocks have BOTH (a) logit shift L2 at
α=3 ≥ 3× random baseline mean, AND (b) Jaccard overlap of top-20 boosted vocab between
intervention result and the T1-predicted boost list ≥ 0.3. Refuted if: no direction in any
block produces a logit shift distinguishable from random (effect size <1.2× baseline).
Inconclusive if: directions cause shifts but the shifts don't match T1's vocab predictions."

# Pre-flight Results

- P1 (dup-check): NEW item, no prior t2_intervention.json in results/.
- P2 (dep-check): cycle 235 DONE — `cycle_235_t1_features.pt`, `cycle_235_t1_decoded.json`,
  `cycle_235_t1_summary.md` all present.
- P3 (calibration): wikitext-2 train (for U_l recompute) and val (for held-out) via HF
  datasets. Same tokenizer+loader pattern as cycle 235.
- P4 (loader contract): `notebooks/bump_features_explore.py` L165–170 expects exact
  filenames `cycle_236_t2_logit_shifts.pt`, `cycle_236_t2_intervention.json`,
  `cycle_236_t2_summary.md`, `cycle_236_t2_examples.md`. L194 expects
  `entry["alpha"]`, `entry["block"]`, `entry["t1_jaccard_overlap"]`.

Clean — proceed.

# Steps

1. Create `experiments/bump_t2_intervention_cycle236.py`. Device mps, seed 42, fp32.
2. Load cycle 235 artifacts: `features = torch.load(cycle_235_t1_features.pt)` →
   per-block `V` (768,32), `S` (32,), `mean_delta` (768,). Load `cycle_235_t1_decoded.json`
   → for each (block, dir) get `top_vocab_boost` id list (top-20) and
   `variance_explained`. Parse `cycle_235_t1_summary.md` to get coherent_count per block
   (fallback: recompute from decoded.json, `label ∉ {mixed, incoherent}`).
3. Recompute U_l std from calibration. Reload GPT-2, wikitext-2 train split, N_SEQS=200 ×
   SEQ_LEN=512, seed 42 (identical to cycle 235). Register `c_fc` forward_pre_hook on all
   12 blocks, forward in batches of 8, stash inputs. For each block l, compute
   `delta_post = gelu(preact) - relu(preact)`, `delta_out = delta_post @ W_proj`,
   `delta_centred = delta_out - mean_delta_l`, `U_l = delta_centred @ V_l` (n_tok, 32),
   `std_l = U_l.std(dim=0)` (32,). Free per-block tensors after use. Save `std_l` into
   the final JSON for reproducibility. Sanity print std_l[:3] per block (must be O(1)).
4. Held-out loader: wikitext-2 val split, seed 42 shuffle. Concat+chunk to 8 sequences ×
   128 tokens = 1024 tokens. Rename vars clearly; this is DISJOINT from calibration.
5. Clean-logits forward: forward held-out ids once, cache `logits_clean` (B, T, V) on cpu
   fp32. Cache per-block residual-stream input tensors via forward_pre_hook on
   `transformer.h[l+1]` for l=0..10 and on `transformer.ln_f` for l=11 (not strictly
   needed since we re-run the forward per intervention, but record baseline for sanity).
6. For each block l = 0..11 (skip blocks where coherent_count < 5):
   - top5 = sorted(records where rec.block==l, key=-variance_explained)[:5].
   - For each direction i in top5 and each alpha in [-3, -1, 0, +1, +3]:
     - `delta_x = alpha * std_l[i] * V_l[:, i]` (768,)
     - Register forward_pre_hook on target module (h[l+1] or ln_f): the hook rewrites
       `args = (args[0] + delta_x.to(device),) + args[1:]`. For alpha==0 the hook is a
       no-op — skip registration to keep the sanity invariant clean.
     - Forward held-out batch → `logits_int` (B, T, V). Remove hook.
     - `logit_shift = (logits_int - logits_clean)` (B, T, V), cast cpu fp32.
     - Per-token L2 shift `l2 = logit_shift.pow(2).sum(-1).sqrt()` (B, T).
     - Mean shift over (B, T): `mean_shift = logit_shift.mean(dim=(0, 1))` (V,).
     - `top20_boost = topk(mean_shift, 20).indices`,
       `top20_supp = topk(-mean_shift, 20).indices`.
     - Jaccard vs T1: `t1_boost = set(first 20 token_ids of rec.top_vocab_boost)`,
       `t2_boost = set(top20_boost.tolist())`,
       `jac = |∩| / |∪|`.
     - Example records: for alpha in {+3} only, pick top-10 (batch_idx, tok_idx) by
       `l2`. For each: decode last-50-char context up to tok_idx, original top1 and
       top5, intervened top1 and top5, alpha, `logit_delta_top = logit_shift[b, t,
       orig_top1]`.
   - Random baseline (per block, once): 10 Gaussian unit-norm seeds in R^768. For each
     seed and alpha in [-3,+3] only, same hook procedure, compute mean L2 shift. Store
     `{block: {random_l2_mean_alpha_3, random_l2_std_alpha_3, random_l2_mean_alpha_neg3,
     random_l2_std_alpha_neg3}}`. Units: scale v_rand by mean(std_l) to match scale.
7. Effect size per (l, i) at alpha=+3:
   `effect_size = l2[alpha=+3].mean() / random_l2_mean_alpha_3[l]`.
8. Jaccard top-20 per (l, i) at alpha=+3 (computed in step 6).
9. Verdict: count `n_hits[l] = Σ_i (effect_size[l,i] ≥ 3 AND jaccard[l,i] ≥ 0.3)`.
   `n_blocks_pass = Σ_l (n_hits[l] ≥ 3)`.
   - CONFIRMED if `n_blocks_pass ≥ 6`.
   - REFUTED if max effect_size across all (l, i) < 1.2.
   - INCONCLUSIVE otherwise (explicitly also log the "effect without Jaccard" case).
10. Save artifacts:
    - `autoresearch/results/cycle_236_t2_logit_shifts.pt` —
      `{block_<l>: {direction_<i>: {alpha_<a>: {mean_shift: (V,), top1000_indices: (1000,),
      top1000_shifts: (1000,)}}}}`. Top-1000 selected by |mean_shift| to bound file size.
    - `autoresearch/results/cycle_236_t2_intervention.json` — list of entries with keys
      `block, direction, alpha, variance_explained, std_l_i, top20_boost (list of
      {token_id, token_str, mean_shift}), top20_suppress (same), effect_size,
      t1_jaccard_overlap, random_l2_mean, random_l2_std`. Plus top-level `u_std_per_block`
      dict, `n_blocks_pass`, `verdict`, `config`.
    - `autoresearch/results/cycle_236_t2_examples.md` — per (block, direction, alpha=+3)
      heading, then 5-row `context | orig_top1 | new_top1 | logit_delta` table.
    - `autoresearch/results/cycle_236_t2_summary.md` — verdict, per-block effect-size
      table (max, mean, n_hits), Jaccard summary, refuted/confirmed bands.

# Validation

- Sanity 1: at alpha=0 the hook is skipped and logit_shift is exactly zero — explicitly
  assert `logits_int == logits_clean` at one (l, i) before the main loop.
- Sanity 2: linearity — at α=±1 the L2 shift should be < α=±3 shift for the same
  direction. Log per (l, i) and assert `l2_alpha1 ≤ l2_alpha3` (allowing 5% slack).
- Sanity 3: print `std_l[:3]` per block during step 3; values must be finite and O(1)
  (expect roughly 1–10 range given post-proj bump scale).
- Sanity 4: cycle 235 per-block coherent counts loaded from summary match `[27, 23, 24,
  27, 21, 19, 19, 15, 17, 15, 12, 10]` — all ≥ 10 ≥ 5, so no block is skipped.

# Artifacts

- `experiments/bump_t2_intervention_cycle236.py`
- `autoresearch/results/cycle_236_t2_logit_shifts.pt`
- `autoresearch/results/cycle_236_t2_intervention.json`
- `autoresearch/results/cycle_236_t2_examples.md`
- `autoresearch/results/cycle_236_t2_summary.md`
- `autoresearch/results/cycle_236_manifest.txt` (written by execute agent)

# Fallbacks

- Wall-time tight: reduce held-out to 4 seqs × 128 = 512 tokens, or random baseline to 5
  seeds. Document in summary.
- `logit_shifts.pt` too big: drop `top1000_indices/top1000_shifts` and only save
  `mean_shift` (V,) per (block, dir, alpha). 12 × 5 × 5 × 50257 × 4B ≈ 60 MB — fine.
- std_l sanity fails (NaN or >100): fallback to `std_l = 1.0` and flag INCONCLUSIVE.

# Update bookkeeping

- Move item 246 from NEXT/P0 to PAST/Cycle 236 in agenda.md; update STATUS.md.
- Append TSV row to `autoresearch/results/experiments.tsv` (columns per convention).
- Commit: `[autoresearch] cycle 236: T2 causal intervention on top bump directions — plan`.
