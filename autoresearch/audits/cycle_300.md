# Cycle 300 Audit — Item 310

**Audit status: PASS**

## Scope
Verify cycle 300 execution against item 310's verbatim hypothesis and outcomes; verify the retrain reproduces cycle 280/299 seed-42 numerics; verify the σ_c / sign-group classification for both directions; verify the boost-token lists and their unigram classification; verify the block-2 Q/K/V cosine computation and the random null baseline; verify the trained-vs-init overlap; verify the verdict logic and the written artefacts.

## Verdict verification

Item 310 outcomes (verbatim from agenda / brief):
- **CONFIRMED structured** iff ≥1 direction has top-20 unembedding cosine > 0.3 with a semantically coherent token cluster.
- **CONFIRMED unigram** iff both directions are dominated by top-20 tokens that are punctuation/stopwords/frequent-tokens (programmatic proxy: ≥14/20).
- **REFUTED interpretable** iff neither cluster is coherent AND the two directions look like random projections (no Q/K/V alignment above noise).

Auditor independently recomputed all threshold quantities from `cycle_300_semantic.json`:

| quantity | value | threshold | result |
|---|---:|---|---|
| dir 0 n_stopword/n_alpha in boost top-20 | **17/3** | ≥14 stop for unigram / ≥10 alpha for structured | unigram clause fires; structured fails |
| dir 1 n_stopword/n_alpha in boost top-20 | **19/1** | ≥14 stop for unigram / ≥10 alpha for structured | unigram clause fires; structured fails |
| any_structured | **False** | ≥1 direction with ≥10/20 alpha | FALSE |
| both_unigram | **True** | both ≥14/20 stop | TRUE |
| all Q/K/V below null mean+1σ | **False** | required for REFUTED | FALSE (K alignments hugely above null) |

CONFIRMED_unigram is the verbatim threshold result. Note that the item 310 "CONFIRMED structured" clause used a literal cosine threshold (`top-20 unembedding cosine > 0.3`) in the outcomes text; auditor checked both formulations:
- `max |cos(b, wte)|` for dir 0 trained = **0.3011** (just above 0.3), for dir 1 trained = **0.1976** (below 0.3).
- By the literal max-cos > 0.3 reading, dir 0 BARELY crosses 0.3 (+0.001) but its top-20 boost tokens are overwhelmingly stopwords (17/20), so even if the max-cos clause triggered technically, the "semantically coherent token cluster" sub-requirement fails. The script uses the programmatic coherence proxy (≥10/20 alpha subwords) which makes the verdict robust to the boundary cos value. Auditor endorses the programmatic check as the cleaner operationalization of "semantically coherent" — the 17 stopwords + 3 alpha words (`' one'`, `' it'`, `' I'`) on dir 0's boost list are not a coherent semantic cluster.

CONFIRMED_unigram matches the JSON `verdict`, manifest, exec_result, and agenda PAST entry.

## Retrain reproducibility (cycle 280/299 cross-check)

| quantity | cycle 280 ref | cycle 299 seed-42 | cycle 300 measured | delta vs 299 |
|---|---:|---:|---:|---:|
| float PPL | 53.895 | 53.895 | 53.895 | 0.000 |
| init rank-2 PPL | 57.664 (3-seed) | 57.6581 | 57.6581 | 0.000 |
| best rank-2 PPL | 49.032 (3-seed mean) | 49.0603 @ step 3900 | 49.0603 @ step 3900 | 0.000 |
| Adam recovery | 8.632 | 8.598 | 8.598 | 0.000 |
| final PPL @ 4000 | — | 49.2498 | 49.2498 | 0.000 |

Cycle 300 reproduces cycle 299's seed-42 trajectory to machine precision — same dataloader seed, same model, same device, same library versions. The `assert abs(init_ppl - CYCLE_280_INIT_REF) < 0.05` and `assert best_ppl < init_ppl` both fire cleanly in the script.

## Independent sanity on input-side statistics

Auditor independently computed from `cycle_300_semantic.json`:

- `sigma_c_stats`: min=0.000 max=0.9952 mean=0.2636 median=0.2717. n_mostly_neg(σ≤0.1) = 1188/3072 (38.67%). n_mostly_pos(σ≥0.9) = 12/3072 (0.39%). n_bump(0.1 < σ < 0.9) = 1872/3072 (60.94%). These sum to 3072 ✓.
- **Only 12 channels out of 3072 are "mostly_pos" at block 1 on the calib slice.** This is a background fact of GPT-2 small block 1 at 200×128 wikitext-2 tokens; the "mostly-negative dominates" hypothesis is partially structural (most channels are not mostly_pos) and partially load-bearing (the top-20 A[:, i] channels avoid even the 12 mostly_pos ones).

Sign-group counts for the top-20 |A[:, i]| channels:
- **dir 0**: mostly_neg=15, bump=5, mostly_pos=**0**, balanced(0.4, 0.6)=0. Sums to 20 ✓.
- **dir 1**: mostly_neg=11, bump=9, mostly_pos=**0**, balanced=1 (the balanced range is a subset of the bump range; balanced is counted separately for diagnostic, the 9 bump count is the main report).

Both directions have **zero mostly-pos channels** in their top-20. Item 310 hypothesis C ("both directions are dominated by mostly-negative-regime FFN channels") is cleanly CONFIRMED.

## Top-20 channel overlap between directions (auditor independent)

Auditor intersected the sets of top-20 channel indices:
- |dir0 ∩ dir1| = 10 (shared): {242, 293, 706, 713, 1120, 1307, 1467, 1859, 2401, 2598}
- |only_dir0| = 10: {463, 605, 615, 780, 1159, 1638, 1910, 2120, 2194, 2765}
- |only_dir1| = 10: {8, 38, 67, 74, 155, 1386, 1702, 1959, 2820, 2859}

Half the top-20 channels are shared. The three highest-magnitude channels of dir 1 (1120, 242, 2401) are all in dir 0's top-5; dir 1 uses the SAME dead-channel basis with different coefficients and magnitudes ~2.5× larger (dir 0 max |a| = 5.67; dir 1 max |a| = 12.68). This is consistent with the manifest's claim that "dir 1 is essentially a rescaling of the same dead-channel basis with a different linear combination".

Correction note: the manifest / exec_result top-5 for dir 0 lists (2401, 1120, **293**, 1467, 242), not (2401, 1120, **242**, …). Auditor confirms the JSON agrees — c=293 at rank 2 (σ=0.813, bump) has |a|=4.691 while c=242 at rank 4 has |a|=4.195. Both appear in the top-5. The written manifest/exec_result top-5 lists are correct.

## Boost/suppress token verification

Auditor re-read `cycle_300_semantic.json.directions_trained.dir_{0,1}.output` and confirmed the boost token lists reported in the manifest/exec_result/agenda are verbatim (same order, same escaping of leading spaces).

Classification check (STOPWORD_SET = punct + ~30 most frequent English stopwords with leading-space variants):
- dir 0 boost top-20: ` the, ',', ' and, ' ", '.', '\n', '-', ' a, ' in, ' (, ' The, ' to, ' of, ' with, ' it, ' one, ':', ' for, '/', ' I`.
  - Stop/punct: ` the, ',', ' and, ' ", '.', ' a, ' in, ' (, ' to, ' of, ' with, ' it, ':', ' for` — plus `'\n'`, `'-'`, `'/'` (non-alpha punct/whitespace, counted as stop by `isalpha()` failure). = 17 stop.
  - Alpha subwords: `' The'` (leading-space alpha, not in STOPWORD_SET), `' one'` (not in set), `' I'` (not in set). = 3 alpha.
  - 17 + 3 = 20 ✓. Matches the reported 17/3.
- dir 1 boost top-20: ` the, ',', '-', ' in, ' and, '.', ' a, ' ", ' (, ' with, ' to, ' of, ' on, '\n', ' is, ':', ' that, ' for, ' The, ' from`.
  - Stop/punct: ` the, ',', '-', ' in, ' and, '.', ' a, ' ", ' (, ' with, ' to, ' of, ' on, '\n', ' is, ':', ' that, ' for, ' from`. Including `' from'` which is not in the base stopword set but… auditor re-checks: the script's `STOPWORD_SET` is `STOPWORD_BASE ∪ {" " + w for w in STOPWORD_BASE}`. `' from'` is in `STOPWORD_BASE` (see line `"for", "on", "with", "as", "by", "at", "from"`). So `' from'` → leading-space stop ✓. `' on'` is also in the base set ✓. `' is'` is in the set ✓. That accounts for all 19 stop-labeled tokens.
  - Alpha subwords: `' The'` (not in STOPWORD_SET — the base set has lowercase `'the'` and `' the'`; `' The'` capitalized is NOT in the set → alpha). = 1 alpha.
  - 19 + 1 = 20 ✓. Matches the reported 19/1.

Both clause counts check out. Item 310 CONFIRMED_unigram threshold (≥14/20 each) passes by 3 and 5 tokens respectively.

## Block-2 Q/K/V alignment and random null

Null (N=100 Gaussian unit vectors in R^768, seed 20260406): Q mean 0.2859 std 0.0429; K mean 0.2819 std 0.0477; V mean 0.2837 std 0.0545. The null for top-5 |cos| sum is the sum of the 5 largest absolute cosines of a uniform random unit vector in R^768 against a fixed 20-column orthonormal basis — the expected value is small (~0.28). Auditor sanity: for a uniform unit vector in R^768, the squared cosines against 20 orthonormal directions have expected sum 20/768 ≈ 0.026, so the expected sum of absolute cosines is O(√0.026 × 20) ≈ 3.2 (loose upper bound). The top-5 |cos| sum ≈ 0.28 is consistent with the empirical distribution of 100 random draws. ✓

Z-scores:
- dir 0: Q z = (0.4048−0.2859)/0.0429 = +2.77; K z = (0.8445−0.2819)/0.0477 = +11.79; V z = (0.4297−0.2837)/0.0545 = +2.68.
- dir 1: Q z = (0.4771−0.2859)/0.0429 = +4.45; K z = (0.9387−0.2819)/0.0477 = +13.77; V z = (0.1902−0.2837)/0.0545 = −1.72.

All above +2σ on Q and K for both directions. Dir 0 V above +2σ; dir 1 V below the null mean (−1.72σ). The K-reader is the dominant downstream reader for both directions. The `all_qkv_below_null` clause (REFUTED interpretable) does NOT fire — K alignments are ~12 and ~14 sigma above null, which is emphatically NOT "random projection of the unembedding".

## Trained vs SVD-init overlap

| direction | intersection / 20 | only_trained | only_init |
|---|---:|---|---|
| dir 0 | 17 | `' one', ' with', '/'` | `' an', ' is', "'s"` |
| dir 1 | 19 | `' from'` | `"'s"` |

Auditor independently diffed the JSON `directions_trained.dir_{i}.output.boost_tokens` with `directions_init_svd.dir_{i}.output.boost_tokens` and confirmed the above. Only 3 tokens changed on dir 0 and 1 token on dir 1 — Adam's 4000-step recovery moves ≤15% of the boost-token set while rotating the B row space by ~11% (cycle 299's 0.91 cos). The token-level semantics is SET at SVD init, and this is consistent with cycle 299's "SVD init is 83% of optimal basin by principal-angle energy" finding.

## Plan adherence

Plan (`autoresearch/plans/cycle_300.md`) vs execution (verified from JSON `config` and script):
- Step 1 (new script) ✓
- Step 2 (SVD of block 1, A2 frozen, B2_init = Vh[:2, :], LowRankCProj installed) ✓
- Step 3 (Adam 4000 steps lr=1e-4 seed 42, asserts init/best) ✓
- Step 4 (σ_c / W_fc col norms / pre-activation stats cached via forward hook on ln_2) ✓
- Step 5 (input side top-20, output side top-20 boost/supp, Q/K/V top-5 cosines) ✓
- Step 6 (SVD-init comparison output-side, overlap report) ✓
- Step 7 (verdict logic matches plan) ✓
- Config fields: TARGET_BLOCK=1, RETRAIN_RANK=2, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, SEED=42, DEVICE=mps, TOPK=20, N_RANDOM_SEEDS=100, trainable 2304 ✓
- Files: all listed files present ✓

## Trainable param count

LowRankCProj: `B` is `(2, 768) = 1536` params; `bias` is `(768,) = 768`. Total trainable = **2304**. All other model params have `requires_grad_(False)`; the frozen `A` is a buffer. Asserted at runtime: `assert n_trainable == expected_trainable` ✓.

## Convergence check (CLAUDE.md verify-task-performance rule)

- float PPL 53.895 matches cycles 266/268/269/270/280/299 to 3 decimals.
- init rank-2 PPL 57.658 matches cycle 299 seed-42 to 0.000 (machine precision).
- best rank-2 PPL 49.060 @ step 3900 matches cycle 299 seed-42 exactly (same seed, same loader ordering).
- `best_ppl < init_ppl` by 8.598 PPL (task performance verified before analysis).
- The model trained correctly (the task is to improve the rank-2 replacement of block-1 c_proj via Adam on B + bias; the 8.6 PPL gain matches cycle 280's 8.6 PPL headline). ✓

## Geometric reading — audit

The auditor independently verifies the substantive interpretation:

1. **Both rank-2 directions are unigram axes**: 17/20 and 19/20 stopwords respectively. The "dir 1 is more structured" half of item 310's hypothesis is refuted by a wide margin (dir 1 is MORE unigram-concentrated, not more semantic). The substantive answer to "what are the 2 directions?" is "TWO PROJECTIONS of the SAME unigram-frequency cluster". Dir 0 and dir 1 share 10/20 top-20 input channels and share all three of their largest-magnitude dead channels (1120, 242, 2401).

2. **Mostly-negative dominance**: 0/20 mostly_pos channels in both directions, against a background fraction of 12/3072 ≈ 0.4% mostly_pos channels in block 1. Even if the top-20 were drawn uniformly at random, the expected number of mostly_pos in top-20 is 20 × 12/3072 ≈ 0.08, so observing 0 is not strong evidence against "random uniform draw" per direction. But the presence of 15 and 11 mostly_neg (expected 20 × 1188/3072 ≈ 7.7 under uniform draw) is +2.6σ and +1.2σ above the uniform-draw null — moderate evidence for a real concentration toward dead-side channels. The stronger observation is the **magnitude concentration**: dir 1's top-5 channels all have σ ≈ 0.008, with |a| values 1.5–12.7, showing heavy tails on dead-side channels specifically.

3. **Block-2 K-reader alignment**: +11.8σ and +13.8σ above null is massive. Block-2 attention keys read the unigram axis heavily. Item 310 hypothesis "Q/K/V alignment" is CONFIRMED with K as the dominant reader, Q secondary, V weak (dir 0) or absent (dir 1). This matches cycles 217/218's "attention attends to frequent tokens" finding — the unigram axis written by block 1 c_proj-rank-2 is read by block 2 K.

4. **Retrain invariance of the boost-token set**: 17/20 and 19/20 overlap with the SVD init. The 8.6 PPL recovery does NOT move the "which tokens get promoted" structure meaningfully. Combined with cycle 299's "83% of optimal basin by principal-angle energy", the full picture is: the rank-2 bottleneck's semantic content (unigram axis) is determined by the SVD truncation of W_proj; Adam fine-tunes a small row-space rotation within the unigram plane plus a bias correction to collect the remaining PPL; the "which tokens are read" structure is set at initialization.

5. **Direction 1 is orthogonal to block-2 V reader**: V top5_sum 0.19 < null mean 0.28. This is the single "negative space" observation — whatever dir 1 writes is NOT read by block-2 values. Block-2 attention uses dir 1 via K only, not V. This is a geometric asymmetry worth noting for cycle 311 / item 311 (curvature residual vs col(A) vs col(V_reader)).

## Corrections

None on the verdict. CONFIRMED_unigram is the literal verbatim threshold result and the substantive answer (both directions are unigram axes, dir 1 sub-hypothesis refuted, mostly-negative dominance confirmed, block-2 K-reader alignment strong) is consistent with it.

Minor wording: the manifest describes dir 1 as "dir 1 is essentially a rescaling of the same dead-channel basis". The shared-channel count is only 10/20, not 20/20, so "rescaling" is a slight overstatement; the precise statement is "dir 1 uses the same 3 top-magnitude dead channels as dir 0 (1120, 242, 2401) at ~2.5× magnitude and shares 10/20 of its top-20 basis with dir 0". Auditor recommends the precise statement but does not require a manifest edit; the manifest's qualitative direction is right.

Minor wording: the item 310 outcome text says "top-20 unembedding cosine > 0.3" for CONFIRMED structured. Dir 0's max |cos| is 0.3011 (above 0.3 by 0.001). If the auditor used the literal cos-threshold interpretation, dir 0 would BARELY trigger CONFIRMED structured, but the top-20 tokens list is overwhelmingly stopwords (17/20), so the "semantically coherent cluster" qualifier fails. The script's programmatic proxy (≥10/20 alpha subwords) is the cleaner operationalization — it makes the verdict robust to the 0.3001 boundary and agrees with what a manual review of the token list would conclude. Auditor endorses the programmatic check.

## Artifact existence

- `experiments/cycle280_A_semantic_cycle300.py` — exists, ~560 lines, committed in 0405442
- `autoresearch/plans/cycle_300.md` — exists, item 310 verbatim Hypothesis section, 8 H2 headings (Item 310 verbatim Hypothesis, Dependencies, Approach, Files, Config, Verification, Baselines, APPROVE), marked `# APPROVE`, within 45-line budget
- `autoresearch/briefs/cycle_300.md` — exists (from brief stage, pre-existing)
- `autoresearch/results/cycle_300_semantic.json` — exists, valid JSON, verdict="CONFIRMED_unigram", both directions populated with input_channels (20), output.boost/supp_tokens (20 each), block2_qkv_cos (Q/K/V), plus directions_init_svd and trained_vs_init_overlap
- `autoresearch/results/cycle_300_summary.md` — exists, verdict + all 3 H2 tables per direction (input channels, boost, suppress) + SVD-init comparison + overlap + verdict logic
- `autoresearch/results/cycle_300_manifest.txt` — exists, commit=0405442 (patched from pending)
- `autoresearch/results/cycle_300_exec_result.txt` — exists, commit=0405442 (patched from pending)
- `autoresearch/logs/cycle_300_run.log` — exists, captures retrain trajectory + directional analysis + verdict (progress bars stripped post-run)
- Primary commit `0405442` exists in git log
- `experiments.tsv` row 300 appended with 8 columns; hash patched from `pending` to `0405442` ✓
- `agenda.md`: item 310 deleted from NEXT P1; added to PAST as Cycle 300; P1 NEXT is now empty; item 311 remains in P2 ✓
- `STATUS.md`: regenerated to "300 Cycles (001-300)", new "Latest: Cycle 300" section ✓

## Final status

**PASS** — keep provisional outcome CONFIRMED_unigram. The verdict logic is correctly applied to the verbatim thresholds (both directions have ≥14/20 stopword boost tokens; neither has ≥10/20 alpha subwords; `all Q/K/V below null` clause is false because K-alignment is massively above null). The retrain reproduces cycle 280/299 seed-42 numerics to machine precision; the model trained correctly (best < init by 8.60 PPL); the substantive finding (both directions are unigram axes, mostly-negative dominance confirmed, block-2 K-reader alignment strong, boost-token set set-at-SVD-init) is documented in the manifest / exec_result / agenda PAST / STATUS.md consistently.

Auditor note: cycle 300 closes item 310 with a clean CONFIRMED_unigram label AND a refutation of item 310's own "direction 1 is semantically structured" sub-hypothesis. The refutation is the more interesting result — it says the rank-2 bottleneck at block 1 is narrower than "unigram + one semantic direction", it's 2D *within* the unigram cluster. This reframes item 311's question (curvature residual vs col(A)) as "does the curvature contribute to the unigram reading or is it orthogonal to it" — a sharper question than item 311 originally posed. A follow-up worth queuing if cycle 311 confirms the curvature lives in col(A)^⊥ would be to ask what the col(A)^⊥ direction encodes at the unembedding — if it's semantic, then block 1's 2D rank-2 LASER truncation is specifically throwing away a curvature-carrying semantic axis, which would be a stronger interpretive claim than cycle 280's current "rank-2 is enough for PPL" finding.
