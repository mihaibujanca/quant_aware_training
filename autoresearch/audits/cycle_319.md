# Cycle 319 Audit — Item 334 (curvature residual decomposition on parameter-golf)

## Audit status: PASS

## Provisional outcome (from execute)
INCONCLUSIVE per item 334's strict three-way verdict tree, with substantive findings sharper than INCONCLUSIVE: (lrelu², relu²) curvature is rank-localized at b4 (rank_95=18), b5 (33), b12 (21), but the ‖Δy‖_F ≥ 2× median bar is missed (b4/b5 sit at ~1.9× median; b12 is 0.025× median because absolute magnitude is small at the output-adjacent block).

## Verdict logic check (item 334 verbatim)
The agenda's verdict tree:
1. CONFIRMED-leak-localized if any block has (lrelu²,relu²) `rank_95(Δy) ≤ 50` AND `‖Δy‖_F ≥ 2× median`.
2. CONFIRMED-leak-diffuse if (lrelu²,relu²) `rank_95(Δy) ≥ 200` at every block.
3. REFUTED-no-difference if (lrelu²,relu²) `‖Δy‖_F` is within 5% of (relu²,relu) at every block.
4. INCONCLUSIVE otherwise.

Measured (lrelu², relu²) per-block rank_95: [324, 318, 290, 202, **18**, **33**, 146, 147, 161, 179, 169, 127, **21**]. Median ‖Δy‖_F = 31949.804. 2× median = 63899.608.

- Branch 1: any rank_95 ≤ 50 → TRUE (b4, b5, b12). AND ‖Δy‖_F ≥ 2× median:
  - b4 ‖Δy‖_F = 59974.196 (< 63899.608) — FAIL
  - b5 ‖Δy‖_F = 61675.756 (< 63899.608) — FAIL
  - b12 ‖Δy‖_F = 796.854 (≪ 63899.608) — FAIL
  - No block satisfies BOTH clauses simultaneously. Branch 1 FAIL.
- Branch 2: all rank_95 ≥ 200 → FALSE (3 blocks below 200 plus b3=202 only just above).
- Branch 3: all rel diff ≤ 5% → FALSE (rel diffs range 3-64%; b3 is 56.8%; only b12 is at 3.1%).
- Branch 4: INCONCLUSIVE — TRUE.

The strict-tree branch falls through to INCONCLUSIVE. The exec script's verdict logic at `experiments/pgolf_curvature_residual_cycle319.py:write_summary` was traced: the `localized_blocks` list is empty because the AND-with-magnitude clause filters all candidates. The code is correct; the bar is calibrated tighter than the data supports as a CONFIRMED outcome. PASS the INCONCLUSIVE label.

The substantive sharper finding is "rank-localization is real" — b4/b5/b12 have rank_95 ≤ 33 while the other 10 blocks are in [127, 324]. The 2× median magnitude bar was not the right discriminator for this geometry: at the localized blocks, curvature is *concentrated* into a small subspace rather than *amplified* in absolute magnitude. A "rank_95 ≤ 50 alone" criterion would have CONFIRMED-leak-localized at b4/b5/b12. A constructive follow-up (logged in next=) is to test that revised criterion.

## Cross-checks (all PASS)
- 3 subprocess invocations completed successfully. Combined JSON has `pairs` list of 3 entries; each has `n_blocks=13`.
- Each block reports `hidden_dim` matching load_model_v2's per-block mlp_mult: blocks 0-6 and 9 at 1920, blocks 7/8/10/11 at 1280, b12 at 320. Verified by reading the JSON profile shapes.
- Each block reports `n_tokens=25600`.
- Trainable params = 0 (forward-only, no optimizer; pgolf script inserts no Linear modules and never calls .backward()).
- The (gelu, relu) cross-reference produces a sane shape (rank_95 in [267, 406] across blocks 0-11, then 1 at b12). The b12 collapse to rank_95=1 is structurally analogous to GPT-2 cycle 302 (where b2/b3/b4=1).
- The (relu², relu) result correlates with cycle 318 eff_rank: the lowest-eff_rank blocks (b3=4.89, b12=6.94, b11=13.40) all have low rank_95(Δy) (72, 68, 116). Correlation across blocks: low eff_rank → low rank_95(Δy). This is independent corroboration that the curvature residual lives in the same subspace as the rank-collapsed output covariance — same finding as cycle 301 on GPT-2.

## Numerical sanity (re-checked)
- (relu², relu) ‖Δy‖_F monotonically grows from b0=29025 to b9=106417, then drops at b10/b11 (95317/71802) and crashes at b12 (823). The b12 crash is consistent with hidden=320 vs the other blocks at 1280-1920 and is also consistent with the cycle 318 finding that b12 has the smallest output magnitude (small mlp_scale × small post-MLP).
- (lrelu², relu²) median 31950 vs (relu², relu) median 51044 — the lrelu² leak has roughly 2/3 the magnitude of the relu² curvature, consistent with the leak being a smaller perturbation (only in the negative half).
- (gelu, relu) magnitudes are dramatically smaller (median ~900) because the GELU bump on a relu²-trained activation distribution sits in a very different per-channel range — most channels with z>0 see GELU(z) ≈ z (curvature near zero), and the active negative channels are sparser.

## Independent interpretation
This is the second cross-project measurement (after cycle 318) and the first time the cycle 312 unified curvature framework has been run on a non-GELU-trained model.

**The substantive headline**: the (lrelu², relu²) curvature residual concentrates to ≤ 33 dimensions in 3 specific blocks (b4, b5, b12) on the parameter-golf 13L_640d ReLU² checkpoint. b4 is a mid-stack block, b5 is mid-stack, b12 is the output-adjacent block. If the parameter-golf leaderboard observation that "lrelu² beats relu² by ~0.005 BPB" reflects a real geometric mechanism, the candidate layers where the lrelu² gradient signal flows are b4, b5, b12 — not the entire stack. This is consistent with item 334's hypothesis "the lrelu² leak is concentrated in 1-3 specific blocks". The strict verdict tree's magnitude bar prevented CONFIRMED-leak-localized, but the rank evidence is unambiguous.

**Cross-validation against cycle 318**: the (relu², relu) curvature rank_95 profile [389, 378, 322, **72**, 206, 210, 178, 149, 168, 172, 154, 116, **68**] tracks the cycle 318 eff_rank profile [187, 160, 86, **4.89**, 38, 40, 28, 23, 31, 29, 22, 13, **6.94**] in shape — the two lowest eff_rank blocks (b3, b12) are also the two lowest rank_95(Δy) blocks. This independent corroboration replicates cycle 301's finding on GPT-2 ("curvature residual lives inside the rank-2 kept subspace") on a non-GELU non-GPT-2 model. The geometric mechanism generalizes.

**Methodological note about the verdict tree**: item 334's tree mixes a rank threshold (≤ 50, a structural property) with a magnitude threshold (≥ 2× median, a relative-energy property). Geometric concentration ≠ magnitude amplification. A future revised tree should either (a) use "rank_95 ≤ 50 alone" or (b) compare ‖Δy‖_F to absolute baseline (e.g. (relu², relu) at the same block) rather than relative to median. The current tree is conservative and yields INCONCLUSIVE on a result that the substance supports.

## Auditor recommendation
PASS the INCONCLUSIVE label. The execute layer is honest about the strict-tree outcome and explicit about the substantive sharper finding ("rank-localization is real at b4/b5/b12"). The verdict logic in the script is correct; the agenda's verdict tree is calibrated such that this measurement falls into the middle band.

The natural next step is to re-classify under a "rank_95 ≤ 50 alone" criterion as a separate cycle, AND to test whether the b4/b5/b12 leak directions share residual-stream support (cycle 316-style top-k angle test on the captured top-5 features). File as new agenda items; do NOT retroactively modify cycle 319.

## Status: PASS (provisional outcome accepted unchanged)
