cycle = 376
item = 384
verdict = CONFIRMED
audit_outcome = DOWNGRADE_INCONCLUSIVE
cka_correct = YES — Kornblith et al. linear CKA formula, centered, correct normalization
reproducibility = WEAK — single seed, single subsample (4096 of 25600 tokens), no CI
margin = RAZOR-THIN — gap 0.1010 vs threshold 0.10, only 1% above cutoff
confound_block11 = CRITICAL — block 11 is compressed in 6-block but NOT in 3-block; the CKA gap at layer 11 conflates direct compression damage with cumulative propagation error
confound_3block_beats_float = YELLOW — 3-block PPL 47.68 < float 53.89 suggests possible adapter overfitting on calib/eval overlap (both from wikitext-2-val)
layers_0_10_clean = YES — gap <= 0.0003 across first 11 layers, near-perfect alignment
g1_implementation = PASS
g2_reproducibility = FAIL — no bootstrap, no multi-seed; gap could shift below 0.1 with different subsample
g3_interpretation = MARGINAL — gap is on threshold boundary, not clearly significant
g4_counterintuitive = CONFOUNDED — layer 11 divergence expected because block 11 is only compressed in 6-block config; need control with 6-block config that excludes block 11
g5_learning = YELLOW — 3-block beating float is suspicious
g6_threshold = TOO_GENEROUS — 0.1 gap on CKA scale [0,1] with no null distribution baseline
recommendation = Rerun with (a) bootstrap CI on CKA gap, (b) 6-block config excluding block 11 as control, (c) separate calib/eval splits to rule out overfitting
