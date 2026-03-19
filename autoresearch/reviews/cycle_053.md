verdict: APPROVE
round: 1

## R1: Duplication

No prior cycle isolates per-block corpus sensitivity. Cycle 005 did single-block swaps (Shakespeare probes only). Cycles 044/045 did cross-corpus evaluation but with all blocks compressed jointly. This experiment decomposes the joint gap into per-block contributions. Not a duplicate.

## R2: Known pitfalls

- **MP1** (threshold above trivial): 50% concentration threshold with 4 blocks means 2x the uniform 25% share. Non-trivial. Pass.
- **MP2**: Not applicable (PPL metric, no activation-regime detection).
- **MP3** (underdetermined fitting): `max_tokens_per_layer=2048` default, fitting 128x768 output projection (~98K params) from 2048 samples. This IS underdetermined. However, this is the same lstsq regime used in all prior cycles (005, 033, 036, 040, 042, 044, 045) and is part of the method being studied, not a flaw in experimental design. The experiment tests *this* pipeline's corpus sensitivity, not lstsq quality in general. Noted but not blocking.
- **MP4**: Same domain (GPT-2 + Shakespeare/WikiText) as cycles 044/045. Pass.
- **MP5**: Teacher model is pretrained GPT-2 — capacity is established. Pass.
- **MP6**: The sensitivity delta metric genuinely varies: cycle 044 showed +6% vs +120% PPL increase. Single-block deltas should be non-degenerate unless each block contributes exactly equally. Pass.

## R3: Method soundness

- `sensitivity_delta = (wt2_ppl_b / wt2_baseline - 1) - (shk_ppl_b / shk_baseline - 1)` is a clean relative-excess measure.
- Implicit assumption: all deltas are positive (WikiText degrades more than Shakespeare per block). This is empirically justified by cycle 044 (+120% vs +6% all-blocks), and each block individually should show the same direction. If any delta is negative, the sum-based concentration metric becomes misleading. Low risk.
- Single-block swap via `layer_ids=(b,)` correctly handled by `_replace_ffn_modules` (skips non-matching blocks).
- `swap_compressed_ffns(inplace=False)` deep-copies — no cross-contamination between block iterations. Correct.

## R4: Feasibility

8 compressions + 16 evals on a single GPU. Conservative estimate ~15 min. Well within budget.

## R5: Verdict criteria

- CONFIRMED: max delta >= 50% of sum. Clear, testable.
- REFUTED: all within +/-25% of mean. Well-separated from confirmation threshold.
- INCONCLUSIVE: concentration switches between channel-selection and refit. Covers the ambiguous case.
- No overlap between verdict regions. Sound.

## Minor note

The plan should log the teacher baseline PPL on both eval sets as a sanity check (step 5 exists but should be done first, before any compression, to confirm the model loaded correctly). This is cosmetic — reordering step 5 before step 3 costs nothing.
