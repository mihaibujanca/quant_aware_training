verdict: APPROVE
round: 1
issues: 0 blocking, 1 minor

## R1: Premise
Sound. Cycle 044 showed +120% zero-shot degradation (Shakespeare channels evaluated on WikiText-2). The root cause could be channel selection mismatch, lstsq refit mismatch, or both. Item 64 showed channels are unstable (Jaccard 0.34-0.58), so this experiment isolates the refit variable by fixing channels from Shakespeare. Even if channels are suboptimal, the refit data distribution still affects W_down quality. The dependency on item 64 (Jaccard >= 0.7) is unmet, but the experiment remains informative: it tests whether mixed refit compensates for unstable selection.

## R2: Measurement
Raw compressed PPL on WikiText-2 validation is the right metric. No post-hoc correction confound. Relative degradation (%) normalizes across seeds. The 10pp threshold for confirmation is meaningful: cycle 044 condition C was ~97% degradation, so 10pp reduction (to ~87%) is modest but detectable.

## R3: Seeds
3 seeds [42, 123, 7], cross-seed std < 30% guard. Adequate for 3-condition comparison.

## R4: Method-architecture match
The plan correctly separates channel selection (from Shakespeare atlas) from lstsq refit (from mixed probe data). `_fit_output_projection` takes arbitrary hidden/output tensors, so mixing corpus samples is straightforward — just concatenate the tensors. The plan builds CompressedFfnPack manually with fixed channels but re-fitted W_down, which is the correct approach.

## R5: Pitfalls
- MP3 (underdetermined fitting): At 6x compression, hidden_dim ~128. W_down is 768x128, so lstsq has 128 predictors (+ bias = 129). With 512 tokens, N/P ratio = ~4. With 2048 tokens, N/P = ~16. Both well above 2x threshold. Not triggered.
- MP4 (domain mismatch): The whole point of this experiment is to test cross-domain behavior. The mixed-corpus approach is the mitigation. Not triggered.
- MP5 (capacity): GPT-2 pretrained, compression at 6x on 4 blocks — known to work from prior cycles. Not triggered.
- MP7 (post-LayerNorm): Probe hooks capture FFN input/output, not post-LayerNorm representations. Not triggered.

## Minor observations

1. **Condition A baseline may not match cycle 044 exactly.** Cycle 044 used `probe_ffn_geometry` with `max_tokens_per_layer=2048` for the atlas (which also feeds the lstsq refit). This plan uses 512 tokens for condition A. The difference in refit sample size alone could affect results. Consider noting this when comparing to cycle 044 numbers, but it is not blocking since the within-experiment comparison (A vs B vs C) uses matched token counts for A and B.
