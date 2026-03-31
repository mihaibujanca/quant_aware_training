status=PASS
old_outcome=confirmed
new_outcome=confirmed
reason=Param-matched AE (d_bn=64, 247K) beats topk (788K) by 3.0-3.4% on both tested blocks (4, 7) with very low seed variance. Scoped to 2 blocks only but result is strong and consistent. Clear CONFIRMED.
guardrail_flags=G2_only_2_blocks_tested
recommendations=Test blocks [1, 11] for generalization. Item 152 (AE stacking) is now critical. The full AE worsening after polish is an important negative finding — large AE overfits, small AE is robust.
interesting=true
interesting_reason=First confirmed method to beat topk+lstsq at matched or fewer params. The d_bn=64 nonlinear bottleneck captures cross-channel correlations that axis-aligned top-k selection misses. This is the clearest evidence that learned compression can surpass no-training methods.
