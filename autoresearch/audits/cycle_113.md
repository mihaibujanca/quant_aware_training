# Cycle 113 Audit — Per-Head Amplification Census (Item 124)

## Verdict: REFUTED — ACCEPTED

## G1: Code correctness
- **OK.** Per-head amplification computed as `||head_out(x+delta) - head_out(x)||_2 / ||delta||_2`, averaged over tokens, then normalized to fractional shares. Standard approach.
- Conv1D weight indexing `W_O[h*d:(h+1)*d, :]` matches GPT-2's `[in_features, out_features]` layout. Correct.
- Causal mask, head splitting, softmax — all standard. No dropout applied (eval mode). Good.
- Ablation zeros out `c_proj.weight[h*d:(h+1)*d, :]` per head. Correctly skips bias (shared). Correlation computed against `target_block=6` transition (src=5), which is the right pairing.
- No bugs found.

## G3: Verdict follows from numbers
- **Criteria (refuted):** max_frac <= 0.15 in >= 8/12 layers (agenda says 8/12, code uses 8/11 transitions — 11 transitions is correct for 12 blocks).
- **Result:** 8/11 transitions have max_frac <= 0.15. Three exceptions:
  - Block 6->7: max=0.152 (barely over, gini=0.276)
  - Block 8->9: max=0.156 (barely over, gini=0.211)
  - Block 10->11: max=0.352 (strongly concentrated, top3=0.722)
- top3_frac ranges 0.29-0.39 in 10/11 transitions, far below 70% confirmed threshold.
- Gini coefficients 0.06-0.28 (excluding block 10->11 at 0.54) confirm near-uniform distribution.
- **Verdict is correct.** REFUTED threshold met exactly (8/11).

## G6: Method failure
- delta_L_mean ranges 11.5 to 43.8 — non-trivial perturbations, no degenerate inputs.
- 256 eval sequences, 128 tokens each — adequate sample size.
- 6x compression with topk+lstsq+500 Adam polish is the established recipe.
- Ablation Spearman rho=0.636 (p=0.026): moderate, statistically significant correlation between amplification and importance. Supports that the metric captures something real.
- No method failure detected.

## Notes
- Block 10->11 is a clear outlier (top3=0.722, max=0.352, gini=0.541). Heads 0 and 8 dominate. This is the only transition where surgical targeting would be viable.
- The claim "no surgical target heads exist" in the agenda writeup slightly overstates — it holds for 10/11 transitions but block 10->11 is an exception worth noting for future work.

## Flags: None
