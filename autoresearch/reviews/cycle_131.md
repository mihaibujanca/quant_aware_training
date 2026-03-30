verdict: REVISE
round: 1
issues:
  1. Single seed (seed 42 only) violates the hard rule: minimum 3 seeds required for empirical claims. The plan's confirmed/refuted criteria depend on PPL comparisons that are meaningless without variance estimates. Use seeds [42, 137, 7] and report mean +/- std.
  2. MP3 (underdetermined fitting): encoder+decoder = 1.62M params, trained on 2000 steps × batch 4 × seq 128 = 1.02M token-level samples. Ratio < 1. Either (a) increase to 4000 steps (2.05M samples), or (b) drop the second encoder layer (Linear(3072, k) only), cutting encoder params to ~1.18M and making the comparison to topk+lstsq fairer in param count too.
feedback:
  - Add >=3 seeds. This is non-negotiable per review rules.
  - Address the sample/param ratio: increase training steps to 4000 or simplify encoder to 1 hidden layer. The latter also reduces the param gap vs topk (1.62M vs 590K), making the PPL comparison more informative.
  - Minor: the plan says "E_linear is the best rank-k linear map (SVD of encoder weight product)" — clarify that this means SVD of W2 @ W1 (the two encoder linear layers composed), evaluated on held-out data. Otherwise the nonlinearity metric is ambiguous.
