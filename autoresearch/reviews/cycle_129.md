verdict: REVISE
round: 1
issues:
  - R3 seed requirement: random-init condition uses no seeds. Different random initializations produce different weight matrices and therefore different eigenspectra. Plan must specify >=3 random seeds for the random-init condition and report mean +/- std of isotropy ratios. Decision thresholds should compare against the trained values using the seed-averaged random-init ratio.
  - R4 minor (Test 2 confounder): ReLU hard zeros mechanically kill ~50% of covariance eigenvalues, reducing isotropy ratio regardless of learned structure. This makes the ReLU-swap isotropy ratio incomparable to the GELU baseline. Suggest: either drop Test 2 or add a control that measures isotropy only over the non-zero channels (positive activations) so the comparison is fair.
feedback:
  1. Add >=3 random seeds (e.g., [42, 123, 7]) for random-init GPT-2. Report per-seed and mean+-std isotropy ratios. Adjust decision logic to compare mean random-init ratio vs trained ratio.
  2. For Test 2: either (a) restrict eigenspectrum to non-zero-activation subspace, or (b) downgrade to qualitative observation and exclude from decision logic. Current form conflates ReLU's structural zeros with isotropy change.
