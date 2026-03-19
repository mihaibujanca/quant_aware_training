verdict: APPROVE
round: 1
issues: minor (no validity risk)
feedback: |
  Plan is well-designed and directly addresses the established k=5 wall. Two minor notes:

  1. The plan reports lstsq residual MSE on training data (Step 3) but does not evaluate lstsq fit quality on probe_holdout before running full PPL eval. Adding a holdout-MSE check would catch overfitting early and costs one extra forward pass. The fit is heavily overdetermined (204K tokens, 769 unknowns per output dim, ratio ~266:1) so overfitting is unlikely, but the diagnostic is cheap insurance.

  2. The compressed blocks {1,3,5,6,7} span most of the model. Error injected at block 1 propagates through 6 nonlinear layers before reaching the correction point at block 8 input. If nonlinear interaction destroys the linear structure of the error, the correction will capture very little — but this is precisely what the experiment tests, and the verdict criteria handle it (REFUTED if <1pp). No change needed, just a note on interpretation: a null result would suggest the error is distributed and nonlinear, not merely that the correction is underpowered.

  Review checks:
  - R1 (premise): Sound. The k=5 wall is established across 4 cycles (033, 036, 037, 038). The hypothesis that residual-stream error is partly affine-correctable is motivated by prior canonical error analysis (metric distortion = 88-98% of error energy, linearly correctable in toy models). Transfer from toy → GPT-2 is the open question, which this experiment tests.
  - R2 (measurement): Sound. PPL relative change on deterministic pipeline with established baselines. The 3pp improvement threshold (+17% to +14%) is well-separated from noise. Null baseline for correction is identity (W_corr=I, b_corr=0), and the plan explicitly reports ||W_corr - I|| to verify the correction is non-trivial.
  - R3 (statistical): Sound. Lstsq fit: 200 sequences * ~1024 tokens = ~204K tokens, solving 768 independent [204K, 769] systems. Ratio ~266:1 — heavily overdetermined. No fitting concern.
  - R4 (method-architecture): Sound. Option A (forward pre-hook) is the right call. Folding into LayerNorm (Option B) would break the normalization. The hook approach applies the affine transform cleanly before the block processes the residual stream.
  - R5 (pitfalls): No pitfall applies. MP3 (underdetermined fitting) was the main candidate but the sample-to-parameter ratio is ~266:1, far above the 2:1 threshold. MP6 is low risk — PPL varies meaningfully across prior configs.

pitfall_scan: No pitfalls apply.
