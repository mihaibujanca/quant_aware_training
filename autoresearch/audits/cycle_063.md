status: FAIL
original_outcome: inconclusive
audited_outcome: inconclusive (understated concerns)
guardrail_flags: implementation-mismatch,underpowered,high-variance
reason: |
  G1 (Implementation): The naive and cascaded variants are structurally identical in the committed code.
  Both `run_naive_corrections` and `run_cascaded_corrections` iterate sorted blocks, collect h_comp from
  the comp_model (which accumulates prior hooks), fit lstsq, then install the hook before the next
  iteration. Despite the docstring claiming naive fits "on float activations", the h_comp source is
  always the comp_model with prior corrections installed — making both variants cascaded. Yet block 1
  (first correction, no prior hooks) shows a 2.7-3.0x fit MSE difference across all seeds (naive ~0.27
  vs cascaded ~0.097). Since both start from fresh `swap_compressed_ffns(teacher, pack, inplace=False)`
  deepcopies with no prior hooks, the block 1 MSE should be identical. This unexplained discrepancy
  indicates either a code version mismatch (the experiment ran on different code than committed) or an
  unidentified confound. The claimed naive-vs-cascaded comparison is unreliable.

  G2 (Reproducibility): Config is explicit, seeds are set, data paths are clear. However, the
  implementation bug above means re-running the committed code would not reproduce the reported naive
  results — the two variants would produce identical corrections.

  G3 (Results): With only 3 seeds and std of 18.78pp, the +25.35% cascaded mean is not statistically
  distinguishable from uncorrected +30.20% (paired t=2.26, p>0.05 needs t>4.303 for df=2). Seed 42 is
  a massive outlier (+46.7% degradation for both variants, vs +11-18% for seeds 123/7), dominating
  the mean and inflating std. The claimed ~5pp improvement (cascaded vs uncorrected) is within noise.

  G5 (Naive interference): The finding that naive corrections (+50.8%) are worse than uncorrected (+30.2%)
  is directionally consistent with cycle 043's interference observation. However, given the G1
  implementation-mismatch finding, the magnitude of the naive-vs-cascaded gap (50.8% vs 25.4%) cannot
  be trusted — the naive variant may have run on different code.
recommendations: |
  1. Fix the naive implementation: collect h_comp for all blocks BEFORE installing any hooks (batch all
     collections first, then fit all corrections, then install). This is the only way to get true
     "independent" naive corrections.
  2. Re-run with 5+ seeds to get adequate power. The current 3-seed design cannot detect a 5pp effect
     with std ~18pp. Alternatively, investigate and fix the seed-42 outlier (likely a bad probe sample).
  3. Until rerun, downgrade the cascaded-vs-uncorrected claim from "slightly beats" to "no evidence of
     improvement" — the effect is not significant.
