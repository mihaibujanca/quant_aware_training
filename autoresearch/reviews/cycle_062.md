verdict: APPROVE
round: 1
issues: 0 major, 3 minor
feedback: |
  Clean, well-scoped experiment. The critical fix from cycle 056 (8192 probe tokens to
  avoid underdetermined lstsq at 2x) is correctly applied, and the no-polish design
  removes confounders. The 6x replication baseline anchors the sweep.

  Review checks:
  - R1 (premise): Sound. Cycle 058 established k=4@6x gives +30.2% PPL degradation
    with stacking across 4 blocks. Sweeping ratio down to 2x tests whether the stacking
    degradation is ratio-sensitive. This is the natural next question.
  - R2 (measurement): Sound. No data leakage: `probe_ffn_geometry` collects activations
    from `corpus.probe_holdout`, lstsq fits in `_fit_output_projection`, and
    `_evaluate_model` evaluates on `corpus.eval_val`. All three splits are disjoint by
    construction in `build_probe_corpus`. The pseudocode showing `corpus` passed to
    `_evaluate_model` looks misleading at first glance, but internally it only reads
    `corpus.eval_val` — no actual leakage.
  - R3 (statistical): Adequate. 3 seeds, 4 ratio levels. The +5%/+20% thresholds are
    reasonable given the 6x baseline at +30.2%. The key comparison (3x vs 6x) has a
    ~25pp expected range — well-powered.
  - R4 (method-architecture): The stacking approach probes all blocks from the teacher
    (uncompressed model), so per-block lstsq fits ignore cross-block error propagation.
    This is known and intrinsic to the method — not a plan bug. At lower ratios, the
    per-block approximation error is smaller, so cross-block compounding should be
    proportionally less severe. The experiment will measure exactly this.
  - R5 (pitfalls): MP3 (underdetermined fitting): at 2x, design matrix is 8192x1537,
    giving 5.3x overdetermination. Passes the 2x threshold. At 3x (1025 columns), 4x
    (769 columns), and 6x (513 columns), overdetermination is even higher. No MP3 risk
    at any ratio. MP6 not applicable — PPL is informative across all ratios. No other
    pitfalls flagged.

  Minor issues:
  1. **Redundant atlas computation**: The pseudocode recomputes `probe_ffn_geometry`
     inside the ratio loop, but the atlas depends only on the corpus and teacher — not
     on the compression ratio. Hoisting it outside the inner loop saves 3x probing
     passes per seed (12 forward passes total). Not a correctness issue, but each
     probing pass runs 8192 tokens through GPT-2, so this is ~30 seconds wasted per
     seed on GPU.
  2. **probe_train unused**: With polish removed, `probe_train` (200 sequences) is
     allocated but never read. Consider setting `PROBE_TRAIN_SEQS = 0` or removing
     that config key to reduce memory and avoid confusion about which data is used
     for what.
  3. **6x sanity check tolerance**: The plan allows 3pp deviation from cycle 058
     baseline, attributed to 8192 vs 4096 probe tokens. This is reasonable but should
     be reported as a delta in the results JSON so the reader knows whether the 8192
     token atlas gives a different 6x result. If the 6x result changes by >3pp,
     the 2x/3x/4x results are on a different calibration curve from cycle 058.

pitfall_scan: MP3 (checked, 5.3x overdetermination at worst case 2x — passes), MP6 (N/A — PPL is informative)
