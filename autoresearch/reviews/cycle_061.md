verdict: APPROVE
round: 1
issues: 0 major, 2 minor
feedback: |
  Solid extension of cycle 059. Fresh student per (seed, n_steps) avoids state leakage,
  data splits are correct (polish on probe_holdout, eval on eval_val), and the 0/50-step
  replication baselines provide a strong anchor.

  Review checks:
  - R1 (premise): Sound. Cycle 059 established 50-step polish reduces degradation from
    +30.2% to +18.4%. Scaling to 500 steps and joint polish are the natural next questions.
  - R2 (measurement): Sound. No data leakage: `_evaluate_model` reads `corpus.eval_val`,
    polish trains on `corpus.probe_holdout`, lstsq fitting uses `corpus.probe_train`. All
    three are disjoint by construction in `build_probe_corpus`.
  - R3 (statistical): Adequate. 3 seeds, expected effect size ~10pp (18.4% → 8%).
    Well-powered for confirmation thresholds.
  - R4 (method-architecture): Joint freeze pattern is correctly specified — freeze all
    via `requires_grad_(False)`, selectively unfreeze 4 c_proj layers. Single Adam over
    the union of params is standard. Independent variant reuses proven `_polish_block_v2`.
  - R5 (pitfalls): MP3 (underdetermined fitting) not directly applicable — this is SGD
    with LM loss, not lstsq. MP6 not applicable — PPL is informative. No other pitfalls
    flagged.

  Minor issues:
  1. **Overfitting risk at 500 steps**: 200 polish sequences at batch_size=4 means each
     sequence is seen ~10 times over 500 steps. The plan tracks training loss curves but
     does not log eval PPL at intermediate checkpoints. Recommendation: log eval_val PPL
     at steps {50, 100, 200, 500} to detect train-eval divergence. If 500-step eval PPL
     is worse than 200-step, that's overfitting, not plateau — and the REFUTED criterion
     would misfire (MP6-adjacent). Cost: 4 extra eval passes per (seed, variant) run.
  2. **Joint polish RNG**: `_polish_block_v2` seeds its RNG with `manual_seed(42)`.
     The planned `_polish_joint` should use the same deterministic seeding for
     reproducibility, but the plan doesn't specify this. Ensure the joint variant uses
     a fixed generator seed so that mini-batch ordering is reproducible across runs.

pitfall_scan: MP3 (N/A — SGD not lstsq), MP5 (N/A — established pipeline), MP6 (flag: see minor #1)
