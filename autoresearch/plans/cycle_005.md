# Task
Measure single-block FFN swap behavior on the current runnable proxy benchmark (`gpt2` + Shakespeare proxy corpus) by sweeping individual layers and matched compression budgets with both the analytic method and the matched-budget distillation baseline.

# Hypothesis
- **Confirmed if:** at least 3 tested blocks achieve `>=4x` FFN parameter reduction with the analytic method while keeping relative perplexity increase `<=10%` and staying within `25%` of the distilled baseline on the same swap.
- **Refuted if:** the analytic method causes relative perplexity increase `>25%` for more than `80%` of tested single-block swaps at `>=4x` reduction.
- **Inconclusive if:** results vary strongly by depth with no stable pattern.

# Pre-flight Results
- P1 PASS: item 10 is not in PAST EXPERIMENTS.
- P2 PASS: dependency on item 9 is satisfied.
- P3 PASS: this can be decomposed into many single-block proxy runs with incremental logging.
- P4 PASS: no prompt/controller edits required.

# Steps
1. Reuse the existing proxy path: `hf_pretrained` `gpt2` teacher plus `data/shakespeare.txt` as clearly labeled proxy corpus.
2. Use the new experiment runner `experiments/nanochat_single_block_localization.py` to:
   - evaluate teacher sanity once
   - sweep single layers with `layer_ids=(layer_id,)`
   - run both `geometric` and `distilled` methods
   - record local metrics and full-model metrics after each swap
3. Start with budgets `hidden_fraction in {0.25, 0.125}` so the main ladder includes `>=4x` FFN compression.
4. Sweep all `gpt2` FFN blocks, writing incremental records to `runs/nanochat_single_block_localization_overnight/results.jsonl` and heartbeat state to `runs/nanochat_single_block_localization_overnight/heartbeat.json`.
5. Leave the sweep running continuously by repeating passes with incremented distillation seeds.
6. Once a meaningful slice is complete, summarize the results in `autoresearch/results/cycle_005_summary.md`.
7. Update `autoresearch/agenda.md`: move item 10 from NEXT to PAST EXPERIMENTS with outcome, commit hash, and artifact paths.
8. Commit with message: `[autoresearch] cycle 005: single-block swap localization curve`
9. Write `autoresearch/results/cycle_005_manifest.txt` with item number, outcome, key metric, commit hash, and artifact paths.

# Validation
- Each record must include layer id, budget, method, teacher baseline, full-model metrics, and local block metrics.
- The overnight run must append records incrementally so partial progress survives interruption.
- The final verdict must be based only on `>=4x` analytic swaps versus the distilled baseline.

# Artifacts
- `autoresearch/briefs/cycle_005.md`
- `autoresearch/plans/cycle_005.md`
- `runs/nanochat_single_block_localization_overnight/results.jsonl`
- `runs/nanochat_single_block_localization_overnight/heartbeat.json`
- future: `autoresearch/results/cycle_005_summary.md`

# Fallbacks
- If full all-layer sweeps are too slow, restrict to a depth-stratified subset first and keep the loop running over that subset.
- If a particular combination crashes, log the error record and continue with the next layer-budget-method combination.
- If `gpt2` loading fails unexpectedly, fall back to cached/random smoke only for debugging, but do not count that as cycle completion.
