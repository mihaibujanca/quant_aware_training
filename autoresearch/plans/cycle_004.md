# Task
Audit the exact-open and proxy benchmark ladder for transformer FFN compression in this repo, with a filesystem check for local nanochat assets and a public-source check for acquisition procedures.

# Hypothesis
- **Confirmed if:** at least one exact open benchmark and one explicit proxy benchmark are documented with concrete local paths or download procedures, plus a clear "what counts / what does not count" policy.
- **Refuted if:** no practical open teacher+data benchmark can be identified from local assets and public sources.
- **Inconclusive if:** candidate benchmarks exist but local asset availability remains unresolved after the audit.

# Pre-flight Results
- P1 PASS: item 9 is not in PAST EXPERIMENTS.
- P2 PASS: no dependencies.
- P3 PASS: one-cycle analysis item with clear success criteria.
- P4 PASS: no prompt or controller edits required during execution.

# Steps
1. Read item 9 in `autoresearch/agenda.md`, `autoresearch/STATUS.md`, recent `autoresearch/results/`, and `git log --oneline -10` for current autoresearch state.
2. Audit local runnable assets:
   - inspect `aleph/lm_geometric_compression.py`, `experiments/nanochat_ffn_compression.py`, and `docs/nanochat_ffn_compression.md`
   - search `/Users/mihai` for a `nanochat` checkout, `climbmix` parquet directories, and likely checkpoint files
   - check whether Hugging Face `gpt2` is cached locally
   - inspect `runs/nanochat_ffn_compression/gpt2_shakespeare_real.json` as the existing proxy sanity point
3. Audit public primary sources for the exact-open benchmark:
   - open official `karpathy/nanochat` sources for checkpoint loading and dataset acquisition
   - extract the expected local checkpoint path convention, dataset cache/parquet path convention, and any concrete `d24`/`d26` run procedures
4. Write `autoresearch/results/cycle_004_summary.md` with:
   - exact-open benchmark status
   - proxy benchmark status
   - smoke/debug benchmark status
   - concrete local paths and/or acquisition procedure
   - what counts and what does not count
   - blocker list and immediate next item recommendation
5. Evaluate the findings against item 9 criteria and set a provisional verdict.
6. Update `autoresearch/agenda.md`: move item 9 from NEXT to PAST EXPERIMENTS with outcome, cycle `004`, commit placeholder, and artifact paths.
7. Commit cycle-owned files with message: `[autoresearch] cycle 004: benchmark lock-in and asset audit`
8. Write `autoresearch/results/cycle_004_manifest.txt` with item number, outcome, key metric, primary commit hash, and artifact paths.

# Validation
- The summary must explicitly name one exact-open benchmark candidate and one proxy benchmark.
- The exact-open section must say whether it is locally runnable now, not just theoretically possible.
- The proxy section must point to a concrete local artifact or cache path.
- The final verdict must map directly to the agenda criteria, not to general impressions.

# Artifacts
- `autoresearch/briefs/cycle_004.md`
- `autoresearch/plans/cycle_004.md`
- `autoresearch/results/cycle_004_summary.md`
- `autoresearch/results/cycle_004_manifest.txt`
- updated `autoresearch/agenda.md`

# Fallbacks
- If no local nanochat assets are found, document the official acquisition procedure and mark proxy as the only currently runnable benchmark.
- If public sources do not prove `d24`/`d26` availability directly, document the exact open benchmark at the repo/path level and note the gap precisely.
- If the asset audit remains ambiguous, mark the outcome `inconclusive` rather than overstating readiness.
