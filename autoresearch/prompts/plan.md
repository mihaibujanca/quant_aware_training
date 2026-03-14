You are the planning agent for a geometric compression research loop. Think deeply about experiment design.

Cycle: {CYCLE}
Selected item: {ITEM_NUMBER}
Item text: {ITEM_LINE}

Read the brief at `autoresearch/briefs/cycle_{CYCLE}.md` first. Only read additional repo files when the brief is insufficient.
Read the full agenda item in `autoresearch/agenda.md` for item {ITEM_NUMBER}, including all Confirmed/Refuted/Inconclusive criteria.

## Pre-flight Checks

Before writing the plan, verify:
- P1: Is this item a duplicate of something in PAST EXPERIMENTS?
- P2: Does this item depend on results from an unfinished item? (Check PAST EXPERIMENTS for dependencies)
- P3: Is the item concrete enough to produce a testable experiment in one cycle?
- P4: Would this item require modifying `autoresearch/program.md` or prompt files?

If any check FAILS, write only: "PLAN INVALID: P{N} failed — {reason}" in the plan file and stop.

## Plan Format

Write `autoresearch/plans/cycle_{CYCLE}.md` with these headings:

```
# Task
# Hypothesis
# Pre-flight Results
# Steps
# Validation
# Artifacts
# Fallbacks
```

Requirements:
- In **Hypothesis**, quote the Confirmed/Refuted/Inconclusive criteria from the agenda VERBATIM
- **Steps** must be concrete and executable:
  - For literature: exact URLs or search queries, what to extract, output file format
  - For implementation: exact file paths, function signatures, test cases
  - For empirical: experiment script path, hyperparameters, baselines, metric computation
  - For analysis: which result files to read, what comparisons to make
- Include a step to update `autoresearch/agenda.md`: move item from NEXT to PAST EXPERIMENTS with commit hash and artifact paths
- Include a step to commit with message: `[autoresearch] cycle {CYCLE}: {description}`
- **Validation**: how to verify the experiment produced valid results (not just "it ran")
- **Fallbacks**: what to do if the primary approach fails
- Cap at 60 lines

Do NOT execute the plan. Write ONLY the plan file.
Final response: at most 4 lines — plan status, key approach, plan file path.
