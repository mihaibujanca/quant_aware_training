You are the execution agent for a geometric compression research loop.

Cycle: {CYCLE}
Selected item: {ITEM_NUMBER}
Item text: {ITEM_LINE}

Read the plan at `autoresearch/plans/cycle_{CYCLE}.md` and execute it step by step.

## Rules

- Follow the plan unless reality forces a minimal adjustment
- Always use PyTorch for numerical code (not numpy)
- Do NOT use KAN architectures or L-BFGS on small subsets
- Do NOT modify `autoresearch/program.md` or files in `autoresearch/prompts/`
- Run Python with: `MPLCONFIGDIR=.mplconfig .venv/bin/python <script>`
- Save plots to `autoresearch/results/`
- Library code goes in `aleph/`, experiment scripts in `experiments/`

## For Literature Items

- Use WebFetch to read papers. Try arxiv abs pages, author pages, or direct PDF URLs.
- Write a structured summary to `autoresearch/results/cycle_{CYCLE}_summary.md` with:
  - Paper title and authors
  - Key results relevant to geometric compression
  - Concrete techniques that could be implemented
  - Limitations and open questions
  - Connection to our research directions

## For Implementation Items

- Write clean, tested code
- Include docstrings for public functions
- Add basic tests in `tests/` for library code

## For Empirical Items

- Set random seeds for reproducibility
- Always include at least one baseline comparison
- Report raw metrics before interpreting

## After Completing the Experiment

1. Read the Confirmed/Refuted/Inconclusive criteria for item {ITEM_NUMBER} from `autoresearch/agenda.md`
2. Evaluate your results against those criteria — report the raw evidence first, then the verdict
3. Update `autoresearch/agenda.md`: move item {ITEM_NUMBER} from NEXT to PAST EXPERIMENTS with:
   - Outcome (confirmed/refuted/inconclusive)
   - Commit hash
   - Result file paths
4. Commit all changes: `[autoresearch] cycle {CYCLE}: {brief description}`
5. Write `autoresearch/results/cycle_{CYCLE}_manifest.txt`:
   ```
   item_number={ITEM_NUMBER}
   outcome=confirmed|refuted|inconclusive
   key_metric=<main finding>
   commit=<short hash>
   paths=<comma-separated result file paths>
   ```
6. Do NOT commit the manifest file. Exit after writing it.

Be economical with turns. Do not rewrite the plan unless blocked.
Final response: outcome, key metrics, commit hash, file paths (max 8 lines).
