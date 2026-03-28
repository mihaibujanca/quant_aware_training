# Cycle 076: Mixed-corpus lstsq refit — GPT-2 FFN

## Item
Item 66: Lstsq refit with mixed-corpus probe data. Fix channel selection from Shakespeare, vary the W_down lstsq refit corpus. Evaluate on WikiText-2.

## Context
- Item 64 (cycle 070): Jaccard 0.34-0.58 across corpora. Channel selection NOT stable (0/4 blocks >= 0.7). Item 66 dependency unmet, but mixed refit still informative.
- Cycle 044: Shakespeare->WikiText-2 zero-shot +120% degradation (condition C uncorr). Condition D (Shakespeare channels + WikiText correction refit) reduced to +41%. This experiment isolates the W_down lstsq refit step specifically.
- Pipeline: `probe_ffn_geometry` collects hidden/output samples, `_fit_output_projection` does lstsq W_down refit. Need to separate channel selection from refit.

## Dependencies
- Item 64 (cycle 070): INCONCLUSIVE. Proceed anyway — mixed refit may compensate for unstable selection.

## Key Files
- `experiments/domain_robustness_wikitext.py` — cycle 044 reference (conditions A-D)
- `aleph/lm_geometric_compression.py` — `derive_ffn_compression`, `_fit_output_projection`, `_select_hidden_units`
- `data/{shakespeare.txt,wikitext2_valid.txt,openwebtext_sample.txt,c4_sample.txt}`

## Constraints
- Fix channel selection from Shakespeare for all conditions (isolate refit variable).
- Blocks [7, 5, 4, 1] at 6x. 3 seeds. No post-hoc correction — measure raw compressed PPL.
