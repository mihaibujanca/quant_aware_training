# Cycle 006: Stable and Near-Linear Channel Census in Pretrained FFNs

## Verdict

Outcome: refuted (proxy)

This summary recovers the hidden proxy item-11 artifacts into the visible
`autoresearch/` bookkeeping. Two hidden runtime executions both reached the
same proxy-level conclusion on cached `gpt2` with `data/shakespeare.txt`:
- `0/12` tested FFN blocks reached the confirmation threshold of `>=5%`
  stable-or-near-linear channels
- `10/12` blocks stayed below the refutation threshold of `<1%`

That satisfies the agenda's refutation rule on the currently runnable proxy.

## Recovered Sources

Recovered hidden summaries:
- `~/.local/state/autoresearch/quant_aware_training/worktrees/ar-1773533504/autoresearch/results/cycle-2-1773533504_item11_summary.md`
- `~/.local/state/autoresearch/quant_aware_training/worktrees/ar-1773533613/autoresearch/results/cycle_2_item11_summary.md`

Both runs used:
- Teacher: `gpt2`
- Corpus: `data/shakespeare.txt`
- Sequence length: `128`
- Blocks tested: `12`
- FFN hidden width: `3072`

## Key Results

Across the recovered runs:
- blocks with `stable_or_near_linear >= 5%`: `0/12`
- blocks with `stable_or_near_linear < 1%`: `10/12`
- largest observed combined fraction was still only about `1.79%`
- next largest was about `1.43%`
- split robustness was strong rather than noisy:
  - median split delta about `0.00033`
  - max split delta about `0.00163`

The nontrivial mass came almost entirely from always-negative channels in the
earliest layers. Near-linear and always-positive fractions were negligible.

## Caveats

- The two recovered executions used slightly different near-linear definitions,
  but they agreed on the item-level verdict.
- This remains a proxy result only; it does not settle the exact-open nanochat
  path.
- These artifacts were never cleanly promoted by the unattended loop because
  the old supervisor mixed real execution with broken audit and skip logic.

## Interpretation

On the current `gpt2` proxy path, exact stable-sign or near-linear channel
removal does not look large enough to be the main no-SGD FFN compression win.
That does not make exact pruning irrelevant:
- it is still the right exact baseline
- but it looks more like a small first pass than the main compression engine

This recovered result should push the next proxy agenda toward:
- multi-block accumulation using item 10's damage ranking
- merge signals that explain why certain blocks are safe
- exact-prune baselines as controls, not as the expected main win
