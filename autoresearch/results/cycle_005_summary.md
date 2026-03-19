# Cycle 005: Single-Block Swap Localization Curve

## Verdict

Outcome: confirmed (proxy)

This summary promotes the already-completed proxy sweep in
`runs/nanochat_single_block_localization_overnight_main/results.jsonl`
into the visible `autoresearch/` bookkeeping.

The teacher was cached Hugging Face `gpt2` on the proxy corpus
`data/shakespeare.txt`. The sweep covered all 12 FFN blocks at two
compression budgets (`hidden_fraction in {0.125, 0.0625}`), pairing the
current geometric method against the matched-budget distilled baseline.

Using the latest successful record for each `(layer, hidden_fraction, method)`
pair:
- 24 geometric swaps achieved `>=4x` FFN compression
- 22 of those 24 met the item-10 success condition
- only layer `0` failed, at both `8x` and `16x` compression

The item-10 criterion was:
- relative perplexity increase `<=10%`
- and full-model perplexity no worse than `1.25x` the distilled baseline
- on at least 3 tested swaps at `>=4x` FFN compression

This recovered run exceeds that bar comfortably on the proxy benchmark.

## Setup

- Teacher: `gpt2`
- Dataset label: `proxy_shakespeare`
- Teacher perplexity: `122.350`
- Layers tested: `0..11`
- Budgets: `0.125` and `0.0625`
- Distillation control: `60` steps, seed `42`
- Raw artifacts:
  - `runs/nanochat_single_block_localization_overnight_main/results.jsonl`
  - `runs/nanochat_single_block_localization_overnight_main/meta.json`
  - `runs/nanochat_single_block_localization_overnight_main/heartbeat.json`
  - `autoresearch/results/cycle_005_live_summary.md`

## Key Results

Best proxy geometric swaps so far at `16x` compression:
- layer `7`: relative perplexity `-0.63%`
- layer `5`: relative perplexity `+0.12%`
- layer `4`: relative perplexity `+0.64%`
- layer `1`: relative perplexity `+0.74%`

The two failures were both layer `0`:
- `8x`: relative perplexity `+17.68%`, still better than distilled `+48.79%`
- `16x`: relative perplexity `+33.12%`, still better than distilled `+54.22%`

Representative successful pairs:
- `8x`, layer `1`: geometric `+0.25%` vs distilled `+4.41%`
- `8x`, layer `11`: geometric `+6.73%` vs distilled `+15.08%`
- `16x`, layer `7`: geometric `-0.63%` vs distilled `+2.20%`
- `16x`, layer `10`: geometric `+1.81%` vs distilled `+3.35%`

## Interpretation

The current analytic FFN method is not uniformly broken inside the full model.
It is sharply depth-sensitive:
- layer `0` is a hard failure case
- most middle and late blocks remain viable under `8x-16x` proxy compression

That means the next priority should be accumulation and ordering, not another
round of blanket single-block scouting:
- use item 10's per-layer damage curve to define easy, medium, and hard blocks
- test easiest-first vs hardest-first vs random accumulation
- keep layer `0` as the canonical hard block for follow-up analyses

## Caveats

- This is a proxy result only; it does not answer the exact-open nanochat path.
- The distilled control is a lightweight `60`-step baseline, so later
  baseline-heavy comparisons should rerun representative pairs with a stronger
  control budget.
- The result is recovered from existing raw artifacts rather than a cleanly
  promoted unattended cycle, so the bookkeeping is reconstructed after the fact.
