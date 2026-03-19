# Cycle 059 Plan: Gradient-Polished lstsq on Stacked Blocks — GPT-2 FFN

## Pre-flight
- P1 (duplicate): No prior cycle tested gradient polish after lstsq. Novel.
- P2 (dependencies): Cycle 057: single-block 6x = sub-baseline PPL. Cycle 058: k=4 stacked = +30% PPL. Polish targets stacking degradation only.
- P3 (concrete): Primary metric: stacked PPL delta before/after gradient polish.
- P4 (feasible): 3 polish variants x 3 seeds on k=4 stacked = 9 runs + 1 baseline. ~45 min.
- P5 (null baseline): Single-block has no degradation (cycle 057). Adapted to stacked k=4.
- P6 (alignment): Same model/blocks/ratio/corpus as cycles 057-058.
- P7 (power): 3 seeds; +30% baseline degradation gives large dynamic range.

## Hypothesis
Adam steps after lstsq refit reduce stacking-induced PPL degradation by compensating for inter-block error compounding. Predicted: 25-50 steps at lr=1e-4 reduces the +30% gap by >=30% relative (to <=+21%).

## Method
Create `experiments/gradient_polish_stacked_gpt2.py`. Fork from `stacking_error_budget_gpt2.py`.

1. **Baseline**: Stack-compress blocks [7,5,4,1] at 6x with topk+lstsq on `probe_train`. Record PPL_stacked.
2. **Polish**: After lstsq on each block, run N Adam steps on that block's `W_down` only:
   - **Data**: Draw polish tokens from `corpus.probe_holdout` (default 32 seqs x 128 = 4096 tokens). This is disjoint from lstsq `probe_train` data and eval data.
   - **Gradient scope**: Before polish loop, set `model.requires_grad_(False)`, then `W_down.requires_grad_(True)`. Only the target W_down column-subset participates in backward.
   - **Optimizer**: Adam(lr=1e-4) on `[W_down]`, cross-entropy loss on polish tokens with full model forward in eval mode.
   - Polish each block sequentially before stacking the next.
3. **Variants**: N in {10, 25, 50}. Seeds [42, 123, 7].
4. **Overfitting check**: Evaluate on `corpus.eval_val` (200 seqs x 128, disjoint from both probe_train and probe_holdout).

## Key Parameters
- `target_blocks = [7, 5, 4, 1]`, `hidden_fraction = 1/6` (6x)
- `polish_steps = [10, 25, 50]`, `polish_lr = 1e-4`
- `polish_data = corpus.probe_holdout` (4096 tokens, disjoint from lstsq/eval)
- `eval_seqs = 200`, `seq_len = 128`, seeds = [42, 123, 7]

## Output Files
- `experiments/gradient_polish_stacked_gpt2.py`
- `autoresearch/results/cycle_059_gradient_polish.json`
- `autoresearch/results/cycle_059_gradient_polish.png` — bar chart: PPL delta by polish steps

## Success Criteria
- **Confirmed**: 50 steps reduce stacked PPL degradation by >=30% relative.
- **Refuted**: <5% improvement across all step counts, or PPL increases (overfitting).
- **Inconclusive**: 5-30% relative improvement — real but insufficient.

## Risks
- Per-block polish ignores inter-block interactions; joint polish out of scope.
- lr=1e-4 may be wrong; if all 3 step counts degrade, try lr=1e-5 as fallback.
