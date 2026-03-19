# Cycle 058 Plan: Multi-block Stacking Error Budget — GPT-2 FFN

## Pre-flight
- P1 (duplicate): Cycle 036 ran progressive stacking k=1..4 at 6x on Shakespeare. Novel here: (a) OpenWebText corpus, (b) per-block single-block baselines to isolate interaction effects, (c) residual-stream MSE decomposition per insertion point.
- P2 (dependencies): Cycle 057: single-block OWT 6x = sub-baseline PPL (negative deltas), stacked 3-block = +20.2%. Cycle 054: k=4@6x = +48%.
- P3 (concrete): Primary metric: marginal cost curve shape (convex = compounding, concave/linear = benign). See §Success Criteria.
- P4 (feasible): 4 single-block + 4 cumulative stacked configs x 3 seeds = 24 evals + 4 MSE hooks. ~30 min.
- P5 (null baseline): If errors are additive, marginal PPL cost per added block should be flat. Cycle 057 shows single blocks help but 3-block stacking costs +20%, suggesting marginals accelerate.
- P6 (alignment): Same model, blocks, ratio, probe budget as cycles 056-057.
- P7 (power): 3 seeds; 4 stacking points suffice to detect convexity via second differences.

## Hypothesis
Marginal PPL cost accelerates as more blocks are stacked (convex marginal curve), consistent with super-additive compounding through the residual stream. Predicted: marginal_k4 > 2× marginal_k1.

## Method
Create `experiments/stacking_error_budget_gpt2.py`. Fork from `failure_decomposition_gpt2.py`.

1. **Single-block baselines**: For each block in [7, 5, 4, 1], compress alone at 6x, eval OWT PPL. Record delta_i = PPL_single_i − PPL_baseline.
2. **Progressive stacking** in easiest-first order [7, 5, 4, 1]:
   - k=1: block 7; k=2: 7+5; k=3: 7+5+4; k=4: 7+5+4+1
   - Record PPL_k and marginal_k = PPL_k − PPL_{k−1}.
3. **Residual-stream MSE**: Hook each compressed block's output, compute MSE(float, compressed) over 4096 probe tokens. Report per-block and cumulative.
4. **Metrics** (in priority order):
   - **Primary — marginal cost convexity**: Fit marginals m_1..m_4. If m_k is increasing (second differences > 0), compounding is super-additive. If flat/decreasing, sub-additive.
   - **Secondary — ratio**: R = stacked_delta_k / sum(single_deltas_1..k). Only reported when denominator > 0 (all single deltas positive). If any single delta ≤ 0, ratio is marked ill-defined and excluded from conclusions.

## Key Parameters
- `target_blocks = [7, 5, 4, 1]`, `hidden_fraction = 1/6` (6x)
- `max_tokens = 4096`, `probe_seqs = 200`, `eval_seqs = 200`, `seq_len = 128`
- Seeds = [42, 123, 7]. OWT corpus from `data/openwebtext_sample.txt`.

## Output Files
- `experiments/stacking_error_budget_gpt2.py`
- `autoresearch/results/cycle_058_stacking_budget.json` — singles, stacked, marginals, MSE, convexity stats
- `autoresearch/results/cycle_058_stacking_budget.png` — dual-axis: marginal PPL cost (bars) + cumulative residual MSE (line)

## Success Criteria
- **Confirmed (super-additive)**: Marginal costs strictly increasing (all second differences > 0, mean > 1 PPL point), AND marginal_k4 > 2× marginal_k1.
- **Refuted (sub-additive)**: Marginals flat or decreasing (second differences ≤ 0).
- **Inconclusive**: Mixed-sign second differences or marginal acceleration < 1 PPL point (within noise).

## Risks
- Single-block deltas may be negative (cycle 057 precedent). This invalidates the ratio metric but NOT the marginal convexity analysis, which depends only on stacked PPL progression.
- Block 1 may behave differently from 4/5/7. Its single-block baseline runs first to catch surprises.
