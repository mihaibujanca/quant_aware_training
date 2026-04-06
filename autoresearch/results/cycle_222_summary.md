# Cycle 222 — Frobenius-weighted channel selection (item 227)

**Verdict: REFUTED**

Frobenius-importance channel selection improves 3-block cascade init PPL by only
0.111 PPL — well below the 0.3 PPL refutation threshold.

## Setup

- BLOCKS=[1,4,7], K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, GPT-2 small.
- Same calibration / eval split as cycle 206 (wikitext-2-raw-v1 validation).
- Two scoring rules feeding the same topk + lstsq W_out init pipeline:
  - **A (baseline, cycle 206)**: `score[c] = act_std[c] * ||W_proj[c]||`
  - **B (Frobenius importance)**: `score[c] = ||W_proj[c]||^2 * sum_t gelu(z[t,c])^2`
- Eval metric: full-model PPL after replacing blocks [1,4,7] (no Adam correction).

## Results

| Variant                       | Init PPL          | Δ vs A_mean    |
|-------------------------------|-------------------|----------------|
| Float baseline                | 53.895            | —              |
| A: topk_act_std (3 seeds)     | 61.7464 ± 0.0000  | 0 (anchor)     |
| B: frobenius_importance       | 61.6355           | **+0.111**     |

- Cycle 206 reproduction delta: 0.0004 PPL (within ±0.1 tolerance).
- Channel-set differences B vs A (out of 512 per block): block 1 → 66, block 4 → 52,
  block 7 → 29. Scoring path is exercised — selections genuinely differ — but lstsq
  W_out compensates.
- Variant A std is effectively zero across seeds because the only randomness in this
  pipeline is `torch.manual_seed`, which has no effect on either the deterministic
  calibration sampling or the lstsq fit. The 3-seed roll is reported per the plan.

## Verdict mapping

- Confirmed: best alternative ≤60.75 (Δ ≥1.0). Not met (B = 61.64).
- Refuted: Δ < 0.3 PPL. **Met (Δ = 0.111).**
- Inconclusive: 0.3 ≤ Δ < 1.0. Not met.

## Plain-language conclusion

Reweighting channel selection by Frobenius output contribution (Σ gelu² · ||W_proj||²
instead of std · ||W_proj||) does **not** help the cascade. It picks a measurably
different subset (29-66 channel swaps per block) but the lstsq W_out fit absorbs
the difference, so init PPL barely moves. Cycle 207 already showed the same on
single-block PPL; cycle 222 confirms it carries through to the 3-block cascade.

The cascade gap (61.75 vs float 53.89) is **not** caused by suboptimal channel
selection at K=512. It is structural (cycles 209/214) and is unlocked only by
gradient-based W_out correction (cycle 206 → 48.11 PPL, 10.7% below float).

## Artifacts

- `experiments/frobenius_selection_cycle222.py`
- `autoresearch/results/cycle_222_frobenius_selection.json`
- `autoresearch/results/cycle_222_summary.md`
