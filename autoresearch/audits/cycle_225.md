# Cycle 225 Audit — Item 235 (K=64/32/16 Adam compression cliff sweep)

**Status: PASS**

## Verdict check
- Criteria: any K best_mean ≤ 53.0 → CONFIRMED; all K ≥ 60.0 → REFUTED; else INCONCLUSIVE.
- Reported: K=64 best_mean = **50.781 ± 0.021** (seeds 50.763 / 50.770 / 50.810).
  K=32 = 53.038 ± 0.009; K=16 = 55.133 ± 0.010. Float = 53.895.
- 50.781 ≤ 53.0 → CONFIRMED. K=32 (53.038) also clears float by 0.86 PPL — independent confirmation. Verdict matches.

## Implementation review
- `experiments/adam_lower_k_cliff_cycle225.py` reuses `CompressedMLP` / `fit_block_lstsq` /
  `eval_ppl` from cycle 223 verbatim. K appears in `scores.topk(K).indices` and trainable
  shape `W_out=(768,K)` — different K genuinely selects different channel subsets.
- W_fc, b_fc are buffers; only `W_out` and `b_out` for blocks {1,4,7} unfrozen
  (lines 142-148). Trainable counts: 149,760 (K=64), 76,032 (K=32), 39,168 (K=16).
- Best-state restore round-trips with `assert abs(restored - best) < 1e-3` (line 196).
- 3 seeds {42, 137, 2024}; std 0.009-0.021 across K — extremely tight, identical init per K.

## Calibration / eval disjointness
- Same disjoint slicing as cycle 223 (`all_ids[:200]` calib, `all_ids[200:300]` eval).
  Adam loss over calib only. No eval leakage.

## Extended efficiency curve
| K   | params  | best_mean | %vs float | Δ vs K=512 |
|-----|---------|-----------|-----------|------------|
| 512 | 1.18M   | 46.556    | −13.62%   | 0          |
| 256 | 592K    | 47.509    | −11.85%   | +0.953     |
| 128 | 297K    | 48.989    | −9.10%    | +2.433     |
| 64  | 150K    | 50.781    | −5.78%    | +4.225     |
| 32  | 76K     | 53.038    | −1.59%    | +6.482     |
| 16  | 39K     | 55.133    | +2.30%    | +8.577     |

Monotone, ~1.5-2.2 PPL per halving. 32× W_out compression (512→16) costs +8.58 PPL vs K=512 and only +1.24 PPL vs float. K=64 is the new free operating point; cliff lies between K=32 and K=16.

## Caveats
- "Beats float" is calibration-aligned W_out fine-tuning, not pure compression — same caveat carried forward from cycles 206/210/223.
- Even K=16 init (68.978) is high relative to its trained best (55.133): Adam absorbs ~14 PPL of post-lstsq error per seed. The "channel capacity" framing is moot when the channels are followed by a learned 768×K projection.

## Bookkeeping
- Manifest, exec_result, JSON, summary, plan all consistent (item=235, cycle=225, outcome=confirmed, numbers match).
- Agenda entry moved to PAST/Cycle 225 with full numbers.

## Recommendations
- Test K∈{8, 4, 2} for the absolute floor (2 octaves below K=16).
- Or: try to erase the K=16 +2.3% gap with more Adam steps, W_fc unfreeze, or KD signal.
- Held-out test set (separate from calib AND eval) for any future "beats float" claim.
