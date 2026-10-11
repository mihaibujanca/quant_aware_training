# Cycle 223 Audit — Item 228 (Lower K with Adam correction)

**Status: PASS**

## Verdict check
- Criteria: K=256 best_mean ≤ 52.0 → CONFIRMED; > 58.0 → REFUTED; else INCONCLUSIVE.
- Reported: K=256 best_mean = 47.509 ± 0.023 (seeds 47.516 / 47.478 / 47.533).
- 47.509 ≤ 52.0 → CONFIRMED. Verdict matches.

## Implementation review
- `experiments/adam_lower_k_cycle223.py` reuses `CompressedMLP` / `fit_block_lstsq` /
  `eval_ppl` from cycle 221. K appears in `scores.topk(K).indices` and trainable shape
  `W_out=(768,K)` — different K genuinely selects different channel subsets.
- W_fc, b_fc are registered as **buffers**; only `W_out` and `b_out` for blocks {1,4,7}
  are unfrozen (lines 133-141). All other model params frozen — Adam trains 3×(768·K + 768).
- Best-state restore round-trips with `assert abs(restored - best) < 1e-3` (line 188).
- 3 seeds {42, 137, 2024}; std 0.023 (K=256) and 0.033 (K=128) — extremely low variance.

## Calibration / eval disjointness
- `all_ids[:200]` → calib, `all_ids[200:300]` → eval (script lines 52-53). Disjoint.
- Adam loss is over `calib_ids` only (line 165). No eval leakage.
- The "Adam beats float by 11.85%" effect is therefore genuine fine-tuning gain on the
  same wikitext-2 validation distribution, NOT data leakage. Consistent with cycles
  206/210/221 where K=512 + Adam already produced ~−13.6% vs float.

## Compression efficiency curve (this run + cycle 210)
| K   | params (3 blocks) | best_mean | %vs float 53.895 |
|-----|-------------------|-----------|------------------|
| 128 | 297,216           | 48.989    | −9.10%           |
| 256 | 592,128           | 47.509    | −11.85%          |
| 512 | 1,181,952         | 46.556    | −13.62%          |

Curve is monotone but flat: 4× compression (512→128) costs only 5.2% relative PPL.
K=256 absorbs nearly the full K=512 gain (+0.95 PPL gap).

## Caveats
- "Beats float" is largely supervised W_out fine-tuning against the calibration
  distribution, not pure compression. The trained corrections borrow capacity from the
  loss objective. Already noted in cycles 206/210 — flag here for completeness.
- No K∈{64, 32} tested; curve breakpoint unknown. Suggested follow-up in `next` field.

## Bookkeeping
- Manifest, exec_result, JSON, summary, plan all consistent (item=228, cycle=223,
  outcome=confirmed, key metrics match).
- Agenda entry moved to PAST/Cycle 223 with full numbers.

## Recommendations
- Open follow-up to push K∈{64, 32, 16} with same Adam protocol — find the breakpoint.
- Consider a held-out *test* set (separate from calib AND eval) so the "beats float"
  comparison reflects generalization, not just calibration-aligned fine-tuning.
