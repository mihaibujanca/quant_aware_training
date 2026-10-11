# Cycle 233 Audit — Item 243

**Status: PASS** — verdict INCONCLUSIVE confirmed.

## Item 243 criteria (verbatim)
- CONFIRMED if any alternative subset yields best_ppl ≤ 58.0
- REFUTED if all three alternative subsets yield best_ppl ∈ [59.0, 62.0]
- INCONCLUSIVE if subsets span > 2 PPL with no clear winner < 59.0

## Phase 1 results (SEED=42, K=512, joint Adam W_out+b_out, 4000 steps)
| subset | init | best | best_step | final@4000 |
|---|---|---|---|---|
| [1,2,3,4,5,6] | 87.631 | 61.785 | 1300 | 76.961 |
| [3,4,5,6,7,8] | 96.758 | 67.826 | 1100 | 86.267 |
| [1,3,5,7,8,10] | 87.237 | 60.981 | 1400 | 75.466 |
| [1,2,4,5,7,8] (c221 ref, 3-seed) | 86.686 | 59.515 ± 0.053 | ~1500 | 72.250 |

## Verdict check
- CONFIRMED (≤58.0)? **No** — min alt = 60.981.
- REFUTED ([59, 62] for all)? **No** — middle subset [3..8] = 67.826 falls outside the band from above by +5.8 PPL.
- INCONCLUSIVE (spread > 2 or no winner < 59)? **Yes** — spread = 67.826 − 60.981 = **6.845 PPL**, and no subset cleared the 59.0 gate.
- → **INCONCLUSIVE per literal criterion.** Consistent with `cycle_233_exec_result.txt` and `cycle_233_manifest.txt`.

## Methodology
- Protocol faithfully matches cycle 221 (K=512, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore). Restore sanity `< 1e-3` enforced via assert in `run_seed`.
- Phase 1 single-seed only — **acceptable per item 243 design**. Phase 2 3-seed rerun was gated on `any best_ppl < 59.0`, which no subset achieved; skipping phase 2 is the correct gate behavior (saves ~20 min compute and cannot change the verdict: the best alt at 60.981 is +2.98 PPL above the 58.0 CONFIRMED bar, much larger than plausible 3-seed variance).
- Float PPL 53.895 unchanged from cycle 221 — eval split reproducible.
- Script `experiments/adam_6block_subset_sweep_cycle233.py` computes verdict correctly (lines 306-313): `any_confirmed`, `all_refuted`, else INCONCLUSIVE.

## Notable finding
Middle-blocks subset [3,4,5,6,7,8] is a clear structural outlier at 67.826 (+8.3 PPL vs c221). Init 96.758 is +10.1 above c221's init — consecutive mid-layer cascade amplifies residual through the (unmodified) attention layers between them. The two non-consecutive-mid subsets land in [60.98, 61.78]; cycle 221's [1,2,4,5,7,8] at 59.515 remains the best-known 6-block selection. The ~59.5 PPL 6-block ceiling is near-structural but mildly selection-sensitive.

## Audit conclusion
Verdict mapping, protocol fidelity, gate logic, and findings summary all internally consistent. **PASS.**
