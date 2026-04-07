# Cycle 294 Audit — Item 304

**Audit status: PASS**

## Scope
Verify cycle 294 execution against item 304's verbatim Hypothesis and verdict thresholds; verify arithmetic and cross-check against the raw JSON, against cycle 276 as the direct template, and against cycle 287 as the K=512 extension precedent.

## Verdict verification

Item 304 thresholds (verbatim from cycle 294 brief / original agenda):
- CONFIRMED iff ≥ 2 of 4 K-tiers shift by ≥ 0.5 PPL (cycle 276 4k best_mean − cycle 294 8k best_mean ≥ 0.5)
- REFUTED iff all 4 K-tiers shift by ≤ 0.1 PPL
- INCONCLUSIVE otherwise

Measured (from JSON `aggregates`):

| K   | cyc 276 4k best_mean | cyc 294 8k best_mean | shift   |
|-----|----------------------|----------------------|---------|
| 32  | 49.64854             | 47.72583             | +1.9227 |
| 64  | 48.04525             | 45.67821             | +2.3670 |
| 128 | 46.93161             | 44.65550             | +2.2761 |
| 256 | 45.38537             | 43.48650             | +1.8989 |

Arithmetic (independent recomputation):
- 49.64854 − 47.72583 = +1.92271 ✓ (matches reported +1.92275 within 4e-5 rounding)
- 48.04525 − 45.67821 = +2.36704 ✓ (matches +2.36704)
- 46.93161 − 44.65550 = +2.27611 ✓ (matches +2.27611)
- 45.38537 − 43.48650 = +1.89887 ✓ (matches +1.89887)

Threshold evaluation:
- ≥ 0.5 PPL shifts: K=32 (+1.92), K=64 (+2.37), K=128 (+2.28), K=256 (+1.90) → 4/4 clear the bar
- Required ≥ 2 of 4 → CONFIRMED clause triggered (4 ≥ 2) ✓
- All ≤ 0.1: FALSE (all 4 shifts exceed 1.8 PPL) → REFUTED clause NOT triggered ✓
- → **CONFIRMED** ✓ (matches manifest, exec_result, JSON)

## Per-seed and aggregate sanity

### K=32 (25,344 params)
Seeds (from JSON `k_results.32`):
- 42   : init 55.6239 → best 47.7445 @ step 8000 → final@8000 47.7445 → restored 47.7445
- 137  : init 55.6239 → best 47.6944 @ step 8000 → final@8000 47.6944 → restored 47.6944
- 2024 : init 55.6239 → best 47.7385 @ step 8000 → final@8000 47.7385 → restored 47.7385

Aggregate:
- mean(47.7445, 47.6944, 47.7385) = 143.1774/3 = 47.7258 ✓ (matches 47.72583)
- std (population): deviations (+0.0187, -0.0314, +0.0127), sq (3.50e-4, 9.86e-4, 1.61e-4), sum=1.497e-3, /3=4.99e-4, sqrt=0.02234 ✓ (matches 0.02233)
- best_min=47.6944 (seed 137) ✓; best_max=47.7445 (seed 42) ✓

### K=64 (49,920 params)
Seeds:
- 42   : init 55.5249 → best 45.6628 @ step 8000
- 137  : init 55.5249 → best 45.6829 @ step 8000
- 2024 : init 55.5249 → best 45.6889 @ step 8000

Aggregate:
- mean = 137.0346/3 = 45.67820 ✓ (matches 45.67821)
- std: dev (−0.0154, +0.0047, +0.0107), sq (2.37e-4, 2.21e-5, 1.14e-4), sum=3.74e-4, /3=1.25e-4, sqrt=0.01116 ✓ (matches 0.01117)

### K=128 (99,072 params)
Seeds:
- 42   : init 55.2727 → best 44.6300 @ step 8000
- 137  : init 55.2727 → best 44.6748 @ step 8000
- 2024 : init 55.2727 → best 44.6617 @ step 8000

Aggregate:
- mean = 133.9665/3 = 44.65550 ✓
- std: dev (−0.0255, +0.0193, +0.0062), sq (6.50e-4, 3.72e-4, 3.84e-5), sum=1.061e-3, /3=3.536e-4, sqrt=0.01880 ✓ (matches 0.01880)

### K=256 (197,376 params)
Seeds:
- 42   : init 54.9026 → best 43.5178 @ step 8000
- 137  : init 54.9026 → best 43.4534 @ step 8000
- 2024 : init 54.9026 → best 43.4883 @ step 8000

Aggregate:
- mean = 130.4595/3 = 43.48650 ✓
- std: dev (+0.0313, −0.0331, +0.0018), sq (9.80e-4, 1.096e-3, 3.24e-6), sum=2.079e-3, /3=6.93e-4, sqrt=0.02633 ✓ (matches 0.02633)

All 12 fits: best at step 8000 (all BUDGET-BOUNDED at the 8000-step cap; floor not yet found).
All 12 fits: |restored − best| = 0 (exact restore) ✓
Per-K init PPLs deterministic across seeds (lstsq + topk init is seed-free, only varies by K) ✓

## Cross-check vs cycle 276 (step-4000 evals, matched-config reproducibility)

The critical scientific cross-check: at identical seed/init/LR, the cycle 294 eval at step 4000 must match the cycle 276 best_ppl (which was also at step 4000 — cycle 276 was budget-bounded).

From JSON `k_results.*.[].eval_at_step_4000` vs `cycle_276_step_4000_ref` (patched post-run with correct per-seed references from cycle 276's k_results):

| K   | seed | cyc 294 step-4000  | cyc 276 best_ppl   | \|Δ\|     |
|-----|------|--------------------|--------------------|----------|
| 32  | 42   | 49.623039          | 49.623039          | 0.00e+00 |
| 32  | 137  | 49.683719          | 49.683719          | 0.00e+00 |
| 32  | 2024 | 49.638872          | 49.638872          | 0.00e+00 |
| 64  | 42   | 48.024281          | 48.024281          | 0.00e+00 |
| 64  | 137  | 48.100182          | 48.100182          | 0.00e+00 |
| 64  | 2024 | 48.011290          | 48.011290          | 0.00e+00 |
| 128 | 42   | 46.865833          | 46.865833          | 0.00e+00 |
| 128 | 137  | 47.020823          | 47.020823          | 0.00e+00 |
| 128 | 2024 | 46.908171          | 46.908171          | 0.00e+00 |
| 256 | 42   | 45.403186          | 45.403179          | 6.93e-06 |
| 256 | 137  | 45.427799          | 45.427796          | 2.92e-06 |
| 256 | 2024 | 45.325150          | 45.325146          | 4.00e-06 |

Max |Δ| = 6.93e-06, all 12 cells within 1e-4 tolerance (declared in the plan).

Nine of twelve cells are bitwise-identical (|Δ|=0.00e+00). Three K=256 cells show 3–7e-06 deltas, consistent with MPS nondeterminism on identical operations (cycle 290's audit recorded the same 1e-5 floor for matched-config reruns on the larger K=512 setup). The 4k→8k PPL gains are therefore unambiguously Adam budget, not init or LR drift.

**Script patch note**: The original script's `CYCLE_276_BEST_PPL_PER_SEED_4K` dict had wrong per-seed assignments (values transposed between seeds within some K groups). This only affected the raw JSON's `cycle_276_step_4000_ref` and `step_4000_delta` fields as originally written — not the experiment data (`eval_at_step_4000`, `best_ppl`, `init_ppl`, `history`). The JSON was patched post-run with the correct per-seed references pulled directly from cycle 276's raw `k_results`. The `patches` field in the JSON records the fix with `max_delta_after_patch = 6.93e-06`. Auditor independently verified the post-patch cross-check table by recomputing `eval_at_step_4000 - c276_lookup[K][seed]` from the patched JSON.

## Config / plan adherence

Plan (cycle_294.md) vs execution (verified from JSON `config` and `k_results`):
- BLOCKS=[1] ✓
- K_VALUES=[32, 64, 128, 256] ✓
- N_STEPS=8000 ✓ (was 4000 in cycle 276)
- EVAL_EVERY=200 ✓ (was 100 in cycle 276)
- SEEDS=[42, 137, 2024] ✓
- LR=1e-4 ✓
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128 ✓ (disjoint)
- MPS, GPT-2 small, wikitext-2 val ✓
- Trainable per fit asserted = {32: 25344, 64: 49920, 128: 99072, 256: 197376} = K×768 + 768 ✓
- Float PPL 53.89489152418256, matches reference 53.895 to 5e-06 ✓
- Per-seed best-state restore: |restored − best| = 0 for all 12 fits ✓
- Template: `experiments/one_block_k_sweep_cycle276.py` adapted with only N_STEPS, EVAL_EVERY, output paths, baseline constants, verdict thresholds, wall-time cap raised to 60 min. Verified structural correspondence (same CompressedMLP class, same collect_mlp_io/fit_block_lstsq/eval_ppl, same training loop, same restore/assert logic) ✓

## Baseline cross-checks

| baseline                         | recorded in JSON | expected     | match |
|----------------------------------|------------------|--------------|-------|
| float_ppl                        | 53.89489         | 53.895       | ✓     |
| cycle 271 [1] K=512 4k           | 43.82045         | 43.82045     | ✓     |
| cycle 287 [1] K=512 8k           | 42.78533         | 42.78533     | ✓     |
| cycle 276 4k best_mean K=32      | 49.64854         | 49.64854     | ✓     |
| cycle 276 4k best_mean K=64      | 48.04525         | 48.04525     | ✓     |
| cycle 276 4k best_mean K=128     | 46.93161         | 46.93161     | ✓     |
| cycle 276 4k best_mean K=256     | 45.38537         | 45.38537     | ✓     |

Delta cross-checks (independent recomputation):
- K=256: 43.48650 − 42.78533 = +0.70117 (cycle 294 K=256 8k − cycle 287 K=512 8k) ✓ — 0.70 PPL worse at 2× fewer params
- K=128: 44.65550 − 42.78533 = +1.87017 ✓ — 1.87 PPL worse at 4× fewer params
- K=64:  45.67821 − 42.78533 = +2.89288 ✓ — 2.89 PPL worse at 8× fewer params
- K=32:  47.72583 − 42.78533 = +4.94050 ✓ — 4.94 PPL worse at 16× fewer params
- Budget-extension family: [1] K=512 4k→8k = −1.035 (cycle 287); [1] K=32 = −1.923; K=64 = −2.367; K=128 = −2.276; K=256 = −1.899 — all four cycle 294 extensions beat cycle 287's K=512 extension in magnitude ✓

## Pareto reading — audit

Cycle 294 (4 new points) vs cycle 287 [1] K=512 8k at 394K params:

| cycle | K   | params  | PPL    | vs cyc 287        |
|-------|-----|---------|--------|-------------------|
| 294   | 32  | 25,344  | 47.726 | +4.94, 16× fewer  |
| 294   | 64  | 49,920  | 45.678 | +2.89, 8× fewer   |
| 294   | 128 | 99,072  | 44.656 | +1.87, 4× fewer   |
| 294   | 256 | 197,376 | 43.487 | +0.70, 2× fewer   |
| 287   | 512 | 393,984 | 42.785 | (anchor)          |

Every cycle 294 point strictly dominates its cycle 276 4k counterpart at the same param count (same K) by the measured shift. Every cycle 294 point is non-dominated vs cycle 287 (different param tier, different PPL). Four new Pareto frontier anchors on the 1-block [1] channel-K curve.

Important: the K=256 anchor (197K, 43.49) is the most interesting — at half the params of cycle 287, it trades only 0.70 PPL, the best small-K-vs-large-K tradeoff seen in the channel-K family so far. This may deserve a "is the Pareto curve convex/concave in this region" follow-up.

## Geometric reading — audit

(i) **Hypothesis shape result**: the brief predicted 0.5–1.5 PPL shifts, with the largest gain at the smallest K (K=32) because "Adam slack is largest when trainable param count is smallest." The measured shifts are:
- K=32: +1.92 (below mid-tier)
- K=64: +2.37 (LARGEST)
- K=128: +2.28 (second)
- K=256: +1.90 (smallest)

The "smaller K → larger gain" framing is FALSIFIED in shape — K=64 and K=128 gain more than K=32. The geometric interpretation: K=32 has the most Adam slack but also the tightest representational ceiling (rank-32 topk subspace cannot capture enough MLP behavior to benefit from 4k more Adam steps). K=256 has the most capacity but the least slack (it's closest to its saturation floor). K=64 and K=128 sit in the sweet spot: enough capacity to benefit from extra Adam, enough slack to do so.

(ii) **All shifts exceed 0.5 PPL by 3.8×–4.7×**. The brief's threshold of 0.5 was conservative; the actual gains are large enough that the budget extension is unambiguously worthwhile across the entire small-K tier. Cycle 276's floor was a budget artifact, not a structural floor.

(iii) **All 12 fits at 8k are still budget-bounded**. No fit found its floor within 8000 Adam steps. A 16k extension is a natural follow-up at all K-tiers, not just K=32 or K=64. Cycle 269 showed [K=1] 5-block 4k→16k gave -5.27 PPL, so 4×-budget gains on small-K tiles can be much larger than 2×-budget gains.

(iv) **Pareto curve rewrite**: cycle 294 gives four new strictly-dominant points on the 1-block [1] channel-K Pareto curve, each dominating the cycle 276 4k point at the same K. The canonical 1-block [1] channel-K small-K reference is now cycle 294, not cycle 276.

(v) **Cross-check vs per-K [2]-beats-[1] trend**: cycle 283 measured [2] K∈{16,32,64} winning over [1] at small K under 4k budget. Cycle 294 is [1]-only, so it doesn't directly resolve whether [2] would also benefit from 8k at these K values. The natural follow-up is [2] K∈{32, 64, 128, 256} 8k extension (item ??? — not in current NEXT).

(vi) **Run quality**: 12/12 fits completed without seed drop, exact restore for all 12, init PPL deterministic per K, step-4000 cross-check 9/12 bitwise-identical and 3/12 within 7e-06 (MPS floor). This is a clean reproducibility run on a trivial extension; all the interesting structure is in the aggregate shifts and the Pareto curve update.

## Artifact existence

- experiments/one_block_1_k_sweep_8k_cycle294.py — exists (~340 lines)
- autoresearch/plans/cycle_294.md — exists, 8 H2 headings
- autoresearch/briefs/cycle_294.md — exists (from brief stage)
- autoresearch/results/cycle_294_one_block_1_k_sweep_8k.json — exists, valid JSON, verdict="CONFIRMED", patches field populated
- autoresearch/results/cycle_294_summary.md — exists
- autoresearch/results/cycle_294_manifest.txt — exists, commit=e81924d
- autoresearch/results/cycle_294_exec_result.txt — exists, commit=e81924d
- autoresearch/logs/cycle_294_run.log — exists
- Primary commit e81924d exists in git log
- experiments.tsv row 294 appended with 8 columns ✓
- agenda.md: item 304 deleted from NEXT P2, added to PAST as Cycle 294 ✓

## Corrections

None. Provisional outcome holds. The script's per-seed reference dict bug was caught and patched post-run without affecting the experimental data — the patched JSON cross-check is bit-level consistent with cycle 276's `k_results`, confirming the 4k→8k attribution.

## Final status

**PASS** — keep provisional outcome CONFIRMED. No agenda fix required beyond the move from NEXT P2 to PAST already executed. Manifest, exec_result, JSON, summary, agenda, TSV, and audit all agree on verdict = CONFIRMED with all 4 K-tiers shifting by 1.89–2.37 PPL (4/4 ≥ 0.5 threshold).

Auditor note: cycle 294 supersedes cycle 276 as the canonical 1-block [1] small-K reference. The channel-K Pareto curve at 1 block now has 5 clean anchors (cycle 294 at K=32/64/128/256 + cycle 287 at K=512), all at 8k Adam budget. The K=256 anchor at 197K/43.49 is the most interesting new frontier point — it halves the param count of cycle 287 at only +0.70 PPL, the best small-K-vs-large-K tradeoff in the family. The script bug was a dict assignment error in a cross-check field, not an experimental issue; it was detected immediately by the cross-check printout (|Δ|=7.59e-02 on the first K=64 fit was physically implausible for matched-config), patched post-run via direct read of cycle 276's raw k_results, and all 12 cells then passed the 1e-4 tolerance with max delta 6.93e-06. Future extension templates should pull reference values from cycle source JSONs at runtime rather than hand-copying, to avoid this class of bug.
