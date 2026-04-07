# Cycle 269 — Item 279

## Goal
K=1 6-block extended budget (16000 steps) — settle whether cycle 240's 96.886 catastrophe is budget-limited or structural.

## Hypothesis
cycle 240's plateau is structural, not budget-limited → best_mean ≥ 90.0 (≤ 7 PPL improvement over cycle 240's 96.886 at 4× the steps).

## Setup
Trivial N_STEPS=4000→16000 + EVAL_EVERY=100→400 swap on cycle 240 template. New script `experiments/k1_6block_16k_cycle269.py`. BLOCKS=[1,2,4,5,7,8]; K=1; W_out+b_out trainable joint Adam LR=1e-4; N_STEPS=16000; EVAL_EVERY=400 (40 eval points, same density as cycle 240); best-state restore; 3 seeds {42,137,2024}; disjoint calib=all_ids[:200]/eval=all_ids[200:300]; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = 9,216 = 6 × (768 + 768).

## Verdict thresholds (item 279 verbatim)
CONFIRMED (structural) if best_mean ≥ 90.0; REFUTED (budget-limited) if best_mean ≤ 80.0; INCONCLUSIVE if best_mean ∈ (80, 90).

## Baselines
cycle 240 K=1 6-block 4000-step 96.886; cycle 230 K=1 3-block 59.245; cycle 254 K=128 5-block 8000-step noop; cycle 221 K=512 6-block 59.515; cycle 231 K=128 6-block 67.562; float 53.895.

## Run cap
~25 minutes (4× cycle 240's ~6 min).

## Artifacts
`experiments/k1_6block_16k_cycle269.py`, `autoresearch/results/cycle_269_k1_6block_16k.json`, `autoresearch/results/cycle_269_summary.md`, `autoresearch/logs/cycle_269_run.log`, `autoresearch/results/cycle_269_manifest.txt`.

## Verification
JSON+log present; verdict matches threshold logic; restore matches recorded best (script-internal assert).
