# Cycle 256 Review — Item 266

**Decision: APPROVE**

## Summary
Trivial K swap on cycle 221. Copy `experiments/adam_6block_earlystop_cycle221.py` → `experiments/six_block_k256_cycle256.py` with K=512→K=256, same BLOCKS=[1,2,4,5,7,8], same LR/steps/seeds/eval cadence. Fills the missing K=256 6-block point between cycle 221 (K=512 59.515) and cycle 231 (K=128 67.562).

## Checks
- Brief matches item 266: BLOCKS=[1,2,4,5,7,8], K=256, W_out+b_out joint Adam, 4000 steps, 3 seeds. PASS
- Hyperparameters (N_CALIB=200, SEQ_LEN=128, LR=1e-4, EVAL_EVERY=100, best-state restore) carried over verbatim. PASS
- Verdict thresholds match brief: ≤62 CONFIRMED, ≥65 REFUTED, else INCONCLUSIVE. PASS
- Disjoint calib/eval split (all_ids[:200]/all_ids[200:300]) inherited from cycle 221 script. PASS
- Trainable param count ≈ 197K/block × 6 = 1.18M, matches cycles 247/249 hetero for head-to-head comparison. PASS
- Output paths namespaced `cycle_256_*` — no collision risk. PASS
- Run cap ~10 min (comparable to cycle 221). PASS

## Risks
None — recipe is the same machinery used in cycles 221/231/240. No novel surface.

## Decision
APPROVE — execute as planned.
