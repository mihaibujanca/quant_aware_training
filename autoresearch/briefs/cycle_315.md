# Item
325 (P0): Calibrated SVD (SVD-LLM) init + Adam retrain at block 1 — does calibrated init improve retrain ceiling?

# Context
- Cycle 280: uncentered SVD init 57.66, retrain 49.03.
- Cycle 313: calibrated SVD no-retrain = 56.70 (slight improvement).
- Open: does calibrated init translate to better retrain floor (< 49.03)?

# Dependencies
- Cycle 313's calibrated SVD code.

# Key Files
- experiments/no_retrain_laser_baseline_cycle313.py — has calibrated SVD.
- experiments/block1_lowrank_cycle280.py — retrain template.

# Constraints
- Block 1 c_proj, r=2, calibrated SVD init.
- Adam retrain B factor + bias, 4000 steps, 3 seeds, best-state restore.
- Disjoint splits.
- Confirmed if best_mean ≤ 48.5 (beats cycle 280's 49.03 by ≥0.5).
- Refuted if best_mean ≥ 49.5.
- Inconclusive otherwise.

# Handoff
Plan agent: combine cycle 313 calibrated SVD + cycle 280 retrain. New script `experiments/calibrated_svd_retrain_cycle315.py`.
