# Item
254 (P0): Sliding 3-block sweep — where does the cascade penalty bite?
Test [3,4,5], [5,6,7], [6,7,8] at K=512 joint Adam W_out+b_out, 3 seeds. Map ΔPPL vs window start index.

# Context
- Cycle 242: [1,2,3] → 47.135 (Δ +0.58 vs 46.556 baseline). [8,9,10] → 55.192 (Δ +8.64).
- Cycle 243: per-block attribution found block 5 (mid) is the biggest single-block contributor (+5.10) inside the jointly-fit [1,2,4,5,7,8] cascade.
- Open question: where exactly does the cascade penalty turn from "cheap" to "expensive"?

# Dependencies
None.

# Key Files
- experiments/early_vs_late_3block_cycle242.py — direct template, just swap CONFIGS.

# Constraints
- CONFIGS = [[3,4,5], [5,6,7], [6,7,8]]
- K=512, N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100
- 3 seeds (42, 137, 2024)
- Best-state snapshot+restore.
- Confirmed if any window ≤ 48.5 PPL mean.
- Refuted if all ≥ 52.0 PPL mean.
- Inconclusive otherwise.

# Handoff
Plan agent: trivial config swap on cycle 242 template. New script `experiments/sliding_3block_cycle244.py`. Output JSON + summary with all 3 windows and comparison to cycle 242 anchors [1,2,3] (47.135) and [8,9,10] (55.192) plus cycle 210 [1,4,7] (46.556).
