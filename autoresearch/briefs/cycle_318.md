# Item
333 (P0): Output-covariance eff_rank census on parameter-golf 13L_640d ReLU² checkpoint.
First cross-project measurement. Per cross_project doc, two-script subprocess pattern.

# Context
- Cycle 278: GPT-2 eff_rank profile {b1: 1.94, b2: 1.05, b5: 179, b9: 195, b11: 22.49}.
- Open: does the rank-collapse generalize to ReLU² + parameter-constrained model?

# Dependencies
- Cross-project doc: `autoresearch/cross_project/parameter_golf.md` (READ).
- Parameter-golf venv at `/Users/mihai/Projects/parameter-golf/.venv/bin/python`.
- Checkpoint: `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz`.

# Key Files
- **`autoresearch/cross_project/parameter_golf.md`** (REQUIRED READING — DONE).
- `parameter-golf/local_scripts/pgolf_eff_rank_census.py` (TO CREATE per doc section 5).
- `experiments/pgolf_eff_rank_census_cycle318.py` (TO CREATE per doc section 3).
- `parameter-golf/local_scripts/analyze_checkpoint.py` (existing, has load_model + load_val_tokens).

# Constraints
- Per item 333 verbatim:
  - Step 0: cat the cross-project doc (DONE in this brief).
  - Step 1: write pgolf-side script at parameter-golf/local_scripts/pgolf_eff_rank_census.py per doc section 5.
  - Step 2: write qat-side wrapper at experiments/pgolf_eff_rank_census_cycle318.py per doc section 3.
  - Step 3: run with `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/pgolf_eff_rank_census_cycle318.py`.
  - Step 4: build cycle_318_summary.md with per-block table and verdict.
- Trainable params: 0.
- 25,600 tokens × seq_len 1024 (per doc section 5).
- Cross-cycle comparison to GPT-2 cycle 278.

# Verdict (item 333)
- CONFIRMED-collapse-generalizes: any block eff_rank ≤ 20.
- CONFIRMED-no-slack: all 13 blocks eff_rank ≥ 100.
- CONFIRMED-different-shape: some blocks ≤20, some ≥100.
- INCONCLUSIVE: all blocks in (20, 100).

# Handoff
Plan agent: per cross-project doc requirements. Note: cross-project items 333+ are now part of the queue (the previous cycle 303 PLAN INVALID was about item 313 which lacked the cross_project doc; this item has the doc). New script names: experiments/pgolf_eff_rank_census_cycle318.py and parameter-golf/local_scripts/pgolf_eff_rank_census.py.
