# Cycle 322 — Item 335 Brief

## Item
Multi-checkpoint eff_rank census on 3 parameter-golf polished artifacts — is the b3 collapse run-stable?

## Context
Cycle 318 measured eff_rank on a single 13L_640d checkpoint, finding b3=4.89 as the lowest-rank block (collapse block). The LASER action items (340-342) depend on b3 being the canonical target across runs. This item tests run-stability: does the rank-collapse position shift across different trained models?

## Dependencies
- Cycle 318 baseline profile (b3=4.89, b12=6.94 as collapse blocks)
- Parameter-golf checkpoint infrastructure (load_model, val tokens, forward pass)
- **Checkpoints available**:
  - `13L_640d_322_7k_wd70_batch20k_best.npz` (cycle 318 baseline, 13L)
  - `13L_640d_322_7k_wd70_best.npz` (non-batch20k variant, 13L)
  - `wd70_11L_5x_polished.npz` (11L variant, polished)
  - `10L_640d_multiskip_22k_wd70_polished.npz` (10L variant, polished)

## Key Files
- `autoresearch/cross_project/parameter_golf.md` — REQUIRED READING for subprocess invocation pattern
- `parameter-golf/local_scripts/pgolf_eff_rank_census.py` — script to reuse
- `experiments/pgolf_eff_rank_census_cycle318.py` — template for multi-checkpoint wrapper

## Constraints
- Run cap: ~15 minutes (3 checkpoints × ~5 min each)
- No SVD work, no training — forward pass + eigvalsh only
- Template-based subprocess invocation with qat/pgolf environment isolation

## Handoff
**Setup**: Copy cycle 318 wrapper to `pgolf_eff_rank_census_multi_cycle335.py`, modify to loop over checkpoints.
**Choice**: Select 2 additional checkpoints (one 11L, one 10L, plus the 13L baseline) from available polished artifacts.
**Output**: Build per-block side-by-side table (columns = checkpoints, rows = blocks); highlight lowest-eff_rank block per checkpoint.
**Verdict**: CONFIRMED-stable if lowest-eff_rank block is same/±1 layer across all 3 (b3 is canonical LASER target). REFUTED-checkpoint-specific if ≥3-layer shift. INCONCLUSIVE otherwise.
