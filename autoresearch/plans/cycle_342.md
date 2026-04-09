# Plan — Cycle 342

**Item**: 351 — LASER with output-covariance gating on pgolf — re-evaluate
**Type**: empirical

## Step 0: Read cross-project reference
Read `autoresearch/cross_project/parameter_golf.md` (done in brief).

## Step 1: Create qat-side wrapper script

Create `experiments/pgolf_laser_retrain_cycle342.py` — adapted from cycle 327/340 wrapper with:
- CYCLE = 342
- Configs: rank=5 at 2000 steps lr=3e-5, rank=10 at 2000 steps lr=3e-5
- Also rank=5 at 2000 steps lr=1e-4 (match old lr, more steps) as a control
- Reuse existing `parameter-golf/local_scripts/pgolf_laser_retrain.py` which already supports --n-train-steps and --lr

## Step 2: Execute

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/pgolf_laser_retrain_cycle342.py`

Expected outputs:
- `autoresearch/results/cycle_342_pgolf_laser_retrain.json`
- `autoresearch/results/cycle_342_summary.md`

## Step 3: Evaluate against criteria

- If any config has BPB delta < 0.005: CONFIRMED — longer retrain rescues LASER on pgolf b3
- If all configs have BPB delta > 0.01: REFUTED — even 4x budget can't fix it, eff_rank 4.89 isn't concentrated enough
- Otherwise: INCONCLUSIVE

## Step 4: Bookkeeping

- Move item 351 from NEXT to PAST in agenda.md
- Write manifest
- Append experiments.tsv row
- Commit

## Verification

Compare cycle 342 results to cycle 327 results:
- Same baseline BPB (should be ~1.362)
- rank=5 cycle 327 delta=+0.038 — does 2000 steps at 3e-5 improve on this?

## Time estimate

Each rank config takes ~3-5 min (2000 steps). 3 configs = ~10-15 min total.
