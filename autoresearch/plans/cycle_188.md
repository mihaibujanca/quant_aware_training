# Cycle 188 — Item 198: Progressive C3 composition across all 12 blocks

## Task

Test whether C3 (per-channel conditional mean replacement in negative GELU regime) composes across all 12 GPT-2 blocks. Cycle 173 confirmed C3 costs +1.4% max per block individually. This tests if stacking 12 independent C3 hooks produces acceptable PPL — without the GELU→ReLU shape change that destroyed cycles 181/182.

## Hypothesis

C3 replaces only the constant part of GELU's negative output (token-dependent mask, but constant value per channel). If composition fails, the failure mechanism is different from shape-change-based cascade. If composition succeeds (<+15%), the shape change in cycles 181/182 was the sole cause of failure.

## Pre-flight

- **P1 Duplicate**: No prior all-12-block C3 test exists.
- **P2 Dependencies**: None. Forks cycle 173 calibration code.
- **P3 Concrete**: 1 calibration pass + 12 progressive PPL evaluations = 13 forward passes.
- **P5 Null baseline**: Measured fresh (cycle 173 reported 22.7 PPL with SEQ_LEN=1024).
- **P6 Model**: GPT-2 pretrained from HuggingFace.

## Criteria

- **Confirmed** if: all-12 C3 PPL < baseline * 1.15 (within +15%)
- **Refuted** if: all-12 C3 PPL > baseline * 1.40 (above +40%)
- **Inconclusive** if: between +15% and +40%

## Steps

### Step 1: Write `experiments/c3_composition_cycle188.py`

Fork `experiments/regime_ablation_table_cycle173.py`. Key changes:
- Remove single-block sweep (C0-C5 per block)
- Add progressive C3 sweep: for k=0..11, accumulate C3 hooks, eval PPL
- Keep same config: SEQ_LEN=1024, EVAL_SEQS=200, BATCH_SIZE=4, SEED=42
- Keep `collect_gelu_stats()` and `ConditionHook` verbatim
- Output: `autoresearch/results/cycle_188_c3_composition.json`

Algorithm:
1. Load GPT-2, load OWT tokens (1024 seq len)
2. Evaluate baseline PPL (sanity check [15, 65])
3. Run calibration (collect mu_neg for all 12 blocks)
4. For k=0..11:
   - Install C3 hook on block k (cumulative: blocks 0..k all hooked)
   - Evaluate PPL
   - Print progress
5. Apply verdict thresholds to step k=11 result
6. Save JSON with progressive PPL curve

### Step 2: Run

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/c3_composition_cycle188.py
```

Expected runtime: ~5-10 min (13 forward passes × 200 sequences × 1024 tokens).

### Step 3: Write results manifest

`autoresearch/results/cycle_188_manifest.txt`

### Step 4: Update agenda.md + STATUS.md

### Step 5: Commit

```bash
git add experiments/c3_composition_cycle188.py autoresearch/results/cycle_188_* autoresearch/plans/cycle_188.md autoresearch/briefs/cycle_188.md autoresearch/agenda.md autoresearch/STATUS.md
git commit -m "[autoresearch] cycle 188: C3 composition all-12 blocks — <VERDICT>"
```

## Validation

- Baseline PPL must be in [15, 65]
- Progressive PPL should be monotone non-decreasing (each added block ≥ previous)
- Block 0 C3 PPL should be within +2% of baseline (cycle 173 result)
- Verdict applied to k=11 (all-12-block) result

## Artifacts

- `experiments/c3_composition_cycle188.py`
- `autoresearch/results/cycle_188_c3_composition.json`
- `autoresearch/results/cycle_188_manifest.txt`
