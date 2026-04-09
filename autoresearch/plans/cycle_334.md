# Cycle 334 Plan — Item 328: Extended budget rank-1 retrain at b11 (8000 steps)

## What
Extend cycle 311's b11 rank-1 LASER retrain from 4000 to 8000 steps with LR decay, to settle whether b11's borderline 59.71 PPL is a plateau or still descending.

## Script
Create `experiments/b11_extended_budget_cycle334.py` (~250 lines, single file). Adapts `experiments/rank1_all_concentrators_cycle311.py` for single-block extended run.

### Imports & Config (~30 lines)
Reuse cycle 311 structure. Key changes from cycle 311:

```python
BLOCK = 11                          # single block (was BLOCKS = [2,3,10,11])
RETRAIN_RANK = 1
N_STEPS = 8000                      # was 4000
LR_PHASE1 = 1e-4                    # steps 0-4000
LR_PHASE2 = 5e-5                    # steps 4000-8000
LR_SWITCH_STEP = 4000
EVAL_EVERY = 100
SEEDS = [42, 137, 2024]
N_CALIB = 200
N_EVAL = 100
SEQ_LEN = 128
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Cross-validation references from cycle 311
CYCLE_311_B11_INIT_PPL = 60.6103
CYCLE_311_B11_BEST_MEAN = 59.7074
CYCLE_311_B11_PER_SEED_BEST = {42: 59.7154, 137: 59.6968, 2024: 59.7099}
FLOAT_BASELINE_REF = 53.8949

# Verdict
CONFIRMED_LE = 58.0
REFUTED_GE = 60.0
```

### Step 1: Load model + data (~30 lines)
Copy verbatim from cycle 311: GPT-2 small, wikitext-2 val, tokenizer, calib/eval split. Same eval_ppl function.

### Step 2: Float baseline + SVD (~30 lines)
1. Eval float PPL, assert within 0.5 of FLOAT_BASELINE_REF.
2. SVD of b11 c_proj.weight (3072, 768). Compute A = U[:,:1]*S[:1], B_init = Vh[:1,:], bias from c_proj.bias.
3. No-retrain r=1 PPL. Assert matches CYCLE_311_B11_INIT_PPL within 0.01.

### Step 3: LowRankCProj (~25 lines)
Copy LowRankCProj class verbatim from cycle 311. No changes — same HF Conv1D semantics (x @ (A @ B) + bias).

### Step 4: Training loop with LR decay (~80 lines)
For each seed in {42, 137, 2024}:
1. Load fresh GPT-2, install LowRankCProj at block 11
2. Freeze all params, unfreeze B (1,768) and bias (768). Assert 1536 trainable params.
3. Optimizer: Adam with LR_PHASE1
4. Training loop:
   ```python
   for step in range(1, N_STEPS + 1):
       # LR decay at midpoint
       if step == LR_SWITCH_STEP + 1:
           for pg in optimizer.param_groups:
               pg['lr'] = LR_PHASE2
           print(f"  LR decayed to {LR_PHASE2} at step {step}")
       # ... train step (same as cycle 311) ...
   ```
5. Eval every 100 steps. Track best_ppl, best_step, full history.
6. Best-state restore and verify.

**Cross-validation at step 4000**: after step 4000, eval PPL. Assert best_ppl at step ≤ 4000 matches CYCLE_311_B11_PER_SEED_BEST[seed] within 0.05 PPL. This confirms the first half of training reproduces cycle 311. If mismatch, print warning but continue.

### Step 5: Aggregate + verdict (~30 lines)
- best_mean_ppl = mean of 3 seeds' best_ppl (over full 8000 steps)
- best_mean_ppl_at_4000 = mean of 3 seeds' best_ppl restricted to steps ≤ 4000 (should match 59.7074)
- trajectory_last_10 = last 10 eval points per seed (for plateau analysis)
- Verdict: ≤ 58 → CONFIRMED-robust-r1; ≥ 60 → REFUTED-borderline-artifact; else INCONCLUSIVE

### Step 6: Output (~40 lines)
- JSON: `autoresearch/results/cycle_334_b11_extended.json`
  ```
  {
    "cycle": 334, "item": 328,
    "verdict": "...",
    "best_mean_ppl": ...,
    "best_mean_ppl_at_4000": ...,
    "per_seed": {seed: {best_ppl, best_step, best_ppl_at_4000, best_step_at_4000, history}},
    "cross_validation": {seed: {cycle311_ref, measured_at_4000, delta}},
    "float_ppl": ..., "init_ppl": ...,
    "config": {...}
  }
  ```
- Summary markdown: `autoresearch/results/cycle_334_summary.md`
- Manifest: `autoresearch/results/cycle_334_manifest.txt`

## Sanity Checks
- Float PPL matches FLOAT_BASELINE_REF (53.89) within 0.5
- Init PPL (no-retrain r=1) matches cycle 311 value (60.61) within 0.01
- Step-4000 best PPL per seed matches cycle 311 within 0.05 (same data, same optimizer, deterministic)
- best_mean_ppl_at_4000 matches 59.7074 within 0.05
- After LR decay at step 4001, optimizer LR is 5e-5
- Trainable params = 1,536 per seed

## Risks
- Trajectory slope at step 4000 is only -0.02 to -0.10 ppl/1000 steps. LR halving will slow this further. Reaching ≤ 58 requires ~1.7 ppl drop in 4000 steps at lower LR — unlikely based on slope extrapolation. INCONCLUSIVE is the expected outcome.
- MPS non-determinism: Adam on MPS may not exactly reproduce cycle 311 step-by-step. Allow 0.05 PPL tolerance at step-4000 cross-validation.

## Time Budget
- Model load + SVD + no-retrain: ~30 sec
- 3 seeds × 8000 steps × ~0.1 sec/step: ~40 min total
- Conservative estimate: 30-45 min wall time
