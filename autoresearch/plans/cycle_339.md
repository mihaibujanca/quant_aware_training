# Cycle 339 Plan

## Goal
Zero out b12's MLP output (set mlp_scale = zeros) and measure BPB delta. Extend to other low-scale blocks (b11, b3).

## Step 0: Cross-project setup
Read `autoresearch/cross_project/parameter_golf.md` for subprocess pattern.

## Step 1: Write pgolf-side script
Path: `/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_mlp_prune.py`

Script logic:
1. Load model with `load_model_v2` (from pgolf_laser_retrain.py — handles variable mlp_mult, lrelu2, xsa, bigram)
2. Eval baseline BPB with `eval_bpb` from analyze_checkpoint
3. Print all 13 blocks' mlp_scale stats (mean, abs_mean, num_near_zero)
4. For each pruning config:
   a. Zero b12 only
   b. Zero b11+b12
   c. Zero b3+b11+b12 (b3 post-scale eff_rank = 4.01)
   d. Zero all blocks with mean |mlp_scale| < 0.01
   - Save mlp_scale, set to zeros, eval BPB, restore
5. Print JSON with baseline_bpb and per-config bpb + delta

## Step 2: Write qat-side wrapper
Path: `experiments/pgolf_mlp_prune_cycle339.py`
Subprocess invokes the pgolf script, writes results JSON + summary.

## Step 3: Execute and collect results
Run wrapper, check BPB deltas against thresholds.

## Step 4: Write manifest, move item, commit

## Verification
- Baseline BPB should match known value (~1.57 relu2 baseline from cycle 335)
- Restoring mlp_scale after each config must give identical baseline (sanity)
