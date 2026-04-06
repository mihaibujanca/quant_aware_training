# Task

Item 243 (P1): **Alternative 6-block selections — is BLOCKS=[1,2,4,5,7,8] representative?**
Cycle 221's 59.5 PPL ceiling uses BLOCKS=[1,2,4,5,7,8] but the block subset was inherited from cycle 211 without justification vs other 6-subsets. Sweep three alternative 6-block subsets: (a) [1,2,3,4,5,6] contiguous early, (b) [3,4,5,6,7,8] contiguous middle, (c) [1,3,5,7,8,10] spread. K=512, joint Adam W_out+b_out, LR=1e-4, 4000 steps + best-state restore, 1 seed (SEED=42) per subset to save time, then rerun the winner with 3 seeds if it beats 59.0. Exclude blocks 0 and 11 per cycles 198/201 (structural uniqueness). Tests whether the 59.5 ceiling is a block-subset artifact or a structural 6-block ceiling.

# Hypothesis

The 59.5 PPL ceiling from cycle 221 is structural to "any 6 blocks" rather than specific to [1,2,4,5,7,8]. Alternative subsets (contiguous early/middle, spread) will all land in [59, 62].

# Pre-flight Results

- `experiments/adam_6block_subset_sweep_cycle233.py`: does not exist (clean).
- `autoresearch/results/cycle_233_*`: no prior results (clean).
- Template `experiments/adam_6block_earlystop_cycle221.py` exists and verified.

# Steps

1. Create `experiments/adam_6block_subset_sweep_cycle233.py` based on `adam_6block_earlystop_cycle221.py`.
2. Set `SUBSETS = [[1,2,3,4,5,6], [3,4,5,6,7,8], [1,3,5,7,8,10]]`. Remove top-level `BLOCKS`, pass subset to a `run_subset(blocks, seeds)` wrapper that calls the existing `run_seed` logic with `BLOCKS` parameterised.
3. Phase 1: `SEEDS_P1=[42]`, `N_STEPS=4000`, `EVAL_EVERY=100`, `K=512`, `N_CALIB=200`, `SEQ_LEN=128`, `LR=1e-4`, best-state snapshot/restore. Run all 3 subsets sequentially; capture per-subset `init_ppl`, `best_ppl`, `best_step`, `final_ppl_at_step_4000`, `history`.
4. Phase 2: after Phase 1, if any subset's `best_ppl < 59.0`, rerun that subset with `SEEDS_P2=[42, 137, 2024]` and aggregate mean/std (reuse seed 42 result).
5. Verdict per item 243: CONFIRMED if any subset's (Phase 2 if run, else Phase 1) best ≤ 58.0; REFUTED if all three ∈ [59.0, 62.0]; else INCONCLUSIVE.
6. Output JSON to `autoresearch/results/cycle_233_adam_6block_subset_sweep.json` and summary markdown to `autoresearch/results/cycle_233_summary.md`.

# Validation

- Assert `abs(final_ppl_restored - best_ppl) < 1e-3` after best-state restore per seed (match cycle 221).
- Float PPL computed once up-front; sanity check close to 29.9.
- Log per-subset Phase 1 best to stdout before Phase 2 gating.
- Cross-reference cycle 221 [1,2,4,5,7,8] best_mean=59.515 as implicit 4th data point in the summary (do not rerun).

# Artifacts

- `experiments/adam_6block_subset_sweep_cycle233.py` (new)
- `autoresearch/results/cycle_233_adam_6block_subset_sweep.json`
- `autoresearch/results/cycle_233_summary.md`

# Fallbacks

- If any subset OOMs on MPS during calibration hook collection: reduce `N_CALIB=150` for that subset and record in JSON.
- If Phase 1 seed 42 runs show PPL divergence (>100): restore best, record, mark that subset INCONCLUSIVE for Phase 2 gating.
- If the 3,4,5,6,7,8 middle subset triggers the cycle 228 gelu_noneg pathology: record init_ppl only and note separately; do not retry.

# Update bookkeeping

- Append cycle 233 row to `autoresearch/results/cycle_233_manifest.txt`.
- Move item 243 from NEXT to PAST in `autoresearch/agenda.md` with verdict.
- Add diary entry summarising best subset and whether 6-block ceiling is structural or selection-dependent.
