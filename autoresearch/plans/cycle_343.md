# Cycle 343 Plan — Item 352

## Design
Incremental LASER accumulation on GPT-2 small c_proj weights.

### Accumulation order (most concentrated first)
1. b2 alone (rank 2, eff_rank 1.05)
2. b2 + b1 (rank 2, eff_rank 1.94)
3. b2 + b1 + b10 (rank 17, eff_rank 16.70)
4. b2 + b1 + b10 + b11 (rank 23, eff_rank 22.49)

### For each stage
- SVD c_proj.weight → A_frozen (U[:,:r]*S[:r]), B_trainable (Vh[:r,:])
- Install LowRankCProj modules for all blocks in the current set
- Joint Adam retrain over all B-factors + biases, 1000 steps, lr=1e-4
- Evaluate PPL on wikitext-2 validation, 3 seeds
- Track: init_ppl (no retrain), best_ppl, best_step

### Implementation
- File: `experiments/gpt2_multiblock_laser_cycle352.py`
- Reuse LowRankCProj from cycle 305 (proven correct)
- Output: `autoresearch/results/cycle_343_accumulation.json`
- 200 calib seqs, 100 eval seqs, seq_len 128, device mps

### Verdict
Compute from accumulation curve:
- CONFIRMED if any stage with >=2 blocks has best_mean <= 54.90
- REFUTED if 1-block best_mean < float and all >=2-block best_mean > 1-block best_mean
- INCONCLUSIVE otherwise

### Risks
- Cycle 305 showed 5-block LASER degrades to 73 PPL. But that used eff_rank-mismatched ranks.
  Here we use ceil(eff_rank) and build incrementally. Still likely that 4-block will degrade.
- b10/b11 have high eff_rank (17/23), so rank-17/23 LASER may not compress much.
