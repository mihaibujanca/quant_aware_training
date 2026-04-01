# Research Status — 183 Cycles (001–183)

## Latest: Cycle 183

193. **Matched-budget distillation control baseline — GPT-2 FFN k=4** — CONFIRMED
   - Local Adam+MSE distillation (cached block I/O, up to 2000 steps) vs topk+lstsq+polish: stacked PPL 80.7 vs 59.4 (+21.3 PPL, 3 seeds).
   - Single-block PPL nearly identical (+0.3-3.3 PPL), but stacking diverges by +21.3 PPL. Geometric compression preserves cross-block compositional alignment that Adam-from-scratch does not.
   - Verdict: CONFIRMED — topk+lstsq+polish beats distillation by ≥3 PPL. Geometric compression justified.

## Session 125-183 (58 cycles this session)

### Key Findings
1. **GELU negative regime = constant bias** (173, 174): absorbable in 12/12 blocks; token-dependent signal is random noise
2. **Single-block ReLU+bias works, full-model fails** (173, 181): per-block within +6%, but cascade to +88% when all 12 swapped simultaneously
3. **z negativity is learned, scales with model size** (165, 168): trained Wx_mean=-1 vs random~0; larger models push more into dead zone
4. **Positive regime is the computation carrier** (164): 380x more destructive to kill than negative
5. **Block count is primary cascade cost** (171): 12@3x=183 vs 4@6x=59 PPL
6. **Topk+lstsq+corrections remains best**: PPL ~59 at 1.6M + 49K params
7. **Distillation control confirmed** (183): topk+lstsq+polish beats local Adam+MSE distillation by +21 PPL. Topk preserves compositional geometry that iterative optimization from scratch does not. Single-block quality ≠ stacking quality.

### Working Recipes (ranked)
1. **topk+lstsq + corrections** -> PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** -> PPL ~70.7 (1.58M params)

### Exhausted Directions
Channel selection (topk optimal), joint training (5 refutations), dead channel pruning (5 refutations), block-adaptive ratio, correction transfer, lstsq targets, sign-pattern partitioning, SoLA split, negative absorption, spectral selection, FLAP bias, wider-shallower, per-block concentration, full-model precomputed bias swap, iterative bias recomputation (inconclusive — 5.6 PPL gain, not enough)

## Next
- P1: 194 (compression ratio sweep k={2,4,6,8})
- P2: 195 (fewer-block compression), 196 (layer-drop baseline)
