# Research Status — 185 Cycles (001–185)

## Latest: Cycle 185

194. **Compression ratio sweep k={2,4,6,8} — GPT-2 FFN** — REFUTED
   - PPL floor is ratio-independent: k=2→61.56, k=4→59.42, k=6→58.30, k=8→57.16 (only 4.4 PPL for 4x more params, 12x→3x compression). k=8 PPL=57.2 > 55 threshold.
   - Per-PPL gain from doubling params is ~1.1 PPL/step (k=4→6→8), suggesting composition-driven floor not capacity-driven.
   - Cascade/composition barrier confirmed again: increasing channel count does not break the stacking floor.

## Cycle 184

197. **Residual stream trajectory divergence: GELU vs ReLU+bias — GPT-2** — REFUTED
   - Divergence is exponential (R²=0.972) and α-smooth (monotone PPL 57.7→38.1 as α: ReLU→GELU), but DIFFUSE (rank_90=300-434 out of 768). Low-rank correction not viable.
   - Transition-zone fraction identical (~0.49) for most- and least-divergent tokens — activation |z|<1 does NOT predict which tokens diverge.
   - Block 11 anomaly: rank_90 drops from 434→215 at final block (possible self-correction).

## Session 125-184 (59 cycles this session)

### Key Findings
1. **GELU negative regime = constant bias** (173, 174): absorbable; token-dependent signal is noise
2. **Single-block ReLU+bias works, full-model fails** (173, 181): per-block +6%, cascade +88%
3. **z negativity is learned, scales with model size** (165, 168)
4. **Block count is primary cascade cost** (171): 12@3x=183 vs 4@6x=59 PPL
5. **Topk+lstsq+corrections remains best**: PPL ~59 at 1.6M + 49K params
6. **Distillation control confirmed** (183): topk beats local Adam+MSE by +21 PPL; single-block quality ≠ stacking quality
7. **Divergence is exponential but diffuse** (184): rank_90=300-434/768, low-rank correction not viable; α-smooth interpolation confirms shape-mismatch as continuous, not phase-transition

### Working Recipes (ranked)
1. **topk+lstsq + corrections** -> PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** -> PPL ~70.7 (1.58M params)

### Exhausted Directions
Channel selection, joint training (5 refutations), dead channel pruning (5 refutations), block-adaptive ratio, correction transfer, lstsq targets, sign-pattern partitioning, SoLA split, negative absorption, spectral selection, FLAP bias, wider-shallower, full-model ReLU swap, iterative bias recomputation, low-rank residual correction

## Next
- P2: 195 (fewer-block compression), 196 (layer-drop baseline)
