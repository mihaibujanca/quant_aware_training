# Research Status — 188 Cycles (001–188)

## Latest: Cycle 188

198. **Progressive C3 composition — per-channel mean replacement across all 12 blocks** — INCONCLUSIVE
   - All-12 C3: PPL=27.314 (+20.33%), between +15% and +40% thresholds. Composes much better than ReLU+bias (~42 PPL). Barrier materializes after block 6-7.
   - Shape change accounts for ~14.3 PPL of the 18.9-PPL cycle-181 cascade gap; constant replacement accounts for ~4.6 PPL.

## Cycle 187

196. **Layer-drop baseline — GPT-2 FFN skip blocks [1,4,7,11]** — CONFIRMED
   - 10/12 blocks individually near-redundant (PPL <40 when dropped; damage +1.5–4.3). Block 0 critical (1923 PPL). Block 11 borderline (41.4 PPL).
   - Multi-block [1,4,7,11] drop: 53 PPL (+18). Cascade interactions produce nonlinear superadditivity even for near-free individual drops — confirms cascade barrier is structural, not due to individual block importance.

## Cycle 186

195. **Fewer-block compression — GPT-2 FFN k=4 on 2 blocks vs 4 blocks** — REFUTED
   - Best 2-block pair (1,4): mean PPL 50.4 ±5.3 vs 4-block 59.4 ±8.2. Halving block count gains 9 PPL but stays above threshold 48.
   - All 6 pairs tightly clustered (50.4–54.8); block pair (1,4) consistently best.
   - High cross-seed variance driven by seed-42 teacher PPL 57.6 (vs 48.8/50.4 for other seeds).

## Cycle 185

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
8. **Block count matters, floor persists** (185, 186): ratio-independent compression floor ~57 PPL; 2-block gains 9 PPL over 4-block but threshold unbreached; cascade barrier is structural
9. **Individual blocks near-redundant, cascade is not** (187): blocks 1-10 all droppable (PPL <40); multi-block drop 53 PPL despite single drops being 36-39. Superadditivity is the core problem
10. **C3 composes better than ReLU, still breaks after block 7** (188): +20.33% all-12 vs +88% for ReLU+bias. Shape change is the larger culprit (~14.3 PPL); constant replacement alone causes ~4.6 PPL

### Working Recipes (ranked)
1. **topk+lstsq + corrections** -> PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** -> PPL ~70.7 (1.58M params)

### Exhausted Directions
Channel selection, joint training (5 refutations), dead channel pruning (5 refutations), block-adaptive ratio, correction transfer, lstsq targets, sign-pattern partitioning, SoLA split, negative absorption, spectral selection, FLAP bias, wider-shallower, full-model ReLU swap, iterative bias recomputation, low-rank residual correction, compression ratio sweep (ratio-independent floor), fewer-block compression (cascade barrier persists at 2 blocks)

## Next
- P0: 199 (cascade error direction analysis), 200 (GELU-neg+ReLU-pos hybrid)
