# Research Status — 181 Cycles (001–181)

## Latest: Cycle 181

191. **Full-model ReLU + per-block bias vector swap — GPT-2** — REFUTED
   - PPL 42.6 (+87.9% vs float 22.7). Single-block within +6% but 12-block cascade severe.
   - Distribution shift: biases precomputed on float model stale for downstream blocks.
   - Item 192 (iterative recomputation) now eligible as follow-up.

## Session 125-181 (57 cycles this session)

### Key Findings
1. **GELU negative regime = constant bias** (173, 174): absorbable in 12/12 blocks; token-dependent signal is random noise
2. **Single-block ReLU+bias works, full-model fails** (173, 181): per-block within +6%, but cascade to +88% when all 12 swapped simultaneously
3. **z negativity is learned, scales with model size** (165, 168): trained Wx_mean=-1 vs random~0; larger models push more into dead zone
4. **Positive regime is the computation carrier** (164): 380x more destructive to kill than negative
5. **Block count is primary cascade cost** (171): 12@3x=183 vs 4@6x=59 PPL
6. **Topk+lstsq+corrections remains best**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** -> PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** -> PPL ~70.7 (1.58M params)

### Exhausted Directions
Channel selection (topk optimal), joint training (5 refutations), dead channel pruning (5 refutations), block-adaptive ratio, correction transfer, lstsq targets, sign-pattern partitioning, SoLA split, negative absorption, spectral selection, FLAP bias, wider-shallower, per-block concentration, full-model precomputed bias swap

## Next
- P0: 192 (iterative bias recomputation — eligible after 181 refuted)
- P1: 193 (distillation control baseline), 194 (compression ratio sweep)
- P2: 195 (fewer-block compression), 196 (layer-drop baseline)
