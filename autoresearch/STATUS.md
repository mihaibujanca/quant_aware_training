# Research Status — 199 Cycles (001–199)

## Latest: Cycle 199

209. **K-sweep for near-rank-1-input blocks** — REFUTED
   - Degradation vs float: +57% at K=8, +47% at K=32, +26% at K=512. No plateau. Rank-1 residual stream (blocks 4,7,11 from cycle 198) does NOT enable smaller K. After LayerNorm, each token has distinct 768D variation regardless of residual rank. Block input rank ≠ MLP input diversity after LN.

## Cycle 198

205. **Block 0 criticality analysis — what makes block 0 unique** — CONFIRMED
   - Block 0 input_rank_90=269 vs median_others=1 (269×). Raw token embeddings span 269 dimensions; later blocks operate on near-rank-1 residual stream. Input rank collapse: b0=269→b1=207→b2=25→b3+=1. Block 0 is irreplaceable because it transforms high-diversity embeddings into the residual stream. Cannot be compressed by the same topk+lstsq approach as blocks 1-11 (which operate on rank-1 input).

## Cycle 197

208. **Block-pair joint compression via sequential progressive fitting** — INCONCLUSIVE
   - Sequential fitting (each block on compressed predecessor's I/O): 29.954 PPL vs all-float 30.089 PPL. Improvement 0.135 PPL (noise-level). Distribution mismatch in fitting data is NOT the cascade bottleneck. Cascade is from intrinsic information loss per block (K=512 cannot represent 3072), not from training/inference distribution mismatch.

## Cycle 196

207. **LayerNorm recalibration for stacked compressed blocks** — REFUTED
   - Compressed PPL 30.089 (blocks [1,4,7,11], K=512). LN-2 recal → 30.337 (WORSE by 0.248). Per-channel cross-token statistics irrelevant to LN's per-token normalization — LN normalizes each token independently over 768 channels. Cascade barrier is in residual CONTENT (correlational structure), not distributional shift. LN direction closed.

## Cycle 195

206. **Jacobian effective rank census — data-active rank of GPT-2 FFN blocks** — REFUTED
   - overall_mean_rank_90=472.4/768 (61%). 0/12 blocks below 100/768 low-rank threshold. Blocks 1-11: mean 387-543. Block 0 lower (166.8/768=22%) but not exploitable. Rank gradient: low early → high middle (b9 peaks at 543) → slight drop b11 (459). Stacked ranks 359-642. R3 MASO active-subspace hypothesis REFUTED: GPT-2 FFN blocks are near-full-rank. Consistent with cycles 184/189 (diffuse high-rank error structure).

## Cycle 194

204. **Self-correction depth analysis — blocks 8-10 partial compensation** — REFUTED
   - max_decrease=2/12 layers. Error magnitude generally keeps growing; PPL improvement at k=8-10 is error REDIRECTION to low-PPL subspace (error rotates into less PPL-sensitive dimensions), not cancellation. Self-correction is passive, not active. Cascade barrier cannot be broken by relying on self-correction.

## Cycle 193

203. **Superadditive interaction isolation — C3+Hybrid combined** — INCONCLUSIVE
   - C3+Hybrid all-12 PPL=41.447 (+82.59%) vs ReLU+bias ~42.6 (+88%). Nearly identical. Superadditivity is GENERAL — not specific to zero-point. Simple sum prediction +48% vs actual +83%: 35-point excess from combining both-regime modifications.

## Cycle 192

202. **Exact dead-channel pruning baseline — GPT-2 FFN** — REFUTED
   - Near-zero dead channels at any viable threshold (eps=0.01: 0.003%; eps=1.0: 2.64% but PPL +80%). GELU networks don't have exact zeros — Serra/Berzins exact pruning is not applicable. Only 2 always-active channels total. R2 exact pruning mandate is closed for GELU architectures.

## Cycle 191

201. **Mean residual error correction — directional coherence-based bias** — REFUTED
   - Corrected PPL=56.781 vs uncorrected=44.842 (WORSENED by 11.938). Mean error vectors (fraction_of_frob 0.13-0.35) are non-trivial but applying as constant residual shifts hurts downstream layers. Cascade barrier is fundamental: any constant modification to the residual stream compounds through subsequent layers.

## Cycle 190

200. **Positive-regime shape isolation — GELU-neg + ReLU-pos hybrid activation** — INCONCLUSIVE
   - Hybrid all-12 PPL=28.980 (+27.67%, baseline 22.700) — just above the refuted threshold (28). Positive-regime shape change alone (+27.67%) costs nearly as much as C3 negative-regime replacement (+20.33%). Full ReLU+bias swap (+88%) implies ~40% superadditive interaction when both regimes are modified simultaneously.
   - Non-monotone progressive: PPL peaks at k=7 (28.484), decreases to k=10 (27.851), then rises at k=11 (28.980) — some deep-layer self-correction.

## Cycle 189

199. **Cascade error direction analysis — GELU vs ReLU+bias residual stream** — CONFIRMED
   - Mean cos(e_l, e_{l-1})=0.716; all 11/11 inter-layer pairs exceed 0.3. Errors compound directionally: ||e||_F grows 685→11472 (17x over 12 layers). Rank_90 stays high (320-553/768) — errors are diffuse in content but directionally coherent. Block 11 anomaly: largest ||e||_F jump + rank_90 drops to 320 (consistent with cycle 184).
   - Reconciles with cycle 184 (diffuse rank, low-rank correction not viable): both are true. Errors have structured direction despite high rank.

## Cycle 188

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
11. **Cascade errors are directionally coherent despite high rank** (189): mean cos(e_l,e_{l-1})=0.716, 11/11 layers. Errors are diffuse in content (rank_90=300-550) but aligned in direction. 17x Frobenius amplification over 12 layers. Not low-rank correctable (cycle 184), but directionally structured.
12. **Both activation error types cost equally; interaction is superadditive** (190): positive-regime shape change alone +27.67%, negative-regime constant replacement alone (C3) +20.33%. Full modification (ReLU+bias) +88% — ~40% extra from superadditive interaction between both types simultaneously. Neither type dominates.
13. **Residual stream corrections are counterproductive without downstream re-calibration** (191): first-moment correction worsens PPL by 12 PPL. Cascade barrier fundamental — any modification to residual stream at depth l compounds through subsequent layers.
14. **Exact dead-channel pruning not applicable to GELU networks** (192): GPT-2 has near-zero dead channels (0.003% at eps=0.01). GELU never reaches exactly zero. Serra/Berzins exact pruning requires ReLU hard zeros. R2 baseline is closed.
15. **Superadditivity is general — not specific to zero-point** (193): C3+Hybrid (non-zero neg replacement) PPL=41.4 ≈ ReLU+bias PPL=42.6. Combining both-regime modifications produces ~35% extra damage beyond simple sum, regardless of the specific form of negative-regime replacement.
16. **Self-correction at k=8-10 is passive error rotation, not cancellation** (194): max 2/12 layers show magnitude decrease. PPL improves because errors rotate into low-PPL subspace (consistent with diffuse high-rank structure from cycle 184). Self-correction is a statistical artifact, not exploitable.
17. **FFN Jacobians are near-full-rank — no active subspace** (195): overall_mean_rank_90=472.4/768 (61%). 0/12 blocks low-rank (<100). Rank gradient: b0=166 → b9=543 → b11=459. Stacked ranks 359-642. R3 MASO active-subspace hypothesis fully refuted. Consistent with cycles 184/189 (diffuse errors). GPT-2 FFN functions are globally distributed across nearly all 768 dimensions.
18. **LN recalibration is ineffective** (196): LN-2 recal worsens PPL by 0.248 (30.089→30.337). Per-channel cross-token statistics don't match what per-token LN normalization needs. Cascade error lives in normalized residual content, not distributional shift.
19. **Sequential fitting provides negligible benefit** (197): fitting each block on compressed predecessors' I/O gains 0.135 PPL (noise-level). Distribution mismatch in fitting data is NOT the cascade bottleneck — cascade is from intrinsic information loss per compressed block.
20. **Block 0 criticality is from high input diversity, not specialization** (198): block 0 input_rank_90=269 vs median=1 for blocks 2-11. Input rank collapse: 269→207→25→1 across blocks 0-3. Block 0 must transform high-dimensional raw embeddings (50257-type vocabulary) into the residual stream. Cannot be dropped or replaced by later-block compression approach.
21. **Rank-1 residual stream does NOT reduce required K** (199): K-sweep for BLOCKS=[1,4,7,11] shows monotone improvement +57%→+26% above float (K=8→512) with no plateau. LayerNorm creates token-specific 768D variation after per-token centering/scaling; block input rank ≠ MLP input diversity. K=512 remains necessary baseline.

### Working Recipes (ranked)
1. **topk+lstsq + corrections** -> PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** -> PPL ~70.7 (1.58M params)

### Exhausted Directions
Channel selection, joint training (5 refutations), dead channel pruning (5 refutations), block-adaptive ratio, correction transfer, lstsq targets, sign-pattern partitioning, SoLA split, negative absorption, spectral selection, FLAP bias, wider-shallower, full-model ReLU swap, iterative bias recomputation, low-rank residual correction, compression ratio sweep (ratio-independent floor), fewer-block compression (cascade barrier persists at 2 blocks), Jacobian active subspace compression (R3 MASO hypothesis refuted — near-full-rank Jacobians), LN recalibration (per-channel stats irrelevant to per-token LN)

## Next
- Item 210: Residual stream rank trajectory under compression (P0)
- Item 211: Block 11 geometry deep-dive (P1)
- Item 212: Rank-1 input FFN compression (P2)
