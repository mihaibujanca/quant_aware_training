# Research Status -- 374 Cycles (001-374)

## Current State

### Working Recipe
**topk+lstsq channel selection + least-squares output refit + MLP adapter** on GPT-2 FFN blocks.

- **Method**: Select top-k channels (k=512 of 3072, i.e. 1/6) by importance, refit output projection via least-squares, then train a small MLP adapter (ReLU hidden layer) to learn the nonlinear residual.
- **Compression ratio**: 4.79x per block (4.72M -> 985K params including adapter).
- **Why it works**: topk+lstsq produces spectrally flat error (eff_rank=463 vs LASER's 6.14), preventing coherent error accumulation across blocks. The MLP adapter's nonlinearity is essential (LoRA at matched params is 3-5 PPL worse).

### Best Numbers
| Config | PPL | Notes |
|--------|-----|-------|
| Float GPT-2 | 53.89 | Baseline |
| Single-block (b2, h=128 adapter) | 46.22 | **Beats float** by 7.67 PPL |
| 3-block [b2,b1,b10] (h=64 adapter) | 47.43 | Joint training, best schedule |
| 3-block (h=128 adapter) | 48.53 | Diminishing returns from larger adapter |
| 6-block (progressive) | 65.54 | Catastrophic -- composition barrier |
| 6-block (adaptive k) | 67.63 | WORSE than uniform (c365) |
| 12-block | 186.38 | Total collapse |
| Distillation baseline (matched params) | 56.38 | Geometric wins by +10.16 PPL |

### Composition Barrier (the #1 problem)
3 blocks is the ceiling. Going from 3 to 6 compressed blocks causes a catastrophic PPL jump (~18 PPL) regardless of:
- Training schedule (joint, sequential, curriculum -- joint is best, c359)
- Progressive compression with intermediate full-model adaptation (c361)
- Cross-block error decorrelation (errors are already orthogonal, mean cosine=0.016, c362)
- Knowledge distillation polish (KD overwrites geometric structure, c363)
- **Adaptive k per block** (Frobenius-proportional k is WORSE than uniform, c365)
- **Selective compression** skipping b10+b11 (PPL 150 for 10 blocks — WORSE than 6-block 65.54, c370)

**The barrier is about block COUNT, not block identity.** The 6-block set [b2,b1,b10,b4,b3,b11] includes the "worst" blocks yet PPL 65.54. Skipping b10/b11 and compressing 10 other blocks gives PPL 150. Scaling: 3-block=47, 6-block=65, 10-block=150, 12-block=186. Superlinear error accumulation is purely a function of how many blocks are compressed.

## Confirmed Findings (cycles 322-367)

1. **topk+lstsq composes where LASER fails** (c348). On the same GPT-2 blocks in the same order, topk stays below float through 3 blocks; LASER exceeds float at 2 blocks.
2. **Spectral explanation for composition** (c358). LASER error is near-rank-1 (eff_rank=6.14, top-1 energy 78.7%) and coherent with the residual stream (std=7.84). Topk error is spectrally flat (eff_rank=463, top-1 energy 4.7%, residual coherence std=0.88). Rank-1 errors accumulate coherently; flat errors cancel.
3. **Adapter is generic** (c352). Works on ALL tested blocks [b1,b2,b4,b7,b10], not specific to b2. Late blocks benefit more (+2.74 PPL at b10 vs +1.44 at b1).
4. **Nonlinearity is essential** (c360). MLP adapter beats LoRA by 3-5 PPL at matched param counts. The value is in ReLU nonlinearity, not extra linear capacity.
5. **Distillation loses badly** (c356). At matched param budget, KD achieves PPL 56.38 vs geometric 46.22. Geometric compresses AND beats float; KD can't even match float.
6. **4.79x compression ratio** (c353). Per-block FFN 4.72M -> 985K params. Adapter cost = 5% of savings.
7. **Output-covariance concentration** (c341). 6/12 GPT-2 blocks have top-5 output-covariance energy >= 50%. Data-dependent concentration amplifies weight spectrum by 3-14x.
8. **Per-block errors are orthogonal** (c362). Mean off-diagonal |cosine|=0.016. Composition failure is NOT from correlated errors.
9. **Frobenius norms vary 4.5x across blocks** (c362). b1=1385 to b11=6293. Uniform k is suboptimal.
10. **GELU bump rank-1 collapse is activation-specific** (c321). ReLU^2 on same model produces rank 174-443 at mid-stack where GELU-relu is rank-1.
11. **Adaptive k is worse than uniform** (c365). Frobenius-proportional and sqrt-proportional k allocation both degrade 6-block PPL (67.63 and 67.52 vs 66.34 uniform). Early-block degradation propagates more than late-block improvement helps.
12. **Error explosion at blocks 10-11** (c366). Residual stream error propagation shows hockey-stick profile: slow growth blocks 0-9 (cumul error 10→50), explosion at blocks 10-11 (50→72→160). Inflection at block 9 in 3/3 seeds. Attention does NOT amplify error (amp ratio 0.91). b10/b11 account for 77% of total local FFN error.
13. **All GPT-2 FFN blocks are task-critical** (c367). No block is prunable at <0.5 PPL threshold. Block 0 is catastrophic (+3021 PPL). Least important: b1 (+1.60), b7 (+2.35). Most important: b0 (+3021), b11 (+9.12). Importance ranking ascending: [1,7,2,6,9,10,8,3,5,4,11,0].
14. **Composition barrier is count-driven, not identity-driven** (c370). 10-block selective (skip b10+b11) gives PPL 150 — far worse than 6-block uniform (65.54) that includes b10/b11. The number of compressed blocks matters more than which blocks.
15. **Composition barrier is NOT visible in second-order geometry** (c371). eff_rank, participation ratio, PC concentration, mean cosine all fail to distinguish working 3-block from failing 6-block. Max eff_rank drop = 22% (threshold 30%). The 18-PPL gap has no covariance-level signature.
16. **More channels don't break the barrier** (c374). k=768 (3.2x compression) gives 6-block PPL 61.85 vs k=512 (4.79x) 65.54. Only 3.69 PPL gain for 50% more channels. The 3→6 block gap narrows from 18→15 PPL but does not close.

## Dead Directions (cycles 322-367)

| Direction | Verdict | Cycle | Why dead |
|-----------|---------|-------|----------|
| LASER on parameter-golf | REFUTED | 327-329 | All blocks have flat weight spectra (<7% top-5 energy) |
| Full 13-block weight spectrum census (pgolf) | REFUTED | 337 | Max top-5 = 5.32% across all 13 blocks |
| Eff_rank as budget allocation oracle | PLAN INVALID | 331 | Checkpoint-specific (c322), pre-scale misleading (c323) |
| Selective LeakyReLU^2 at b4/b5 | REFUTED | 335 | Gain is distributed across all 13 blocks, not localizable |
| Static rank-2 correction at b4/b5 | PLAN INVALID | 336 | Upper-bounded at 36% recovery (below 50% threshold) |
| Jacobian clustering (pgolf) | REFUTED | 345 | 99.99% unique activation patterns; no clustering structure |
| b12 pruning + LASER composition (pgolf) | PLAN INVALID | 344 | Both dependencies refuted |
| Multi-block LASER accumulation (GPT-2) | REFUTED | 343 | Single-block is the ceiling; 2-block exceeds float |
| 12-block topk+lstsq + adapter | REFUTED | 357 | PPL 186.38; superlinear error accumulation |
| Training schedule (sequential/curriculum) | INCONCLUSIVE | 359 | Joint is BEST; schedule is not the bottleneck |
| Cross-block error decorrelation | REFUTED | 362 | Errors already orthogonal (cosine=0.016) |
| KD fine-tune after geometric compression | REFUTED | 363 | KD overwrites geometric structure; PPL 57-62 vs 48.38 |
| Progressive compression + intermediate retrain | REFUTED | 361 | 6-block still PPL 65.54; adaptation doesn't help |
| Port to nanochat | INCONCLUSIVE | 364 | No checkpoints available locally |
| Adaptive k per block (Frobenius-proportional) | REFUTED | 365 | 67.63 vs 66.34 uniform; early-block degradation dominates |
| Block importance pruning (zero-FFN) | REFUTED | 367 | No block below 0.5 PPL delta; min=1.60 (b1) |
| Mixed compression (importance-based k) | PLAN INVALID | 368 | Both deps refuted; same starvation mechanism as c365 |
| Accumulation order sensitivity (GPT-2) | REFUTED | 369 | <1% PPL gap across 3 orderings; adapter compensates |
| Selective compression (skip b10+b11) | REFUTED | 370 | 10-block PPL 150 >> 6-block 65.54; count matters, not identity |
| Representation geometry collapse | INCONCLUSIVE | 371 | No covariance-level signature of 18-PPL gap; metrics too coarse |
| Importance-weighted k (floor=384) | PLAN INVALID | 372 | Third k-allocation variant; same mechanism as c365 |
| Jacobian probing (R3) | PLAN INVALID | 373 | Duplicate of c068/c195; GPT-2 FFN Jacobians near-full-rank |
| Error budget k=768 on 6-block | REFUTED | 374 | PPL 61.85; 50% more channels only buys 3.69 PPL |

## Open Questions

1. **Why does error accumulate superlinearly past 3 blocks?** PARTIALLY ANSWERED (c366): error grows slowly through blocks 0-9, then explodes at b10/b11 which have the highest Frobenius norms. Attention does not amplify. The remaining question is whether b10/b11 can be compressed better or should be left uncompressed.
2. **Inflection point identified** (c366): block 9 is where error growth transitions from linear to superlinear, consistent across 3 seeds.
3. **No blocks are skip-safe** (c367). All 12 FFN blocks are >0.5 PPL important. But the importance ranking gives a principled basis for mixed k allocation (item 377).
4. **Accumulation order doesn't matter** (c369). <1% PPL gap across 3 orderings with joint adapter training. Toy model finding (27% gap) does not transfer to GPT-2.
5. **Mixed compression strategy** (item 377): after 374 refuted, this needs block importance from 376 rather than Frobenius-proportional k.
