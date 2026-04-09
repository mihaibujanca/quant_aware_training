# Research Status -- 364 Cycles (001-364)

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
| 12-block | 186.38 | Total collapse |
| Distillation baseline (matched params) | 56.38 | Geometric wins by +10.16 PPL |

### Composition Barrier (the #1 problem)
3 blocks is the ceiling. Going from 3 to 6 compressed blocks causes a catastrophic PPL jump (~18 PPL) regardless of:
- Training schedule (joint, sequential, curriculum -- joint is best, c359)
- Progressive compression with intermediate full-model adaptation (c361)
- Cross-block error decorrelation (errors are already orthogonal, mean cosine=0.016, c362)
- Knowledge distillation polish (KD overwrites geometric structure, c363)

The failure mode is **superlinear error accumulation** despite spectrally flat per-block errors. Per-block Frobenius norms vary 4.5x (b1=1385 to b11=6293, c362), suggesting uniform k across blocks is wasteful.

## Confirmed Findings (cycles 322-364)

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

## Dead Directions (cycles 322-364)

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

## Open Questions

1. **Why does error accumulate superlinearly past 3 blocks?** Per-block errors are orthogonal and spectrally flat, yet 6+ blocks collapse. Is it the cumulative Frobenius norm? The 4.5x norm variation (c362) suggests some blocks contribute disproportionately.
2. **Can adaptive k per block equalize error norms?** Give high-norm blocks more channels (larger k) and low-norm blocks fewer. Direct intervention for the composition barrier.
3. **Which blocks matter least?** Block importance pruning could identify blocks that can be skipped or compressed more aggressively.
4. **Where in the residual stream does error become unrecoverable?** Monitoring error through the full forward pass (not just per-block) could reveal the failure point.
5. **Accumulation order**: item 360 (easiest-first vs hardest-first) remains untested on GPT-2.
