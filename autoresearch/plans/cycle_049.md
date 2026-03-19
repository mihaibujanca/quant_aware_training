# Cycle 049 (rev1) — Item 51: Per-block learned compressor — GPT-2 FFN

## Pre-flight
- P1 (duplicate): Cycle 030 Baseline B already ran per-block distillation (768->384->768, Adam 1e-3, 500 steps) on blocks 8-11. Geometric won all 4. However, cycle 030 used max_tokens=4096 with 591K params (144:1 param-to-sample ratio). This cycle tests whether lifting the data cap changes the outcome.
- P2 (dependencies): Item 14 CONFIRMED, item 18 INCONCLUSIVE. Both available.
- P3 (concrete): Yes.
- P5 (null baseline): Cycle 030 Baseline B numbers are the explicit prior.
- P6 (alignment): Same architecture, same param budget as cycle 030.

## Prior (cycle 030 Baseline B, blocks 8-11, max_tokens=4096)
| Block | Geometric PPL | Per-block distill PPL | Winner |
|-------|--------------|----------------------|--------|
| 8     | 122.53       | 122.94               | Geom   |
| 9     | 122.54       | 122.91               | Geom   |
| 10    | 122.14       | 122.88               | Geom   |
| 11    | 126.03       | 131.86               | Geom   |

## Hypothesis
Cycle 030 Baseline B's per-block distillation failed due to a 144:1 param-to-sample ratio (591K params, 4096 tokens). With 32K tokens (18:1 ratio), the learned compressor has enough data to converge and may beat geometric on blocks where the cycle 030 margin was small (<1 PPL: blocks 8-10).

## Method
1. Load GPT-2, build probe corpus with `probe_train_sequences=256` (32K tokens).
2. For all 12 blocks: geometric baseline (topk k=384 + lstsq) -> PPL.
3. For all 12 blocks: per-block 768->384->768 distillation, 500 steps, Adam 1e-3, max_tokens=32768 -> PPL.
4. Ablation: also run at max_tokens=4096 on blocks 8-11 to reproduce cycle 030 Baseline B (sanity check).

## Key parameters
- `k_channels = 384`, `distill_steps = 500`, `lr = 1e-3` (all match cycle 030)
- `max_tokens = 32768` (lifted from 4096; probe_train_sequences=256)
- Eval: validation set PPL (same eval pipeline as cycle 030)

## Implementation
- New file: `experiments/per_block_compressor_data_scaling.py`
- Reuse `DistillMLP`, `geometric_compress_block`, `collect_block_activations` from cycle 030 code.
- Two runs: (a) max_tokens=32768, all 12 blocks; (b) max_tokens=4096, blocks 8-11 (reproduction).

## Outputs
- `autoresearch/results/cycle_049_per_block_data_scaling.json`
- CONFIRMED if 32K-token distillation beats geometric on >=8/12 blocks by >=1 PPL.
- REFUTED if geometric still wins on >=8/12 blocks even with 32K tokens.
- INCONCLUSIVE otherwise.
- Report the 4096-token reproduction vs cycle 030 numbers as sanity check.

## Data cap decision
The 4096 cap is NOT kept for the main run. It is tested as an ablation to verify cycle 030 reproduction. The main experiment uses 32K tokens.
