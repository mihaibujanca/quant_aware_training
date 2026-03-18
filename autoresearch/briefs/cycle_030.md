# Cycle 030 Brief — Learned Compressor Generalization on Held-Out Blocks

## Item 18 Summary
Train a shared low-rank compressor on 8 of 12 GPT-2 FFN blocks, evaluate zero-shot on 4
held-out blocks. Tests whether the learned track discovers reusable structure across blocks
or merely memorizes block identities.

## Confirmed if
Held-out blocks: learned compressor beats geometric baseline in majority of blocks and stays
within 25% of its in-distribution (train-block) performance.

## Approach
Use GPT-2 + Shakespeare probe tokens. Split 12 FFN blocks: train on blocks [0,1,2,3,4,5,6,7],
held-out on blocks [8,9,10,11]. 8x compression: 3072→384→768.

**Shared low-rank bottleneck** (simplest valid design):
- Train shared U (768→384) and V (384→768) across all 8 train blocks
- Per-block bias b_i (768-dim) learned via lstsq
- At test: fix U,V; adapt per-block bias on held-out probe data (lstsq, no SGD)
- Baselines: (a) geometric per-block (channel select + lstsq); (b) per-block distillation
  (separate 768→384→768 MLP trained per held-out block on same probe tokens)

## Key risk
Shared U,V may underfit if block output distributions differ substantially (each block computes
different function). Per-block distillation baseline provides the zero-generalization ceiling.

## Depends on
Item 10 (nanochat_ffn_compression infrastructure) — confirmed, infra reused from cycle 029.
