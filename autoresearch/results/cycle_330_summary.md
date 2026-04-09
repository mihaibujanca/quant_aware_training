# Cycle 330 — Selective lrelu2 at b4/b5

## Verdict: REFUTED-distributed-mechanism

## BPB Results

| Config | BPB |
|---|---|
| global-lrelu2 (as trained) | 12.802887 |
| selective (b4+b5 only) | 16.755052 |
| global-relu2 (full ablation) | 18.414793 |

## Deltas

- selective vs global-lrelu2: +3.952165
- global-relu2 vs selective: +1.659741
- global-relu2 vs global-lrelu2: +5.611907

## Verdict Criteria

- CONFIRMED-localized-mechanism: |selective - global-lrelu2| < 0.003 AND (global-relu2 - selective) >= 0.003
- REFUTED-distributed-mechanism: selective >= global-lrelu2 + 0.003
- INCONCLUSIVE: otherwise

## Parameters

- Checkpoint: /Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz
- n_tokens: 51200
- lrelu_blocks (selective): [4, 5]
- n_blocks: 13
