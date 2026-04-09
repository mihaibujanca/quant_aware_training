# Cycle 335: Selective LeakyReLU² Re-run

## Verdict: REFUTED-distributed-mechanism

## BPB Results

| Config | BPB |
|---|---|
| global-lrelu2 (baseline) | 1.393100 |
| selective (lrelu2 at b4,b5) | 1.508522 |
| global-relu2 | 1.573986 |

## Deltas

- selective vs global-lrelu2: +0.115422
- global-relu2 vs global-lrelu2: +0.180886

## Loader Config

- mlp_act: lrelu2
- xsa_last_n: 0
- rope_dims: 0

## Validation

- Cycle 330 broken value (wrong loader): 12.8 BPB
- This run global-lrelu2 BPB: 1.393100 — FIXED

## Thresholds

- CONFIRMED-localized-mechanism: delta_sel < 0.003 AND delta_relu2 >= 0.003
- REFUTED-distributed-mechanism: delta_sel >= 0.003
- INCONCLUSIVE: neither threshold met
