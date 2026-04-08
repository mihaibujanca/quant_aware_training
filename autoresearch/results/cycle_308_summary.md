# Cycle 308 Summary — Item 318: Synthetic rank-2 unigram projector at block 1

**Verdict: REFUTED**

- best_synth_ppl = **127.4956** at (s1, s2) = (0.5, 0.5)
- float_ppl = 53.8949 (ref 53.895)
- cycle_280 no-retrain init PPL reproduced = 57.6581 (ref 57.664)
- cycle_280 trained best (reference, NOT this cycle) = 49.032

## Verdict logic (item 318 verbatim)

- CONFIRMED iff `best_synth_ppl <= 60`
- REFUTED   iff `best_synth_ppl >= 100`
- INCONCLUSIVE otherwise.

## Synthetic construction

- A_frozen = U[:, :2] @ diag(S[:2]) (cycle 280 LEFT factor, frozen, no training)
- d1 = wte^T @ log_p, normalised (||d1_raw|| = 1257193.3750)
- d2 = wte^T @ s_vec, Gram-Schmidt vs d1, normalised (||d2_raw|| = 1318.6527, |d1.d2| = 2.50e-07)
- Stopword indicator vocabulary: 747 unique single-token ids from 420 base strings (each encoded with and without leading space)
- B_synth(s1, s2) = stack([s1 * d1, s2 * d2])  shape (2, 768)
- replacement: c_proj_1.weight <- A_frozen @ B_synth
- Trainable params = 0 (grid search over 2 scalars is not training)

## Grid (PPL per (s1, s2))

| s1\s2 |      0.5 |      1.0 |      2.0 |      5.0 |     10.0 |
|---:|---:|---:|---:|---:|---:|
|  0.5 |  127.496 |  331.542 | 1704.011 | 10048.413 | 25433.509 |
|  1.0 |  344.077 |  577.694 | 1990.227 | 9950.720 | 23963.790 |
|  2.0 | 5153.016 | 2962.268 | 3249.084 | 10665.587 | 22287.787 |
|  5.0 | 52486.165 | 33428.380 | 14551.892 | 14476.112 | 24322.521 |
| 10.0 | 27502.150 | 23110.984 | 17641.493 | 13383.280 | 21439.383 |

## Config

`TARGET_BLOCK=1, RANK=2, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, SEED=42, DEVICE=mps, GRID=[0.5, 1.0, 2.0, 5.0, 10.0]`

Wall: 39.3 s
