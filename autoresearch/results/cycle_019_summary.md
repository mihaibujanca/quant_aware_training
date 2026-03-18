# Cycle 019 Summary — Activation Pattern Entropy Cascade (Toy ReLU)

## Config
Architecture: 2→32×4→1 ReLU MLP
Dataset: spirals n=2000 test=500 turns=3 scale=2.0
Epochs: 3000  lr=0.001
Test accuracy: 0.9920

## Entropy Cascade
| Layer | H(marginal) bits | cH(L|prev) bits | drop ratio | r_affine |
|-------|-----------------|-----------------|------------|----------|
| L1   | 6.3339          | 6.3339          | 1.000       | 0.1678   |
| L2   | 7.0846          | 1.6395          | 0.259       | 0.0104   |
| L3   | 5.4667          | 0.2551          | 0.040       | 0.0127   |
| L4   | 4.4157          | 0.2713          | 0.043       | 0.0076   |

## Correlation
Spearman(conditional_H_ratio, r_affine) = 0.4000  p=0.6000

## Verdict: CONFIRMED
Layer 3 conditional H ratio=0.040 ≤ 0.5 AND 3 low-cH layer(s) have r < 0.5×r_L1 (r_L1=0.1678); low-r layers: [2, 3, 4]
