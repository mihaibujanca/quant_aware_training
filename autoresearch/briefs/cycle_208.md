# Cycle 208 Brief: PCA-Conditioned lstsq for Block 1 (Item 219)

## Result: REFUTED

PCA-projected lstsq is worse than standard at all tested dimensions:
- Standard (no PCA): 54.689
- PCA P=512: 54.770 (−0.08 vs standard)
- PCA P=384: 54.910
- PCA P=256: 54.913
- PCA P=192: 55.074 (worst; loses most information)

lstsq already handles near-zero-variance input dimensions gracefully through numerical
conditioning. Explicit PCA projection introduces additional approximation error in the
input-to-activation transformation. The "extra noise" from 576 low-variance dimensions
does not hurt lstsq in practice.
