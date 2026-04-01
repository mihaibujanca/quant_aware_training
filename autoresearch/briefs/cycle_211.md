# Cycle 211 Brief: 6-Block Adam Correction (Item 221)

## Result: INCONCLUSIVE

6-block BLOCKS=[1,2,4,5,7,8] K=512 Adam correction PPL trajectory:
- Init (topk+lstsq): 86.686
- Step 500:  61.792
- Step 1000: 60.307
- Step 1500: 59.680 ← best (below 60.0 criterion!)
- Step 2000: 61.141 (overfitting starts)
- Step 4000: 72.250 (final, severe overfitting)

With strict early stopping at step 1500, PPL=59.680 (criterion ≤60.0 met). Without
early stopping, final PPL=72.250 (+34% vs float 53.895). INCONCLUSIVE.

**Key pattern**: More trainable params → faster overfitting:
- 3 blocks (1.18M params): optimal at ~4000 steps
- 6 blocks (2.36M params): optimal at ~1500 steps

6-block compression IS achievable (59.680 = +10.7% vs float) but requires early
stopping and is fragile. The overfitting rate scales roughly linearly with param count.
