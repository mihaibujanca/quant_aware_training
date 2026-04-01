# Cycle 212 Brief: 10-Block Adam Correction (Item 222)

## Result: REFUTED

10-block BLOCKS=[1..10] K=512 Adam W_out correction:
- Init (topk+lstsq): 152.338 (+182.6% vs float 53.895)
- Step 200: 107.933
- Step 400: 101.305
- Step 600: 98.962
- Step 800: 98.493 ← best
- Step 1000: 99.419 (slight overfitting)
Best: 98.493 (+82.7% vs float). REFUTED (threshold: ≤65.0).

**Adam correction scaling pattern:**
- 3-block (1.18M params): init +14.6% → best −10.7% vs float (CONFIRMED)
- 6-block (2.36M params): init +60.7% → best +10.7% vs float (INCONCLUSIVE, early stop)
- 10-block (3.94M params): init +182.6% → best +82.7% vs float (REFUTED)

W_out-only Adam correction cannot fix the cascade for 10 compressed blocks.
The cascade grows super-additively and exceeds the capacity of W_out correction.
Practical limit: ~6 blocks with early stopping.
