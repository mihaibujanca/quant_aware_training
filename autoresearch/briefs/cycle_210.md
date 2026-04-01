# Cycle 210 Brief: Adam W_out Step-Count Curve (Item 220)

## Result: INCONCLUSIVE

Step-count curve for BLOCKS=[1,4,7] K=512 Adam W_out correction:
- Step 500:  50.260
- Step 1000: 48.173
- Step 2000: 46.806
- Step 4000: 46.556 ← optimal
- Step 8000: 51.223 ← overfitting

Plateau between steps 2000-4000 (delta=0.25 PPL, <0.5 threshold). After step 4000
(~20 epochs over 200 calib sequences), the model overfits to calibration data (+4.67 PPL
at step 8000). INCONCLUSIVE: plateau found in 2000-4000 window, but overfitting occurs.

**Practical recommendation**: use early stopping at eval plateau; optimal is ~4000 steps
for N_CALIB=200. With larger calibration data, overfitting would occur later.
Best achievable PPL: 46.556 (vs float 53.895 = −13.6% below float).
