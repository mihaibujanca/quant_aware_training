verdict: REVISE
round: 2

## Round 1 issue status

- **I1 (blocking): Condition C for filtering vs bias.** RESOLVED. Step 6 adds condition C (positive-only topk, no b_leak). Step 7 reports gain_C alongside gain_B to attribute effects.
- **I2 (blocking): b_leak overfitting.** RESOLVED. P5 splits probe (first 100 seqs) vs fit (last 100 seqs). Step 2 logs per-channel GELU(z_neg) variance as reliability diagnostic.
- **I3 (blocking): lstsq order / b_leak absorption.** PARTIALLY RESOLVED. Order is clarified, but creates a new issue (see I4 below).
- **A1: Exclude blocks where N_pos < k.** RESOLVED. Step 7 excludes them.
- **A2: Force INCONCLUSIVE if N_neg < 100 everywhere.** RESOLVED. Step 7 includes this guard.

## New issues

**I4 (blocking): Condition B reconstruction overshoots by b_leak.** Step 5 fits lstsq to target y, then adds b_leak post-hoc. The final reconstruction is `lstsq_approx(y) + b_leak`. Since lstsq already tries to approximate y, adding b_leak produces `≈ y + b_leak`, which overshoots by b_leak. This makes Condition B *systematically worse* than Condition A by construction — it's not testing whether b_leak helps, it's adding a known bias term on top of a reconstruction that didn't ask for it. **Fix:** In Condition B, set lstsq target = `y - b_leak`, so the final output is `lstsq_approx(y - b_leak) + b_leak ≈ y`. This way lstsq only needs to reconstruct the portion of the output *not* explained by b_leak, and the comparison to Condition A/C is fair.

## R3: Known pitfalls

No new pitfall violations beyond round 1 analysis. MP3 still satisfied (25600 >> 512). MP7 correct (pre-GELU hook point).

## Advisory

**A3:** Specify what the 3 seeds control (Adam init? data shuffle within fit split? both?). Minor but aids reproducibility.
