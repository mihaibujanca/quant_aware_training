# Cycle 046 Brief — Lstsq Refit Quality Degradation (Item 55)

## Item
Measure lstsq refit condition number and held-out MSE at compression ratios {2,4,6,8,10} on blocks 7, 5, 4. Determine whether the refit becomes ill-conditioned at high ratios and whether overfitting explains the domain gap (Shakespeare vs WikiText-2).

## Dependencies
- 54 (CONFIRMED, cycle 040): 10x single-block is free on Shakespeare, all 3 blocks tolerate 8-10x

## Key context
- Cycle 040: single-block 10x works on Shakespeare (blocks 7,5,4 all <1% PPL increase)
- Cycle 044: method fails on WikiText-2 (+44% uncorrected, +31% corrected)
- Hypothesis: lstsq refit overfits narrow Shakespeare distribution but is underdetermined for diverse WikiText-2 tokens

## Verdict criteria (from agenda)
- **Confirmed**: cond# increases >10x between 4x and 8x AND held-out MSE/train MSE >2.0 at 8x for majority of blocks
- **Refuted**: cond# stays <1e4 across all ratios and held-out MSE tracks train MSE
- **Inconclusive**: cond# rises but held-out MSE stays close to train MSE

## Added scope
Run on BOTH Shakespeare and WikiText-2 probes (motivated by cycle 044 domain gap finding).
