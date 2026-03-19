# Cycle 046 Design Review — Lstsq Refit Quality Degradation

## R1: Hypothesis clarity
PASS. Clear mechanistic question: does the lstsq refit degrade at high compression ratios, and does it degrade worse on WikiText-2 than Shakespeare? Both are testable with condition number and overfit ratio.

## R2: Measurement soundness
PASS. Condition number is a standard numerical diagnostic. Overfit ratio (held-out MSE / train MSE) directly tests for overfitting. The 80/20 split on 200 sequences (~3200 tokens) gives ~640 held-out tokens per config — sufficient for MSE estimation.

**One concern**: at ratio 10x, the number of selected channels is 3072/10 = 307. With ~2560 train tokens, the system is overdetermined (2560 > 307), so lstsq is well-posed in the row sense. The ill-conditioning, if any, comes from column correlations (multicollinearity among selected channels), not from an underdetermined system. The plan's condition number measurement correctly captures this. PASS.

## R3: Capacity / baseline adequacy
N/A — analysis experiment, no model training.

## R4: Pitfall check
- **MP1 (threshold above trivial baseline)**: N/A — no thresholds, just measuring condition numbers.
- **MP2 (fitting on eval data)**: RISK. The plan uses probe tokens for both channel selection AND lstsq fitting. The 80/20 split is within the probe set, so channel selection sees the held-out tokens. **Mitigation**: this mirrors the actual compression pipeline (which also selects and fits on the same probe). The analysis is measuring how the real pipeline behaves, not introducing a new method. ACCEPTABLE.
- **MP4 (domain mismatch)**: ADDRESSED. The plan explicitly runs both Shakespeare and WikiText-2 as separate conditions. This IS the analysis question.
- **MP6 (verdict misclassification)**: LOW RISK. The confirmed/refuted criteria are quantitative (>10x condition increase, >2.0 overfit ratio). Unlikely to be ambiguous.

## R5: Confounds
- Channel selection itself changes with ratio (fewer channels = different subset). The condition number captures the joint effect of fewer channels + which channels. This is the right thing to measure — the refit quality includes the selection quality. PASS.

## R6: Data availability
PASS. Shakespeare at `data/shakespeare.txt`. WikiText-2 cached from cycle 044.

## R7: Agenda criteria match
PASS. Plan's verdict criteria match item 55 exactly.

## Verdict: APPROVED

Clean analysis experiment. The added WikiText-2 condition is well-motivated by cycle 044 and could directly explain the domain gap mechanism. No design issues found.
