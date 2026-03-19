# Cycle 038 — Context Brief

## Item 57: Block Ordering Optimization for 5-Block Stacking

### Hypothesis
The k=5 wall (+17.46% PPL, cycle 033) may be order-dependent. Cycle 007 established easiest-first ordering is optimal for k≤4; item 57 tests whether different 5th-block choices or reverse insertion order can beat the +17.46% baseline.

### Dependency Status
- **Item 53** (cycle 036): INCONCLUSIVE. k=4 at +9.59% PPL in the 8–12% range. No hard gate; item 57 can proceed.
- **Cycle 037** (item 56): Confirmed the block-count wall is hard. k=4@6× = +9.77% PPL (27.8% param reduction). All k=5 tested = +16.98% to +20.54%. The wall exists but ordering hasn't been tested.

### Candidates for 5th Block
From cycle 005 damage ranking (easiest first: blocks 7, 5, 4, 1, 3, 6):
- Base set (k=4): 7, 5, 4, 1 (confirmed <10% PPL)
- 5th-block candidates from remaining easiest: **blocks 3, 6** (next two easiest)
- Previously untested: **blocks 8, 9, 10, 11, 2, 0** (harder blocks, exploratory)

### Test Plan
1. Test all 6 possible 5th-block choices from the 6 easiest blocks (add one, fix k=4 base as 7,5,4,1)
2. Test reverse-order insertion (compress 5th block first, then layer in 4 easy ones)
3. Compare to cycle 033 baseline: **k=5 needs <12% relative PPL to confirm** (beating cycle 033's +17.46% by ≥5pp)

### Success Criterion (from agenda)
- **CONFIRMED**: ≥1 config achieves <12% relative PPL
- **REFUTED**: all configs >15%
- **INCONCLUSIVE**: best is 12–15%

### Key Data from Cycle 037
- Baseline: 120.32 PPL (seed 42) / 127.76 PPL (seed 123)
- Best k=5 from heterogeneous sweep: block 6 at 4× = +16.99% / +16.49% (mean +16.985%)
- All k=5 variants stayed ≥+16.49% even with low 5th-block ratio
