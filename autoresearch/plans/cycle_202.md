# Cycle 202 Plan: BLOCKS=[1,4,7] vs [1,4,7,11] Compression (Item 213)

## Overview

Compare PPL for BLOCKS=[1,4,7] vs BLOCKS=[1,4,7,11] with K=512.
Hypothesis: excluding block 11 (output decompressor, cycle 201) improves PPL.

Script: `experiments/block11_exclusion_cycle202.py`
Output: `autoresearch/results/cycle_202_block11_exclusion.json`

## Result: CONFIRMED

BLOCKS=[1,4,7] PPL=61.746 (+14.6% above float=53.895)
BLOCKS=[1,4,7,11] PPL=68.138 (+26.4% above float)
Improvement: 6.4 PPL by excluding block 11.
