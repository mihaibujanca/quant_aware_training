# Cycle 044 Brief — Domain Robustness (Item 62)

## Item
Does geometric compression (k=4@6x + correction) generalize beyond Shakespeare?

## Dependencies
- 037: REFUTED overall, but k=4@6x is solid (+9.77% uncorrected)
- 039: CONFIRMED, correction at block 8 recovers 1.23pp on k=4 (9.59% -> 8.35%)

## Key question
Channel selection is fit on Shakespeare probe tokens. If the same channels are kept but eval uses WikiText-2, does PPL cost stay within 3pp of corpus-specific fit?

## Baseline numbers (Shakespeare, from 039)
- k=4@6x uncorrected: +9.59%
- k=4@6x + correction: +8.35%

## Data
- Shakespeare: already available at `data/shakespeare.txt`
- WikiText-2: download raw validation split (~2MB) from HuggingFace
- OpenWebText: too large; skip per agenda "2 of 3 corpora" suffices

## Conditions (3 seeds each, 200 eval seqs)
1. Shakespeare probe + Shakespeare eval (baseline replication)
2. WikiText-2 probe + WikiText-2 eval (corpus-specific)
3. Shakespeare probe + WikiText-2 eval (zero-shot transfer)
