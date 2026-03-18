# Cycle 032 Plan — Item 47: GPT-2 Dead Channel Census v2 (GELU-Correct)

## Goal
Measure GPT-2 FFN channel removability with GELU-correct definitions. Test whether ANY free structural compression exists.

## Script
`experiments/exact_prune_linear_collapse_gpt2_v2.py` — exists, cycle-number fixes already applied.

## Execution
```
cd /Users/mihai/Projects/quant_aware_training
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/exact_prune_linear_collapse_gpt2_v2.py
```

## Expected Runtime
~5-10 minutes on CPU (GPT-2 inference on 200 probe sequences × 3 forward passes, plus 8 pruning curve evaluations).

## Output Artifacts
- `autoresearch/results/cycle_032_prune_collapse.json` — per-block stats + pruning curve
- `autoresearch/results/cycle_032_prune_collapse.png` — 4-panel plot

## Verdict Criteria
- **Confirmed**: avg dead+linear (T=±3) ≥5% AND pruning 5% causes ≤1% PPL increase
- **Refuted**: <1% removable at all thresholds AND pruning >2% always causes >1% PPL increase
- **Inconclusive**: channels identified as dead but pruning them hurts PPL

## Commit
`[autoresearch] cycle 032: GPT-2 dead channel census v2 GELU-correct`
