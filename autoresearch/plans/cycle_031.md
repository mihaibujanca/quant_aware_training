# Cycle 031 Plan — Item 46: Grokking as Compression Catalyst (Modular Arithmetic v2)

## Goal
Run grokking modular arithmetic experiment, compare student compression pre- vs post-grokking.

## Script
`experiments/grokking_compression_catalyst_v2.py` — exists but needs two fixes.

## Fixes Required

### Fix 1: Remove dead eval_acc(None, ...) call (lines 315-318)
Replace the buggy lines:
```python
    teacher_acc = eval_acc(
        make_model(2 * P, HIDDEN, P, DEPTH).tap(
            lambda m: m.load_state_dict(checkpoints[idx])) if False else None,
        X_test, y_test)
    # Inline teacher eval
```
With just:
```python
    # Inline teacher eval
```
(The teacher_acc computation that follows at lines 320-323 is correct.)

### Fix 2: Update output file names from cycle_021 to cycle_031
- Line 445: `cycle_021_grokking_trajectory.png` → `cycle_031_grokking_trajectory.png`
- Line 481: `cycle_021_grokking.json` → `cycle_031_grokking.json`

## Execution
```
cd /Users/mihai/Projects/quant_aware_training
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/grokking_compression_catalyst_v2.py
```

## Expected Runtime
- 50K epochs on a tiny network (2*97 → 128 → 97): ~5-15 minutes on CPU
- Grokking expected around epoch 10K-30K (typical for this setting)
- Distillation: 3 × 5000 epochs = fast (tiny network)

## Output Artifacts
- `autoresearch/results/cycle_031_grokking.json` — full trajectory + compression results
- `autoresearch/results/cycle_031_grokking_trajectory.png` — 6-panel plot

## Verdict Criteria
- **Confirmed**: grokking detected AND post-grokking student acc > pre-grokking student acc by ≥10pp
- **Refuted**: no grokking after 50K epochs OR compression gain <2pp
- **Inconclusive**: grokking detected but gain 2-10pp

## Commit
`[autoresearch] cycle 031: grokking compression catalyst modular arithmetic`
