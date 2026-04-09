# Cycle 378 — Item 386: Held-out verification on wikitext-2-test

## Pre-flight

### P1: Duplicate check — PASS
No prior cycle evaluates on wikitext-2-test. All adapter experiments (c352-c377) use val for both train and eval.

### P2: Dependency check — PASS
Depends on c377 (CONFIRMED). Template script exists.

### P3: Falsifiability — PASS
Confirmed: held-out gap >= 6. Refuted: gap < 2. Inconclusive: [2, 6).

### P4: Cost estimate — PASS
6 training runs (3 seeds × 2 configs) × 2000 steps × ~0.15s/step ≈ 30 min on MPS.

## Plan

### Script: `experiments/gpt2_heldout_test_cycle378.py`

Fork from `experiments/gpt2_adapter_only_cycle377.py`. Key changes:

1. **Data loading**: Load BOTH splits. `ds_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")` for training AND checkpoint selection (N_CALIB=200). `ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")` for final eval only (N_EVAL=200). Training and model selection always on val; test touched exactly once per config at the end.

2. **Two configs per seed**: For each seed run:
   - **(A) Float+adapter**: same as c377 — `FloatGPT2MLPWithAdapter` on blocks [2,1,10], h=64, 297K params. Train on val, select checkpoint by val loss, final eval on test.
   - **(B) Compressed+adapter**: Copy `CompressedGPT2MLPWithAdapter` from `gpt2_adapter_schedule_cycle367.py`. Topk+lstsq k=512 on blocks [2,1,10], adapter h=128, ~595K params. Train on val, select checkpoint by val loss, final eval on test.

3. **Float baseline on test**: Eval unmodified GPT-2 on test split (no training). Record as `float_test_ppl`.

4. **Per-seed loop** (seeds 42, 137, 2024):
   - Run config A: fresh GPT-2, attach float adapters, train 2000 steps on val, select best checkpoint by val loss, then evaluate that checkpoint once on test.
   - Run config B: fresh GPT-2, compress 3 blocks (topk k=512 lstsq), attach adapters h=128, train 2000 steps on val, select best checkpoint by val loss, then evaluate that checkpoint once on test.
   - Record `float_adapter_test_ppl` and `compressed_adapter_test_ppl` per seed.

5. **Output JSON**: `autoresearch/results/cycle_378_heldout_test.json`
   ```json
   {"float_test_ppl": ...,
    "float_adapter": {"mean_ppl": ..., "std": ..., "seeds": [...]},
    "compressed_adapter": {"mean_ppl": ..., "std": ..., "seeds": [...]},
    "gap": ..., "verdict": "..."}
   ```

6. **Verdict**: gap = mean(compressed_adapter_test) - mean(float_adapter_test).
   gap >= 6 → CONFIRMED. gap < 2 → REFUTED. Else INCONCLUSIVE.

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_heldout_test_cycle378.py
```

### Verification
- Float test PPL should differ from val PPL (53.89) — test is a different split.
- Init PPL (before training) for float+adapter on test should match float test PPL ± 0.1.
- Compressed init PPL on test should be higher than float (compression degrades).
- Trainable param counts: 297,408 (float+adapter), ~595,200 (compressed+adapter).
