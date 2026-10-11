# Cycle 382 — Item 390: Minimal-penalty compression on [b1,b7,b2]

## Pre-flight

### P1: Duplicate check — PASS
No prior cycle measures compressed+adapter on blocks [1,7,2]. c378 used [2,1,10].

### P2: Dependency check — PASS
Depends on c367 (COMPLETED, block importance) and c378 (CONFIRMED, adapter protocol).

### P3: Falsifiability — PASS
Confirmed: penalty < 3 PPL. Refuted: >= 5 PPL. Inconclusive: [3, 5).

### P4: Cost estimate — PASS
2 configs x 3 seeds x 2000 steps x ~0.15s/step ~ 30 min on MPS.

## Plan

### Script: `experiments/gpt2_least_important_cycle382.py`

Fork from `experiments/gpt2_heldout_test_cycle378.py`. Changes:

1. **BLOCKS**: `[1, 7, 2]` (the 3 least-important by c367).

2. **Thresholds**: `CONFIRMED_LT_PENALTY = 3.0`, `REFUTED_GE_PENALTY = 5.0`.
   Penalty = mean_comp_test - mean_float_adapter_test.

3. **Adapter config**: `ADAPTER_H = 64` for BOTH float and compressed configs.
   This isolates compression damage: the only difference between configs is whether
   the FFN is compressed. No capacity confound.

4. **Expected trainable params** (h=64, d_model=768, d_ff=512 compressed):
   - Float adapter per block: W1(768×64) + b1(64) + W2(64×768) + b2(768) = 49,152 + 64 + 49,152 + 768 = 99,136.
   - Float adapter total: 3 × 99,136 = 297,408.
   - Compressed adapter per block (same arch): 99,136 adapter + proj weights from compression.
   - Compressed adapter total: 3 × 99,136 + proj weights.

5. **Output JSON**: `autoresearch/results/cycle_382_least_important.json`
   ```json
   {"float_test_ppl": ...,
    "float_adapter": {"mean_test_ppl": ..., "std": ..., "seeds": [...]},
    "compressed_adapter": {"mean_test_ppl": ..., "std": ..., "seeds": [...]},
    "penalty": ...,
    "penalty_c378_ref": 7.49,
    "blocks": [1, 7, 2],
    "verdict": "..."}
   ```

6. **Verdict logic**:
   ```python
   penalty = mean_comp - mean_float_adapter
   if penalty < 3.0: verdict = "CONFIRMED"
   elif penalty >= 5.0: verdict = "REFUTED"
   else: verdict = "INCONCLUSIVE"
   ```

7. **Print comparison**: Also print c378 penalty (7.49) for direct comparison.

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_least_important_cycle382.py
```

### Verification
- Float test PPL matches c378 baseline (~56.52 +/- 0.5).
- Init PPL (step 0) for float+adapter ~ float test PPL.
- Trainable param counts match expectations.
- Both configs converge (PPL decreasing).
- Compare penalty to c378 penalty (7.49): expect lower since b7 replaces b10.
