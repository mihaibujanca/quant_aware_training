# Cycle 243 audit — item 253

**Status: PASS (CONFIRMED, interesting=true)**

## Phase 1 — literal verdict

Criterion: Confirmed if any block X has `ppl_if_X_restored ≤ 56.0`.

Sorted `ppl_if_X_restored` (held-out eval_ids[200:300], SEQ_LEN=128):

| rank | block | ppl_if_X_restored | ΔPPL | residual_mse |
|---|---|---|---|---|
| 1 | 5 | 54.291 | +5.100 | 3.24e-1 |
| 2 | 4 | 54.495 | +4.896 | 2.12e-1 |
| 3 | 7 | 55.256 | +4.135 | 5.65e-1 |
| 4 | 8 | 55.608 | +3.783 | 9.28e-1 |
| 5 | 2 | 55.861 | +3.530 | 2.22e-1 |
| 6 | 1 | 56.699 | +2.692 | 1.02e-1 |

**5 of 6 blocks (all except block 1) satisfy `ppl_if_X_restored ≤ 56.0`.** Literal verdict CONFIRMED — in fact strongly so: the criterion asks for at least one, and only block 1 misses it (by 0.7). float=53.895, fully_compressed=59.391, gap=+5.496.

## Phase 2 — methodology checks

- **Phase 0 refit reproduces cycle 221 seed 42 exactly**: init=86.686, best=59.391 at step 1300 — same as the cycle 221 reference best and best_step. lstsq determinism gives exact init match; training trajectory match is exact to the recorded value.
- **Round-trip sanity** performed after every restoration: `|rt - fully_compressed| < 1e-4` asserted for all 6 blocks.
- **Calib/eval disjoint** by id assertion; eval held out, calib used only for lstsq init and Phase 2 MSE.
- **Float MLP extraction** uses a fresh `GPT2LMHeadModel.from_pretrained` and `copy.deepcopy`, not mutation of the compressed model.
- **Residual MSE** measured on calibration activations (not eval) — fair for error magnitude but not directly comparable to eval ΔPPL.

## Phase 3 — interesting findings

1. **Non-additivity**: `sum(Δ) = 24.14 PPL` vs actual gap `5.50 PPL` (4.4×). This is expected — ΔPPL here measures "what if ONLY block X were restored," not a partition. The 4.4× ratio quantifies how strongly the six compressed blocks compound.

2. **MSE does NOT rank ΔPPL**. Block 8 has the largest residual MSE (0.928) but only rank-4 ΔPPL (+3.78); block 5 has the largest ΔPPL (+5.10) with moderate MSE (0.324). So per-block calibration-MSE is not a reliable proxy for per-block contribution to the cascade PPL gap.

3. **Contradicts cycle 242's late-block-dominance finding at face value**. Cycle 242 measured the penalty of placing fresh-fit compressed blocks at [8,9,10] vs [0,1,2]; here under a jointly-fit [1,2,4,5,7,8] cascade, **block 5 (mid) and block 4 (early-mid) contribute more than blocks 7 and 8**. These are different measurements: cycle 242 = fresh-fit position sensitivity; cycle 243 = restoration delta from a jointly-optimised compressed cascade. Both can hold simultaneously — the late-block penalty at [8,9,10] in cycle 242 may be an **interaction** between late blocks 8+9+10, not a single-block effect. Worth a dedicated follow-up.

4. **Remediation implication (valid)**: no single-block targeted remediation can close the 5.5 PPL gap — every block carries ≥2.7 PPL of the compound error. The gap is a cascade phenomenon.

No issues with methodology. Verdict stands.
