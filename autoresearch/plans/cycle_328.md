# Task
LASER on parameter-golf b12: spectrum preflight, then conditional rank-7 SVD + B-factor retrain. Modified from item 341 to incorporate cycle 327's lesson (eff_rank ≠ weight rank).

# Hypothesis
- **Gate**: b12's weight SVD top-7 singular values capture > 50% of total energy (sum of squared singular values). If < 10%, skip retrain.
- **If gate passes**: rank-7 retrain BPB ≤ baseline + 0.001.

## Verdicts
- **CONFIRMED-LASER-transfers** if rank-7 retrain BPB ≤ baseline + 0.001.
- **CONFIRMED-partial** if rank-7 retrain BPB in (baseline + 0.001, baseline + 0.005).
- **REFUTED-flat-spectrum** if top-7 energy < 10%: same flat-spectrum problem as b3, retrain skipped.
- **REFUTED-LASER-fails** if top-7 energy ≥ 10% but rank-7 retrain BPB ≥ baseline + 0.01.
- **INCONCLUSIVE** otherwise.

# Pre-flight Results
- P1: Not a duplicate. No LASER retrain on pgolf b12 in PAST.
- P2: Dependency 340 (cycle 327) REFUTED — but refutation was b3-specific (flat weight spectrum). b12 is architecturally different (320x640 vs 640x1920). The spectrum preflight gate resolves this before spending compute. PASS.
- P3: Concrete — run spectrum check (seconds), conditionally run retrain (minutes). PASS.
- P4: No program.md changes needed. PASS.
- P5 (soft): Null baseline = b_init (SVD truncation, no retrain) and full-weight baseline BPB.
- P6 (soft): b12 `proj.weight` is (640, 320). SVD gives U(320,320) S(320) Vh(320,640) with full_matrices=False. rank-7 captures at most 7/320 directions. If spectrum is flat like b3, each direction carries ~0.3% — 7 directions = 2.2% total. Gate would correctly abort.

# Steps
0. `cat autoresearch/cross_project/parameter_golf.md` (REQUIRED by agenda).
1. Write `experiments/pgolf_laser_b12_cycle328.py`:
   - Step A (SPECTRUM CHECK): subprocess-invoke pgolf .venv to load checkpoint, compute `np.linalg.svd(W)` on `blocks.12.mlp.proj.weight`, print singular value spectrum and top-7 energy fraction.
   - If top-7 energy < 10%: write results with verdict REFUTED-flat-spectrum, STOP.
   - Step B (RETRAIN): if gate passes, subprocess-invoke `pgolf_laser_retrain.py --target-block 12 --rank 7`. Also test rank-3 (if spectrum very concentrated) and rank-14 (sensitivity).
   - Collect results, write `autoresearch/results/cycle_328_summary.md`.
2. Run the experiment. Timeout: 300s for spectrum check, 1800s for retrain (if triggered).
3. Move item 341 from NEXT to PAST in `agenda.md` with verdict and commit hash.
4. Commit: `[autoresearch] cycle 328: LASER b12 spectrum check + conditional retrain — {VERDICT}`.

# Validation
- Spectrum check: singular values should sum-of-squares to match `||W||_F^2` (Frobenius norm). Verify.
- If retrain runs: baseline BPB must match cycle 327 baseline (same checkpoint, same val tokens).
- b_init_bpb > baseline (SVD truncation loses information).
- If multiple ranks tested: monotonic BPB improvement with rank.

# Artifacts
- `experiments/pgolf_laser_b12_cycle328.py` (cycle artifact)
- `autoresearch/results/cycle_328_summary.md`
- `autoresearch/results/cycle_328_spectrum.json` (singular values for b12)

# Fallbacks
- If `pgolf_laser_retrain.py` needs modifications for b12 (different weight shape): patch it in Step 1.
- If spectrum is in the gray zone (10-50% energy at rank-7): run retrain but lower expectations. Verdict likely CONFIRMED-partial or REFUTED-LASER-fails.
- If retrain script errors on b12's smaller hidden dim: check that A/B shapes are correct for (640, 320) weight.
