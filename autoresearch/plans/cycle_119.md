# Task
Item 136: Cascade corrections at k=5 stacking — GPT-2 FFN.
Run r=8 LoRA cascade corrections on k=5 compressed blocks [7,5,4,1,6] at 6x.
(Note: brief says [1,3,4,7,11] — this is wrong; use [7,5,4,1,6] per cycle 106 and existing script.)

# Hypothesis
r=8 cascade corrections recover ≥30% of k=5 degradation (uncorrected ~+63% PPL).
k=4 achieved 74% recovery (cycle 115); k=5 adds one block so recovery may drop but should exceed 30%.

# Pre-flight Results
- P1 (duplicate): No. Item 133 timed out (cycle 117); 136 is the approved k=5-only split.
- P2 (deps): Met. k=4 cascade (cycle 115) CONFIRMED. Script exists from cycle 117.
- P3 (scope): Yes — k=5 only, single rank (r=8), 3 seeds, ~15 min/seed.
- P4 (program): No modifications needed.
- P5 (null baselines): Uncorrected k=5 PPL is ~93.9 (cycle 106, blocks [7,5,4,1,6], seed 42).
  Float baseline ~51.7 (3-seed mean). Degradation ~42 PPL points.
- P6 (alignment): Same LoRA architecture as cycle 115 — validated on k=4.
- P7 (power): 3 seeds. k=4 had std ~7.5 PPL; 30% of ~42 = 12.6 PPL improvement. SNR ~1.7, adequate.

# Steps
1. Modify `experiments/cascade_correction_k5_k8_gpt2.py`:
   - Strip all k=8 code paths (out of scope for item 136).
   - Set `SEEDS = [42, 123, 7]` (run all 3 from the start — reviewer requires ≥3).
   - Set `FFN_BLOCKS = [7, 5, 4, 1, 6]`, `CORRECTION_RANK = 8`.
   - Add abort: if uncorrected PPL > 200 for any seed, log and skip that seed.
   - Save to `experiments/cascade_correction_k5_gpt2.py` (new file, preserve original).
2. Run: `python experiments/cascade_correction_k5_gpt2.py`
   - Expected: ~45 min total (3 seeds × ~15 min).
3. Collect results into `autoresearch/results/cycle_119_cascade_correction_k5.json`:
   - Per-seed: baseline_ppl, uncorrected_ppl, corrected_ppl (r=8).
   - Aggregates: mean/std for each, recovery_pct = (uncorrected - corrected) / (uncorrected - baseline) × 100.
   - Total correction params: 5 × 2 × 768 × 8 = 61,440.
4. Generate comparison plot: k=4 (cycle 115) vs k=5 recovery, saved to results dir.
5. Verdict logic:
   - **Confirmed** if mean recovery ≥ 30%.
   - **Refuted** if mean recovery < 10%.
   - **Inconclusive** if 10–30%.
6. Update `autoresearch/agenda.md`: move item 136 to PAST with outcome line.
7. Commit all artifacts.

# Validation
- Float baseline PPL must be within ±2 of cycle 115's 51.7 (same model, same eval).
- Uncorrected k=5 PPL must be within ±10% of cycle 106's ~93.9 (same blocks, same compression).
- If either sanity check fails, abort and investigate before reporting.

# Artifacts
- `experiments/cascade_correction_k5_gpt2.py`
- `autoresearch/results/cycle_119_cascade_correction_k5.json`
- `autoresearch/results/cycle_119_cascade_correction_k5.png`

# Fallbacks
- If script times out (>60 min): reduce EVAL_SEQS to 100 and re-run.
- If correction training diverges: halve lr to 5e-5, keep 2000 steps.
- If recovery is near boundary (25–35%): report as-is, let verdict logic decide.
