# Task
Cycle 183 / Item 193: KL-distillation control baseline on GPT-2 FFN blocks [1,4,7,11], k=4 (768→512→768 GELU, ~787K params/block). Compare stacked 4-block PPL against topk+lstsq+polish (PPL ~70.7).

# Hypothesis
Geometric compression (topk+lstsq+polish) beats naive distillation at matched params.

Criteria (from agenda):
- Confirmed if: topk+lstsq+polish beats distillation by ≥3 PPL (geometric justified).
- Refuted if: distillation matches or beats topk+lstsq+polish (geometric adds no value).
- Inconclusive if: within ±3 PPL.

# Revision (R1): Convergence fairness fix
Review flagged initialization + budget asymmetry: topk+lstsq warm-starts from teacher weights and polishes ~394K params for 500 steps; distillation cold-starts ~787K params. 500 steps likely insufficient for convergence.

Fix: run distillation for up to 2000 steps with early-stop on plateau (mean loss over steps [t-50, t] within 5% of previous 50-step window). Record 500-step snapshot for matched-budget context. Use converged result for the verdict.

# Steps
1. Modify `experiments/distillation_baseline_cycle183.py`:
   - Change max steps from 500 → 2000.
   - Log training loss at steps [0, 100, 200, 500, 1000, 1500, 2000].
   - Save 500-step PPL snapshot (matched-budget) alongside final PPL.
   - Add plateau detection: stop early if mean loss over last 50 steps is within 5% of previous 50-step mean.
2. Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/distillation_baseline_cycle183.py`
3. Verify output JSON at `autoresearch/results/cycle_183_distillation_baseline.json` contains:
   - Loss trajectory at logged steps (convergence evidence)
   - 500-step stacked PPL (matched-budget snapshot)
   - Converged stacked PPL + step count at convergence
   - Per-block single-swap PPLs, 3 seeds, param counts
   - Verdict string based on CONVERGED PPL (not 500-step)
4. Sanity checks:
   - Float PPL ~22-25; distillation loss monotonically decreasing
   - If loss still dropping >5%/100 steps at step 2000, flag as possibly unconverged
   - Student params ~787K/block, ~3.15M total
   - 3-seed std: if >5 PPL → INCONCLUSIVE
5. Record in `autoresearch/results/cycle_183_manifest.txt`
6. Update agenda: move 193 from NEXT to PAST
7. Commit: `[autoresearch] cycle 183: distillation control baseline — {VERDICT}`

# Validation
- Float PPL: 20-30.
- Loss trajectory: final < initial (learning occurred); ideally plateaued.
- Converged PPL < 500-step PPL (more training helped, or equal if already converged).
- Per-block PPLs finite and < 300.
- Param parity within 5%.

# Artifacts
- `experiments/distillation_baseline_cycle183.py` (modified)
- `autoresearch/results/cycle_183_distillation_baseline.json`
- `autoresearch/results/cycle_183_manifest.txt`

# Fallbacks
- If diverges: reduce lr to 1e-5, retry. If still diverges, try MSE loss.
- If still dropping at 2000 steps: extend to 4000 and note in manifest.
- If stacked PPL variance > 10: increase to 5 seeds.
