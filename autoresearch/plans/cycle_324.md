# Task
Re-classify cycle 319 verdict from INCONCLUSIVE (strict tree) to rank-only criterion.

# Hypothesis
Under "rank_95 <= 50 alone," cycle 319 verdict shifts from INCONCLUSIVE to CONFIRMED-leak-localized at exactly 3 blocks (b4, b5, b12). All 10 other blocks have rank_95 > 100.

# Pre-flight Results
- P1 (duplicate): No prior reclassification exists in `autoresearch/results/`.
- P2 (dependencies): `autoresearch/results/cycle_319_pgolf_curvature_residual.json` exists.
- P3 (concrete): Yes — extract rank_95, count blocks, write annotation.
- P4 (modifies program.md): No.

# Steps
1. `python3 -c` on `cycle_319_pgolf_curvature_residual.json`: extract the (lrelu2, relu2) pair, print per-block `rank_95`.
2. Count blocks with `rank_95 <= 50` and blocks with `rank_95 > 100`.
3. If exactly 3 blocks pass (b4, b5, b12) and 10+ others >= 100: verdict = **CONFIRMED-reclassification-clean**.
   If more than 3 blocks pass at rank_95 <= 50: verdict = **REFUTED** (rank-only criterion too lenient).
4. Write `autoresearch/results/cycle_319_reclassification.md` with three sections:
   - Strict tree verdict (INCONCLUSIVE, original).
   - Rank-only criterion verdict (CONFIRMED-leak-localized at b4=18, b5=33, b12=21).
   - Justification for audit upgrade, cross-ref to `autoresearch/audits/cycle_319.md` line 28.
5. Commit: `[autoresearch] cycle 324: cycle 319 verdict re-classification under rank-only criterion — <VERDICT>`

# Validation
- Printed rank_95 values match brief expectations (b4=18, b5=33, b12=21).
- Exactly 3 blocks <= 50, exactly 10+ blocks > 100.
- Reclassification doc contains all three required sections.

# Artifacts
- `autoresearch/results/cycle_319_reclassification.md` (new)

# Fallbacks
- If JSON structure differs (no per-block rank_95 or pair key mismatch): dump keys, adapt extraction, proceed.
- If block counts don't match expectation: report REFUTED, document actual counts.
