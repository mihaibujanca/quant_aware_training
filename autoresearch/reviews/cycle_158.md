verdict: APPROVE
round: 2

round_1_issues_addressed:
  - "Dead dimensions: noise injection added (step 2, bullet 4). Scale 1e-3 * std(active) preserves warm start while seeding gradient flow to all 64 dims. Fallback to 1e-2 if rank stays stuck."
  - "Effective rank diagnostic: added as step 5. SVD-based (singular values > 1% of max), recorded per-block before and after training. Clear failure threshold (median rank <= 5). Also appears in Validation and Fallbacks."

review_checks:
  R1_testable: "Clear three-way criteria: confirmed (PPL < 68, std < 2.0), refuted (within 2% of 70.7), inconclusive (between). Effective rank adds mechanism verification."
  R2_fair_comparison: "Same architecture (rank-64), hyperparams (2000 steps, lr=1e-4, cosine), and seeds as cycle 155. Noise injection ensures all 64 dims can activate, making rank-64 warm vs rank-64 random a fair comparison."
  R3_measures_hypothesis: "Stacked PPL directly tests compression quality. Effective rank diagnostic catches collapse to rank-4, preventing false attribution of improvement to joint training vs lstsq polish."
  R4_pitfalls: "MP6 addressed via inconclusive category and effective rank diagnostic. MP3/MP5 not applicable (sufficient samples, same capacity). No post-LayerNorm measurements (MP7). No domain transfer (MP4)."
  R5_fallbacks: "Three fallbacks: increase noise 10x if rank stuck, debug init if pre-training PPL wrong, halve lr if divergence. Pre-training validation at ~95 PPL catches init bugs early."

notes: "Clean revision. Both round 1 issues addressed with appropriate mechanisms and clear diagnostic criteria. No new issues found."
