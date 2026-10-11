# Cycle 226 Review — Item 236

**Verdict: APPROVE**

## Summary

Item 236 (P0): KD distillation baseline rerun with corrected loss normalization. This is the corrected rerun of cycle 224, which was downgraded to INCONCLUSIVE because `F.kl_div(..., reduction='batchmean')` on `(B=1, T=128, V=50257)` divided only by B=1, returning the SUM over T·V entries (~30-50) while CE averaged over tokens (~4) — effectively a ~10× over-weighted KD term.

## Plan correctness

- **Loss fix is correct.** Reshaping student/teacher logits to `(B*T, V)` before `F.kl_div(..., reduction='batchmean')` causes batchmean to divide by `B*T`, yielding per-token KD loss on the same scale as CE. Equivalent to `reduction='mean'` but explicit. Fix matches the diagnosis of cycle 224's bug.
- **Trainable surface unchanged** from cycle 224: BLOCKS=[1,4,7], K=512, FullCompressedMLP (W_out+b_out+W_fc+b_fc), ≈2.36M params, tau=2.0, LR=1e-4. Apples-to-apples comparison.
- **CE path uses next-token shift** (`student_logits[:, :-1]`, `ids[:, 1:]`) — consistent with standard LM training.

## Scoping

- α-sweep at 2000 steps × 1 seed × 3 α values is reasonable for picking α* before committing to the full 3-seed run.
- Full run: 4000 steps × 3 seeds at α* — matches cycle 224 methodology.
- Total ≈ 18k step-equivalents (with dual forward for α>0). Cycle 224 fit 30k in budget, so feasible.

## Sanity checks

- **α=0 reproduces cycle 213.** With α=0, the loss reduces to pure CE on the same surface as cycle 213, so best PPL should be ~46.424 ± noise. Good built-in regression test.
- **KD magnitude check.** Plan prints first 5 steps of KD loss; expected O(1-10), not O(30-50). Direct empirical check that the fix worked.
- **Init PPL ~61.7** matches cycle 224 init.
- **Restore-best assertion** `abs(best_ppl_restored - best_ppl) < 1e-3` per seed.

## Verdict thresholds

Match item 236: CONFIRMED if best_mean ≤ 44.0; REFUTED if best_mean ≥ 47.0; else INCONCLUSIVE. Reasonable: cycle 213 CE-only is 46.424, so 44.0 is a meaningful improvement and 47.0 is "no better than CE alone."

## Risks / minor notes

- The fallback to drop α=0 from the sweep is fine, but the α=0 sanity check is the most valuable diagnostic for the loss wiring — keep it unless wall time is actually tight.
- If α* turns out to be 0.0, the verdict should be INCONCLUSIVE (KD adds nothing), not CONFIRMED, regardless of PPL. Plan implicitly handles this since α=0 gives ~46.424 which falls in INCONCLUSIVE band.
- Plan does not specify which CE loss to use during the sweep when α=0 vs α>0 — should be the same `F.cross_entropy` call in both branches (it is, per the snippet).

## Decision

APPROVE. Plan correctly diagnoses and fixes the cycle 224 bug, scopes within budget, and includes proper sanity checks.
