# Cycle 239 Design Review / Item 249

**Verdict: APPROVE**

## Summary
Rerun of cycle 234 (`gelu_noneg + μ_l + LoRA r=8` on all 12 `mlp.c_proj`) with a disjoint calib/eval split to close the cycle 234 data-leakage loop flagged in `autoresearch/audits/cycle_234.md`. Plan matches item 249 verbatim.

## Strengths
- **The fix is exactly the fix.** `calib_ids = all_ids[:200]`, `eval_ids = all_ids[200:300]`, disjointness asserted by object id AND tensor equality on the first pair. μ precomputed on calib only; Adam trained on calib only; every PPL print + best-state selection + `final_ppl` on `eval_ids` only. This is the canonical safe-split template (cycles 206/210/223/225/230).
- **LoRA scaffolding reused verbatim** from cycle 234 (r=8, all 12 blocks, A init N(0, 0.01), B init zero, 368,640 trainable). No moving parts beyond the split — good experimental hygiene.
- **Honest baselines**: float ref on eval pre-patch, μ-only `init_ppl_eval` on eval post-patch but pre-train. The plan correctly refuses to assert `init_ppl_eval == 61.664` (that was calib; held-out is a different number).
- **Sanity check that catches real bugs**: `final_ppl ≤ init_ppl_eval` flagged as a test, not an assert. If calib training degrades held-out, that is itself the signal — exactly right.
- **Fallbacks cover the two real failure modes**: (a) wikitext-2 val can't yield ≥300 seqs at SEQ_LEN=128 → drop to 64 and log; (b) wall-time tight → SEED=42 only with justification. Neither collapses the experiment.
- **Verdict thresholds published up front** (≤48 CONFIRMED, ≥58 REFUTED) and tied to the audit's expected landing (45.064, 61.664). Bound-based, not narrative-based.
- **Does not delete cycle 234 artefacts** — correct; the audit links them for the leakage record.

## Concerns / notes (non-blocking)
- The audit's "expected landing" of 45.064 (float) to 61.664 (calib μ-only) assumes held-out μ-only tracks calib μ-only. In practice the held-out `init_ppl_eval` could land anywhere in ~55-70 depending on which 100 sequences fall in `[200:300]`. That's fine — the plan already declines to assert a specific value. Just note in the JSON where held-out μ-only actually lands so future cycles have a reference.
- Object-identity disjointness (`set(map(id, ...))`) is sufficient because the ids are all freshly created tensors from the loader; no aliasing path exists. The additional tensor-equality check on the first pair is belt-and-braces — keep it.
- 3 seeds {42, 137, 2024} is the right ambition but Phase 4 alone is 2000 Adam steps per seed. If wall time becomes the binder, SEED=42 alone with `single_seed=true` is acceptable per the plan — just do not silently drop seeds 2/3 without flagging.
- The `base_c_proj_0_unchanged == True` check guards against an `LoraConv1D` bug where the base weight gets wrapped into an nn.Parameter that later receives gradients. Keep it; cheap and catches a real failure mode.
- Verdict thresholds are reasonable given 45.064 (float) lower bound and 61.664 (μ-only calib) upper bound: 48 gives ~3 PPL slack above float for CONFIRMED, 58 gives ~4 PPL slack below μ-only for REFUTED. The inconclusive band (48-58) is wide enough that a marginal result won't be overclaimed.

## Decision
**APPROVE as written.** No changes required. Execute per plan.
