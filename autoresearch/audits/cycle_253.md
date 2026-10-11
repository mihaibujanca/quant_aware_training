# Cycle 253 Audit — Item 263

**Status: PASS**
**Verdict: INCONCLUSIVE (confirmed)**

## Verdict logic
Item 263 rule: `≤46 confirmed, ≥47 refuted, else inconclusive`.
Observed: `best_mean = 46.5096 ± 0.0381` → `46.0 < 46.5096 < 47.0` → **INCONCLUSIVE** ✓.

## Per-seed reproducibility
- per_seed_best = [46.4718, 46.4953, 46.5618] (seeds 42, 137, 2024)
- per_seed_best_step = [4000, 4000, 4000] (boundary — no overshoot)
- final = best per seed within 1e-3 (best-state restore round-trip OK)
- Tight 3-seed cluster (std 0.0381); means and finals coincide.

## Δ vs prior baselines
- Δ vs cycle 210 K=512 [1,4,7] raw GELU (46.556) = **-0.046 PPL**, within seed std (0.038). Statistically inert.
- Δ vs cycle 246 all-12 patched K=128 (57.811) = -11.30 (sidesteps that failure mode but yields no compensating gain).
- Δ vs cycle 239 LoRA (47.692) = -1.18.
- Δ vs float (53.895) = -7.39 PPL (-13.70%).

## Sanity checks
- `mu_dict` has exactly 3 entries (blocks {1,4,7}), not 12. ✓
- Targeted patch: only blocks {1,4,7} → `GeluNonegCorrected`; other 9 raw GELU (assertion in code). ✓
- μ ≤ 0 elementwise (max_positive = 0.0 across all 3 blocks). ✓
- `post_compress_eval_ppl = 63.732` is ~13 PPL worse than cycle 210 lstsq init, but Adam fully recovers it — consistent with cycle 246 cold-start observation.
- Disjoint calib/eval by object id and tensor equality. ✓

## Interpretation
Targeted activation surgery on the compressed blocks alone is statistically indistinguishable from raw GELU at K=512 [1,4,7]. Combined with cycle 246 (all-12 patch REFUTED at K=128), this closes the `gelu_noneg + μ` activation-surgery line at both ends of the patching spectrum for the K∈{128,512} [1,4,7] composition direction.
