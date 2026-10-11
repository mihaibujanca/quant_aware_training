# Audit — Cycle 249 / Item 259

**Status: PASS** — verdict REFUTED is correct and well-supported.

## Claim vs criteria
- Criteria (item 259): Confirmed ≤61.0, Refuted ≥65.0, Inconclusive 61–65.
- Claimed: `best_mean=65.275 ± 0.094` → 65.275 ≥ 65.0 → **REFUTED**. ✓

## Numerical cross-check (vs `cycle_249_hetero_k_complement.json`)
- Aggregate best: 65.27477728 → rounds to 65.275 ✓; std 0.0938 → 0.094 ✓.
- Per-seed best: 65.4020, 65.1784, 65.2439; all best_step=2200 ✓.
- Final@4000 mean 68.642; overshoot = 68.642 − 65.275 = +3.37 PPL (claim ✓; ~3× cycle 247's 1.08).
- Δ vs cycle 247 hetero(4,5) 64.453 = +0.822 (worse) ✓.
- Gap recovery = (67.562 − 65.275) / (67.562 − 59.515) = 2.287 / 8.047 = 0.2842 → 28.4% ✓ (cycle 247: 38.6% ✓).
- Trainable = 4·(128·768+768) + 2·(512·768+768) = 396,288 + 787,968 = **1,184,256** ✓.
- Per-block init residual MSE on blocks 7,8 dropped from 0.581/0.818 (cycle 247 K=128) to 0.415/0.579 (cycle 249 K=512) = −28.6% / −29.2% ✓ — local fit improved, as expected for 4× channel bandwidth, yet downstream PPL got worse.

## Interpretation
Phase-2 quality is high: 3 seeds, σ=0.094, deterministic best-step at 2200 across all seeds. The per-block local MSE improvement is real but fails to translate to PPL — confirming that per-block residual MSE is a local diagnostic that does NOT predict downstream cascade effect under heterogeneous-fit joint Adam. Combined with cycle 247 (INCONCLUSIVE at 64.453 via ΔPPL ranking from cycle 243), the pair closes the "localise the K-bandwidth bottleneck to 2-of-6 blocks at 1.18M budget" family: neither ranking breaks the 39% gap-recovery ceiling. Also notable: cycle 249's blocks 7,8 K=512 surfaces are substantially more overfit-prone than cycle 247's blocks 4,5 K=512 under identical Adam schedule (overshoot 3.37 vs 1.08).

**Interesting: true** — double-negative localisation result cleanly closes the sub-family.
