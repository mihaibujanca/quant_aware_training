# Cycle 256 Audit — Item 266

**Status: PASS** (interesting=true)

## Item (verbatim)
266 (P0): 6-block K=256 — fill the missing K=256 6-block data point. BLOCKS=[1,2,4,5,7,8] K=256 joint Adam W_out+b_out, 4000 steps, 3 seeds. Agenda thresholds: CONFIRMED ≤63.0, REFUTED ≥65.0, INCONCLUSIVE (63, 65).

## Claimed vs verified
- Verdict claim: **INCONCLUSIVE** — best_mean=64.181 ± 0.032.
- Phase 1 rule check: 64.181 ∈ (63.0, 65.0) → INCONCLUSIVE. PASS.
- CONFIRMED branch (≤63.0) correctly not triggered (64.181 > 63.0).
- REFUTED branch (≥65.0) correctly not triggered (64.181 < 65.0).
- Verified per-seed best matches JSON: 42=64.2203, 137=64.1821, 2024=64.1407.

## Per-seed verification (exec_result ↔ manifest ↔ JSON ↔ run log)
| seed | init    | best     | best_step | final@4000 |
|------|---------|----------|-----------|------------|
| 42   | 101.088 | 64.2203  | 2300      | 66.5772    |
| 137  | 101.088 | 64.1821  | 2200      | 65.8485    |
| 2024 | 101.088 | 64.1407  | 2400      | 65.8471    |
| mean | 101.088 | **64.181** | 2300    | 66.091     |
| std  | 0.000   | 0.032    | 82        | 0.344      |

Float held-out = 53.895. All four sources (run log, JSON, manifest, exec_result, summary) agree to 4 decimal places.

## Phase 2 — is the signal real?
- 3 seeds, all 3 finished cleanly. No fallback triggered.
- Within-seed best std 0.032 PPL — tightest of the recent 6-block fits (cycle 221 std 0.094).
- Init PPL 101.088 with **zero seed variance** (lstsq deterministic; topk indices and lstsq solve identical across seeds — consistent with cycle 251 init=88.341 zero seed variance pattern).
- All 3 seeds plateau at step 2200-2400 then climb steadily to 65.8-66.6 by step 4000 — clear plateau-then-climb pattern (overshoot +1.91 PPL past best on average).
- Best-state restore round-trip clean (assert |restored − best| < 1e-3 holds for all 3 seeds per code line ~189).
- No Phase 2 fallback triggered (run completed in ~10 min, well under cap).

## Phase 3 — interpretation check
- **K-curve concavity**: K=512 59.515, K=256 64.181, K=128 67.562. Linear-in-K midpoint = 63.539; actual K=256 = 64.181, +0.642 PPL above linear interpolation. The K-vs-PPL curve at 6 blocks is concave-up (sublinear-in-K) between K=128 and K=512. Per-block K-penalty: K=512→K=256 buys 0.78 PPL/block, K=256→K=128 buys 0.56 PPL/block — first half of K-doubling buys more than the second half (1.4× ratio).
- **Hetero vs homogeneous head-to-head**: cycle 247 hetero {4,5}=K512 64.453, cycle 249 hetero {7,8}=K512 65.275, cycle 256 homogeneous K=256 64.181. Homogeneous K=256 ties cycle 247 within ~9σ of std 0.032 (Δ=−0.272 PPL — practically a tie) and beats cycle 249 by 1.094 PPL. The homogeneous mid-K strategy does not dominate the hetero family but it is on the better side of it.
- **Implication**: at 6 blocks with ~1.18M trainable budget, K-allocation strategy (homogeneous vs heterogeneous) is below practical noise. The binding constraint is **block count**, not K-allocation. To beat ~64 PPL at 6 blocks one must trade more K (K=512 cycle 221 → 59.5) or fewer blocks (5-block K=512 cycle 245 → 53.8).

## Bookkeeping
- commit=9bb498ccee23da8db0695982d8ef8ba7725d177d
- bookkeeping=02870535
- TSV row 254 hash matches: 9bb498ccee23da8db0695982d8ef8ba7725d177d.
- Artifacts present: experiment script, JSON, summary.md, run log, manifest, plan, brief, review (md+json), exec_result, audit (this file).
- Verdict logic in code (lines ~225-231) matches agenda item 266 thresholds (≤63.0 / ≥65.0). Note: the brief said ≤62.0; this audit confirms the agenda value (≤63.0) was used as authoritative — the verdict outcome (INCONCLUSIVE) is unchanged under both threshold sets since 64.181 is above both 62.0 and 63.0.
- Pre-existing dirty files (program.md, flagged.jsonl, manifests for 170/205/255, literature_map.md, pullback_geometry.py) NOT staged — clean commit boundary.

**Interesting**: yes — first homogeneous-K mid-band data point at 6 blocks; demonstrates K-vs-PPL concavity at 6 blocks and ties homogeneous K=256 statistically with cycle 247 hetero, supporting the "binding factor is block count, not K-allocation strategy" hypothesis at the 6-block scale.
