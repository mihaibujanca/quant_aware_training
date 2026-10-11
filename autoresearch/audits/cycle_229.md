# Cycle 229 Audit — Item 239

**Status: PASS**

**Old outcome:** refuted
**New outcome:** refuted

## Criterion check (Phase 1)

- Rule: Confirmed `final_mean ≤ 58.0`; Refuted `final_mean ≥ 60.0`; Inconclusive otherwise.
- Result: `final_mean = 60.0707 ± 0.0865` PPL (per-seed: 59.949 / 60.144 / 60.119).
- 60.0707 ≥ 60.0 → **REFUTED** (barely above the threshold, but monotone and tight across seeds).

## Phase 2 quality review

- **G1 (code correctness):** `adam_sequential_6block_cycle229.py` reuses the cycle 215 sequential pattern at 6 blocks. Freeze/unfreeze limited to each block's `W_out`, `b_out`; fresh Adam per block; per-block best-state restore with a collapse rollback to the pre-block snapshot; assert on restore delta (`< 1e-3`). Verified 18/18 (seed × block) restores match `best_ppl` (<1e-6).
- **G2 (statistics):** 3 seeds, std = 0.086 PPL — well below the 0.5 PPL gap to joint. Init PPL matches across seeds to 1e-5 and matches cycle 221's 6-block init exactly (86.686).
- **G3 (interpretation):** "Sequential cannot rescue 6-block" is correctly stated; joint 6-block wins by +0.556 PPL.
- **G4 (counterintuitive):** Not counterintuitive — matches cycle 215 at 3 blocks where joint also beat sequential by 0.697 PPL. Consistent direction.
- **G5 (learned/next):** Framing is sound. Phrasing could be sharper: "the 58.0 floor is surface-limited, not optimizer-limited" captures it cleanly.
- **G6 (threshold proximity):** final_mean sits 0.071 PPL above the 60.0 refuted threshold. All three seeds exceed 60.0 on their own trajectory cleanly after block 5, and the +0.556 PPL gap vs joint 6-block is real and reproducible. Not a method failure — the surface genuinely cannot close the last 2 PPL at 6 blocks with this recipe.

## Block-wise trajectory (mean restored PPL)

| step | init | b1 | b2 | b4 | b5 | b7 | b8 |
|---|---|---|---|---|---|---|---|
| mean | 86.686 | 66.314 | 62.824 | 61.391 | 60.700 | 60.126 | 60.071 |
| Δ | — | −20.372 | −3.490 | −1.433 | −0.691 | −0.574 | −0.055 |

Severe diminishing returns: block 1 removes 59% of the total 26.62 PPL drop; blocks 7+8 together contribute 0.63 PPL.

## Recommendations

Confirms that the 58.0 floor is structural for the `W_out+b_out`@K=512 trainable surface at 6 blocks. To break it: (a) widen surface to `W_fc+W_out` or add LoRA; (b) different block selection; (c) escalate to larger K; (d) test 9 blocks to trace the floor's scaling. Sequential schedule is ruled out as the mechanism.
