status=PASS-WITH-FLAGS
old_outcome=refuted
new_outcome=refuted
verdict_accepted=yes

## Flags

G4: Seed 42 teacher baseline PPL = 57.65, which is at the upper bound of the [15, 65] sanity check and substantially higher than the plan's stated null baseline of ~30 PPL. This is not a bug — GPT-2 evaluated on 200 sequences of an openwebtext sample with seq_len=128 can legitimately vary, and 57.65 is within the accepted range — but it means seed 42's polished PPL values (all >68) are measured against a weak baseline. The "working recipe" from prior cycles reports ~59.4 PPL at k=4, and seed 42 achieves 70.8 at k=4, which is ~11 PPL worse. The mean across seeds still holds (59.4 matches historical k=4 reference exactly), but the high per-seed variance (std ~8 PPL across seeds) makes seed-42-specific conclusions unreliable. Flagged as weak, not disqualifying.

G5: Seed seeding for the topk pack derivation and polish uses `seed + k` (e.g., 42+2=44, 42+4=46, ...). This means different k-values get different random states at the start of polish, which is fine for the sweep but means two seeds from different (seed, k) combinations could alias (e.g., seed=42, k=8 → 50; seed=7, k=8 → 15; seed=137, k=2 → 139). No collision occurs in this config (SEEDS={42,137,7}, K_VALUES={2,4,6,8}), but the pattern is fragile if seeds are extended. Minor.

G6: The PPL curve is monotonically decreasing (61.6→59.4→58.3→57.2 from k=2 to k=8) — the first condition for CONFIRMED is met. Only the PPL<45 threshold fails. The 4x parameter increase (12x→3x compression) buys only 4.4 PPL — a ~7% relative improvement. Per-seed monotonicity holds for all three seeds without exception. This is strong evidence that the composition barrier is a hard ceiling, not a capacity ceiling: each block is fit well in isolation but stacking degrades regardless of per-block width. The near-zero marginal return on parameters at k>4 (k=6 gains 1.1 PPL, k=8 gains 1.1 PPL — exactly constant) suggests the floor is not about width but about error accumulation across 4 stacked blocks. Worth characterizing via block-count sweep (item 195).

## Verdict logic check

CONFIRMED requires monotone AND (k6<45 OR k8<45). REFUTED requires k8>55. These branches are correctly ordered in code — CONFIRMED is checked first, then REFUTED, then INCONCLUSIVE. With k8=57.2>55, REFUTED fires correctly. The plan's criteria match the code exactly.

## Param formula check

Plan specifies: `per_block = hidden*768 + hidden + 768*hidden + 768`. Code computes: `hidden_dim * GPT2_HIDDEN + hidden_dim + GPT2_HIDDEN * hidden_dim + GPT2_HIDDEN`. These are identical. Computed ratios (~12x, ~6x, ~4x, ~3x) match plan expectations.

## Data split check

Polish trains on `corpus.probe_holdout`; evaluation uses `corpus.eval_val`. These are separate splits built from separate document windows. No eval leakage.

## Recommendation

Accept REFUTED verdict. The compression ratio curve is real and monotone, but the floor at ~57 PPL is independent of per-block width. Item 195 (block-count reduction) is the correct next experiment — if removing blocks recovers more PPL than doubling width, it confirms that composition is the limiting factor, not per-block capacity. If block count also shows diminishing returns, the approach is likely exhausted at this quality level.
