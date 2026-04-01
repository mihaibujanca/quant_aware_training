# Design Review — Cycle 186 (Item 195)

**Plan**: Fewer-block compression — GPT-2 FFN k=4 on 2 blocks vs 4 blocks
**Reviewer**: Claude Sonnet 4.6
**Date**: 2026-04-01

---

## R1: Hypothesis testability

**Pass, with a caveat.**

The hypothesis ("halving the number of compressed blocks reduces cascade depth and yields better PPL at half the compression scope") is testable as stated. Comparing best-2-block mean PPL against the 4-block k=4 baseline (~59.4 PPL) is a clean within-protocol comparison.

Caveat: the experiment conflates two things — block count and total compression scope. The 2-block runs compress half as many parameters at the same per-block ratio (k=4, hidden=512). The 4-block baseline compresses twice as many parameters. Any improvement in 2-block PPL could be attributed to (a) fewer cascade interactions, (b) simply leaving more of the model uncompressed (i.e., a model size effect). The plan acknowledges "half the compression scope" but the hypothesis attributes the mechanism to cascade depth. These cannot be separated from this design alone. This is acceptable for an exploratory experiment, but the verdict should be framed carefully.

---

## R2: Verdict calibration

**Concern — thresholds may be too optimistic.**

- CONFIRMED: best 2-block pair < 42 PPL
- REFUTED: all 2-block pairs > 48 PPL
- Inconclusive: 42–48 PPL

The 4-block k=4 baseline is ~59.4 PPL. A 2-block run compresses only 2 of 12 layers; even a naive uncompressed model scores ~29 PPL (teacher). The confirmed threshold of 42 is ambitious but reachable given the reduced compression scope. However:

1. The floor is not well-anchored. The plan lacks a "2-block unpolished" baseline, which would show how much of the PPL gain comes from having fewer blocks vs from the 500-step polish.
2. The REFUTED threshold of 48 is only ~11 PPL better than the 4-block baseline. If cascade depth is truly the primary driver (cycle 171 evidence), even the worst pair should clear 48. If REFUTED fires, the conclusion would be that cascade depth doesn't matter — but it could instead mean that the 2 most damaging blocks are still covered by the worst pair. The threshold is reasonable but the verdict logic is slightly underpowered.
3. The "best pair" selection introduces implicit multiple comparisons across 6 pairs. Selecting the minimum of 6 noisy PPL estimates and comparing against a fixed threshold inflates the chance of a CONFIRMED verdict when no true effect exists. With 3 seeds, PPL variance from cycle 185 should be checked — if std is ~3-5 PPL, a 6-comparison minimum could shift the apparent best pair by several PPL points.

---

## R3: Confounders and methodological concerns

**One material flaw, one minor concern.**

**Material**: The polish tokens are drawn from `corpus.probe_holdout`, the same corpus used for `_fit_output_projection` (lstsq). The output projection is fit on `layer.output_sample` (probe train), and polish runs on `corpus.probe_holdout`. This is consistent with cycle 185 practice, so not a new issue, but worth noting: the 200-sequence holdout set used for polish is small, and overfitting the c_proj weights to these 200 sequences cannot be ruled out. This affects all conditions equally, so comparative conclusions hold, but absolute PPL numbers should be interpreted cautiously.

**Minor**: The plan probes the atlas on ALL_BLOCKS=(1,4,7,11) once and then calls `_derive_topk_pack(atlas, list(pair), ...)` for each pair. This is correct — `_derive_topk_pack` filters `atlas.layer_geometries` by `block_ids`. Verified against the cycle 185 implementation: the loop `for layer in atlas.layer_geometries: if layer.layer_id not in block_ids: continue` will correctly subset. No bug here.

**Minor**: The polish seed formula `pair_seed = seed + pair[0]*100 + pair[1]` can collide for pairs from the same block set if any pair[0]*100 + pair[1] values coincide. For pairs from {1,4,7,11}: (1,4)→104, (1,7)→107, (1,11)→111, (4,7)→407, (4,11)→411, (7,11)→711. No collisions. Fine.

---

## R4: Statistical concerns

**Marginal — 3 seeds is thin for the selection protocol.**

3 seeds is consistent with prior cycles (185 etc.), so this is not a regression. However, the "best pair" selection from 6 options means the reported best-pair mean PPL is a minimum-of-6 statistic. With only 3 seeds, each mean PPL has non-trivial variance. The plan's fallback (use median if one seed diverges) reduces this slightly. Acceptable given the exploratory nature, but any CONFIRMED verdict near the 42 threshold should be treated with caution until replicated.

No explicit std threshold for flagging a diverged run. The plan says flag PPL > 200 but not what happens for, e.g., a run at PPL = 80 (possible if a bad pair hits a problematic block combo). Consider flagging pairs where std_across_seeds > 10 as unreliable.

---

## R5: Implementation pitfalls

**Two issues not flagged in the plan.**

**R5a — `swap_compressed_ffns` return type**: The function returns `nn.Module` (verified in library: `return _replace_ffn_modules(model, ...)`) but the plan calls `student.eval()` and passes `student` to `_joint_polish`. The reference code in cycle 185 does this correctly (`student = swap_compressed_ffns(...); student.eval()`). The plan's pseudocode matches this pattern, so no bug — but the plan doesn't note that `_joint_polish` calls `student_model.train()` and then `student_model.eval()` internally, so the caller's `.eval()` call before `_joint_polish` is harmless but redundant.

**R5b — Polish trains `c_proj` only across all compressed blocks simultaneously.** For the 2-block pairs, this means `_joint_polish` will iterate over `block_ids=list(pair)` and find 2 compressed MLP modules. If `swap_compressed_ffns` replaces only those 2 blocks, the other 10 blocks remain as original GPT-2 MLP modules (not `CompressedFfnLayer`). The `_find_mlp` + `c_proj` attribute access should still work on original GPT-2 MLP modules (`mlp.c_proj` exists in GPT-2), meaning the polish could inadvertently train the uncompressed blocks' output projections. **This is a genuine bug risk.** The plan should specify that `_joint_polish` must only update parameters in the compressed blocks, not any GPT-2 block that happens to be named in `block_ids`. Since `block_ids` passed to `_joint_polish` matches the compressed pair exactly, the risk only materializes if the implementation incorrectly passes `ALL_BLOCKS` to polish instead of the pair. The plan's pseudocode passes `list(pair)` — correct. Flag for implementation review.

**R5c — JSON key format**: The plan uses `str(pair)` as the dict key, e.g. `"(1, 4)"`. This is readable but not portable across Python versions (tuple `__repr__` format is stable in practice for small int tuples). Acceptable, but noting for the implementer.

---

## Summary

| Criterion | Status |
|-----------|--------|
| R1: Testable | Pass (mechanism confounded with scope, acceptable for exploratory) |
| R2: Calibration | Concern — best-pair selection inflates CONFIRMED risk; add unpolished 2-block baseline |
| R3: Confounders | Minor — polish on holdout is consistent with prior cycles |
| R4: Statistics | Marginal — 3 seeds thin for min-of-6 selection; flag high-std pairs |
| R5: Pitfalls | R5b is material: verify `_joint_polish` does not touch uncompressed blocks |

**Recommendation: PROCEED** — the design is sound enough to run. The selection bias concern (R2) and R5b should be noted in the brief and verified during implementation, not treated as blockers.
