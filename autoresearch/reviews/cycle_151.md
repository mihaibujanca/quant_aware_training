# Review — Cycle 151 (Item 161)

## Checklist
- [x] Critical fix: polish removed from `build_compressed_model()`. This was the cycle 142 failure mode.
- [x] Unpolished PPL assertion [75, 100] will catch if polish removal is incomplete.
- [x] LoRA protocol matches cycle 115: r=8, 2000 steps, lr=1e-4, CE loss on unpolished model.
- [x] 2-way decomposition (local vs cascade) is simpler and justified: cycle 127 showed attention subspace is near-null.
- [x] Per-block ablation (one at a time) correctly isolates block-level subspace contributions.
- [x] Pythagorean sanity check catches projection bugs.
- [x] Corrected PPL range [60, 75] is reasonable given cycle 115 achieved ~59.

## Risks
- **Hook placement**: cycle 142 hooks at block output (post-residual-add). This is correct for measuring the correction's effect on the residual stream. The plan inherits this without changes. OK.
- **Rank-8 SVD reconstruction of ablated components**: C_local projected onto the 8-dim error subspace has rank <= 8, so rank-8 SVD is lossless. C_cascade = C - C_local also rank <= 16 (sum of two rank-8 matrices), but the SVD truncation to rank 8 may lose some cascade energy. This is a minor concern — the plan should log the SVD truncation loss. Not blocking.
- **Single seed**: acceptable for screening given cycle 142's method failure. If results are clear, one seed suffices for the CONFIRMED/REFUTED verdict. If borderline, the INCONCLUSIVE path is available.

## Verdict
**PASS** — minimal, targeted fix of cycle 142's method failure. The only code change is deleting the polish loop. Risk profile is low.
