# Cycle 246 Audit — Item 256

**Claim:** gelu_noneg + μ + LoRA r=8 stacked with K=128 W_out topk+lstsq on [1,4,7] → REFUTED at best_mean=57.811 ± 0.277 (3 seeds, 2000 Adam steps, LR=1e-4).

**Thresholds:** confirm ≤ 47.0, refute ≥ 49.5, else inconclusive.

## Phase 1 — verdict check
57.811 ≥ 49.5 → **REFUTED** ✓. Std 0.277 tight across seeds {42, 137, 2024}; per-seed best all land at step 2000 (57.899 / 57.437 / 58.097). Deltas: +8.822 vs c223 K=128 (48.989), +10.119 vs c239 LoRA-only (47.692), +3.916 vs float held-out (53.895). Verdict label is correct.

## Phase 2 — quality checks
- **Install order** matches the plan: μ precompute on CALIB → patch 12 × GeluNonegCorrected(μ_l) → K=128 topk+lstsq on patched block I/O → LoRA r=8 on 9 uncompressed c_proj → joint Adam.
- **Trainable params** 595,200 = 276,480 (9 float LoRA) + 21,504 (3 compressed LoRA) + 297,216 (3 W_out+b_out). Verified by breakdown field.
- **Disjoint splits** calib=all_ids[:200], eval=all_ids[200:300], SEQ_LEN=128, matching cycles 230/239.
- **μ sanity:** all 12 blocks report `n_positive_entries=0` and `max ≤ 0` (up to float epsilon, e.g. −2.84e-11 at block 1). μ ≤ 0 invariant holds.
- **LoRA-zero init** reproducible: post_compress_eval_ppl=106.742 identical across all 3 seeds → B=0 init honoured, seed variance only enters during Adam.
- **Init delta:** K=128 lstsq init against the bump-removed target is 106.74 vs ~50.8 for raw GELU in cycle 223 — init is ~56 PPL worse, and 2000 Adam steps recover only ~49 PPL of that deficit. Post-μ EVAL alone is 74.30, so the K=128 lstsq step on the patched activation actively *damages* the mu-only baseline before Adam starts.

## Phase 3 — interpretation
This is a clean structural refutation: the two mechanisms do not compose. The bump-removed `relu(gelu(z))+μ` target is *harder*, not easier, for K=128 topk+lstsq — directly contradicting the plan hypothesis. Activation-level bump removal and channel-level topk+lstsq share the same linearisation budget. Follow-ups flagged in the manifest (K=512 variant, install-order swap, topk-basis mismatch diagnostic, trainable μ) are well-scoped.

**Audit status:** PASS. Interesting: composition failure between orthogonal-looking mechanisms is a load-bearing finding for the overall compression program.
