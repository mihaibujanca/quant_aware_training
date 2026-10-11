# Cycle 230 Audit — Item 240 (K=8/4/2/1 Adam compression floor sweep)

**Status: PASS**

## Verdict check
- Criteria: CONFIRMED if K=4 best_mean ≤ 59.28 (+10% of float); REFUTED if K=8 ≥ 61.98
  (+15%); else INCONCLUSIVE.
- Reported: K=8 = 56.323 ± 0.011, K=4 = 57.328 ± 0.026, K=2 = 58.289 ± 0.035,
  K=1 = 59.245 ± 0.029. Float = 53.895.
- 57.328 ≤ 59.28 → CONFIRMED. K=1 (59.245) also sits inside the +10% band —
  independent confirmation that there is no cliff. Verdict matches.

## Implementation review (K=1 is the crux)
- `CompressedMLP` registers `fc_weight`/`fc_bias` as buffers (frozen), exposes
  `W_out ∈ R^{768×K}` and `b_out ∈ R^{768}` as `nn.Parameter`. `fit_block_lstsq`
  uses `scores.topk(K).indices` and indexes `W_fc[:, topk_idx]` so the forward
  path literally carries only `K` intermediate channels — no silent fallback.
- At K=1: fc path is `(B,T,768) @ (768,1) + (1,)` → gelu → `(B,T,1)`, then
  `@ (1,768) + (768,)` → `(B,T,768)`. `W_out.T` broadcasts the scalar gate onto
  a free 768-dim direction; `b_out` absorbs any constant residual. Trainable
  count 3·(1·768+768) = 4,608 matches the script (`per_block = K*768 + 768`).
- Trainable surface is exactly `{W_out, b_out}` × 3 blocks: all model params
  are frozen via `param.requires_grad_(False)` then only the six tensors are
  re-enabled (lines 149-157). `W_fc`/`b_fc` are buffers so grads cannot leak.
- `lstsq` at K=1 fits a 2-column design `[act_sub | 1]` against `proj_out`
  shape `(N,768)` — solution rows become `W_out` (768×1) and `b_out` (768).
  Init PPL at K=1 = 71.409 (worst of the sweep, as expected). Adam then
  absorbs 12.16 PPL — comparable in magnitude to K=8's 12.90 PPL recovery.
- Best-state restore round-trips (assert `|restored - best| < 1e-3`) for all
  12 seed×K runs.
- Per-seed std 0.011-0.035; identical init per K across seeds (init_std = 0)
  because lstsq init is deterministic, only Adam trajectory varies.

## Calibration / eval disjointness
- `all_ids[:200]` calib, `all_ids[200:300]` eval — same slicing as cycles
  223/225/229, no leakage. Float baseline 53.895 matches cycle 225 exactly.

## Is the "no cliff" finding real?
- Yes. The W_out+b_out surface at K=1 is not a near-identity hack: the FFN
  contribution is `scalar_gate(x) · v + c` for a learned residual-stream
  direction `v` and bias `c`. That is a genuine 2-parameter-per-dimension
  additive update to the residual stream and is functionally rich enough for
  three GPT-2 blocks where the cascade ceiling (cycle 221: 59.5 at 6 blocks
  K=512) is already structural, not channel-bound.
- Cross-config check: monotone in K (8→4→2→1 gives 56.3→57.3→58.3→59.2 with
  ~1.0 PPL per halving), consistent with the K=512→16 curve extrapolation
  (~1.5-2.2 PPL per halving in the high-K regime). No discontinuity.
- Floor diagnostic (|best-init| ≥ 1.0): all four K values pass with Δ ≈ 12 PPL.
  None approach 2× float (107.79).

## Extended efficiency curve
Full monotone curve K ∈ {512..1} → {46.556, 47.509, 48.989, 50.781, 53.038,
55.133, 56.323, 57.328, 58.289, 59.245}. Float is crossed near K=32. The
sub-16 regime costs ~1.0-1.2 PPL per halving vs ~1.5-2.3 PPL in the high-K
regime — slope actually *flattens* at low K, not steepens.

## Caveats
- "3-block W_out+b_out, blocks {1,4,7}" is a narrow configuration. The cliff
  could still exist at (a) 6 blocks, (b) deeper blocks, (c) denser positions.
- "Beats float at K=32" is calibration-aligned fine-tuning, not pure
  compression — same carryover caveat from cycles 206/210/223/225.
- K=1 at 59.245 is suspiciously close to the 6-block K=512 joint-Adam ceiling
  59.5 (cycle 221). Could be coincidence or could indicate both regimes are
  hitting the same structural floor.

## Bookkeeping
- Manifest, exec_result, JSON, summary consistent (item=240, cycle=230,
  outcome=confirmed, numbers aligned). Agenda moved to PAST/Cycle 230.

## Recommendations
- Next: item 241 (P0) — 6-block K=128 to test whether the near-log-linear
  curve survives depth (vs cycle 221's 6-block ceiling at 59.5).
- Quantize the K=1 W_out matrices (768×1) to 4-bit / ternary / binary and
  see if the 3-block PPL stays ≤ 58. At K=1 the per-channel quantiser has
  almost nothing to corrupt.
- Held-out test set separate from calib AND eval for any "beats float at K=32"
  claim going forward.
- Reframe "below-float compression at matched budget" as "PPL recovered per
  trainable param": K=1 leads at ≈2.6e-3 PPL/param of recovery from init.
