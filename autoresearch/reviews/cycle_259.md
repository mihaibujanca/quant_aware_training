# Cycle 259 Review — Item 270

**Decision: APPROVE**

## Summary
Pure analysis cycle (0 trainable params). Single forward pass each for GELU baseline and ReLU+b_leak (cycle 258 recipe, per-channel intermediate-space first-moment correction) on the same 200×128 wikitext-2 calib slice. Capture residual-stream trajectories at all 13 layer boundaries, compute per-token d_l(x) = ‖x_G^l − x_R^l‖_G under the pullback metric G = U^T U, report median/p10/p90 vs l, fit log-linear/linear/saturating curves to the median growth law, and report per-layer median cos(d_l, d_{l+1}) over the (B·T) token axis as the per-token analogue of cycle 189's aggregate cosine.

## Checks
- **Brief alignment**: matches item 270 verbatim. PASS.
- **Patch reuse**: plan reuses cycle 258's `ReluPlusBleak(b_leak_l)` module verbatim (per-block per-channel b_leak = E_x[gelu(z) − relu(z)] in the 3072-dim intermediate space; 12 distinct instances, cycle 228 distinctness discipline). PASS.
- **Calib slice**: 200 wikitext-2 val × 128 tokens, seed 42 — matches cycle 220 / 228 / 252 / 258. PASS.
- **Geometry calls**: `collect_residual_trajectory(model, ids)` → 13 tensors each [B, T, 768]; `pullback_norm(d_l, U)` per layer; `pullback_inner(d_l, d_{l+1}, U)` per cos. Token-order preservation across A/B variants is explicit (same seed, same `calib_ids` list, same per-seq loop order). PASS.
- **Flatten order**: per-layer `per_token_d_l` is shape [B, T] and flattened to [B·T] via `.reshape(-1)` — consistent across variants and layers so median/p10/p90 are over the same 25,600 tokens. PASS.
- **Growth-law fits**: 3 curve fits (log-linear via lstsq on `log(median)`; linear via lstsq on `median`; saturating via grid over `b ∈ {0.05, ..., 1.0}`, closed-form `a = median[0]`, `c = median[L] − a`). Best-fit = min SSR. Report `b_loglin` estimated over the l ∈ [0, 6] subset for the verdict threshold. PASS.
- **Verdict thresholds**: from agenda item 270 (CONFIRMED log-fit growth ≥ 0.20/layer in l ∈ [0, 6] AND median per-token cos ≥ 0.6 in ≥4 of those layers; REFUTED sub-linear growth OR cos ≤ 0.2 in ≥3 early layers). PASS.
- **Output namespacing**: `cycle_259_*` paths, no collision risk. PASS.
- **Run cap**: ~5 min on MPS — two forward passes × 200 short seqs + per-layer pullback_norm / pullback_inner (light). PASS.

## Risks
- **Memory blow-up on `v @ U^T`**: `pullback_norm` expands [B, T, 768] → [B, T, 50257]. Per-layer peak for full calib at float32: 200·128·50257·4 ≈ 5.15 GB. Mitigation stated in plan: either process per-sequence inside the layer loop (200 × per-seq peak 26 MB, totally fine) or per-minibatch of 8 seqs (~200 MB). Script should use per-sequence to be safe on MPS.
- **Layer 0 d_l**: `x_G^0` and `x_R^0` are both the embedding+positional output (before any block), so d_0 should be ≈ 0 for identical inputs. This is a trivial sanity check (first layer of every variant output of `transformer.drop` is identical). The script should print d_0 so the auditor can verify ≈ 0.
- **Log-linear fit on log(0)**: if d_0 ≈ 0 the log-linear regression will blow up. Mitigation: fit log-linear on l ∈ [1, 12] and still report the b-constant over the l ∈ [1, 6] subset for verdict clause 1. The agenda's "l ∈ [0, 6]" refers to the descriptive range of layer indices where the growth is evaluated — the fit itself should use l ≥ 1 once we know d_0 is near-zero by construction. This is a principled workaround, not a methodology cheat — and should be noted in both the script and the summary.
- **Per-token cos at d_0 ≈ 0**: cos(d_0, d_1) is numerically undefined. The denominator is `pullback_norm(d_0) * pullback_norm(d_1)` ≈ 0 × something. Skip the cos[0] entry from the verdict count (the verdict talks about the "early layers" with non-trivial d), and evaluate cos coherence over cos indices 1..6 (layers 1..7).
- **fp precision**: aggregation uses `.double()` for regression and float32 for storage. Median/p10/p90 via `torch.quantile(... reshape(-1), q=...)`.
- **Distinct module instances**: cycle 228 bug discipline — instantiate 12 distinct `ReluPlusBleak` modules, assert `len(set(id(...))) == 12`.
- **Restore originals**: end with a PPL sanity check on the GELU baseline after restore.

## Decision
APPROVE — execute as planned. The d_0 ≈ 0 and log-fit-on-l≥1 clarifications above should be captured in the script's print statements and the summary so the audit can verify the log-linear growth constant is fit over l ∈ [1, 12] (not l ∈ [0, 12]) and evaluated against the threshold on the [1, 6] subset.
