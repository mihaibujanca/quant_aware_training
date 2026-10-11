# Cycle 217 Design Review — Item 231

## Verdict: APPROVE

## R1: Scope and duplication
PASS. Cycles 184/188/190/193 measured output-level PPL/residual stats under GELU→ReLU swap. None performed per-token Δ SVD nor vocabulary projection of right singular vectors. The per-token direction structure of Δ is new territory. P1 check passes.

## R2: Method soundness
All critical mechanics verified:

- **Single-block swap (not full ReLU model).** Step 4 registers a forward pre-hook on `model.transformer.h[l].mlp.c_fc` during ONE fp32 forward pass, capturing the float (GELU-trajectory) residual stream input to each block independently. The ReLU branch is evaluated at the same captured input. Brief explicitly states "We're not running a fully-ReLU model." Correct.
- **Hook target.** `mlp.c_fc` input == `ln_2(residual_stream)` in HF GPT-2 (`GPT2Block.forward`: `hidden_states = self.ln_2(hidden_states); feed_forward_hidden_states = self.mlp(hidden_states)`). Correct. MP7 does NOT apply: the SVD target is `delta_out` (MLP output delta in unnormalized space), not a geometric property of a post-LN representation. The LN2 entry is simply where Δ is defined as a function of the layer's actual input.
- **Conv1D semantics.** Plan uses `block.mlp.c_fc(x)` and `block.mlp.c_proj(...)` — HF Conv1D handles the (in_features, out_features) shape correctly. Avoids manual weight/bias sign errors.
- **Bias cancellation.** Step 5e computes `delta_out = c_proj(delta_post) - c_proj.bias`, equivalent to `delta_post @ c_proj.weight`. Bias is a constant across tokens and would pollute rank_95 only with a single mean offset; subtraction is cleaner and explicitly acknowledged.
- **Pure ReLU.** Step 5c uses `F.relu(preact)` with no bias correction. Brief §Constraints specifies pure GELU vs pure ReLU, not the cycle-184 ReLU+bias variant. Correct and cleaner as a geometric decomposition.
- **SVD convention.** `Vh[k]` is the k-th right singular vector in R^768. `torch.linalg.svd(delta_out, full_matrices=False)` with delta_out shape (25600, 768) returns Vh of shape (768, 768) where rows are right singular vectors. `Vh[k]` is correct.
- **Vocabulary projection.** `model.lm_head.weight @ v` with lm_head.weight shape (50257, 768) and v shape (768,) → (50257,) logit-impact vector. topk ±10 with tokenizer.decode gives boost/suppress lists. Sign ambiguity of v is handled by reporting both tails.
- **Cross-block alignment.** `Vh[:3]` shape (3, 768), `U_next[:, :20]` shape (768, 20), product (3, 20), `svdvals` returns length min(3,20)=3. These are the cosines of principal angles between the 3-dim Δ direction subspace and the 20-dim reading subspace. Correct. Random baseline with 3 Gaussian unit vectors against the same U_next20 is the right null.
- **Sample size.** 25600 tokens for a 768-dim SVD is 33× oversampled, far above MP3's 2× floor. Rank and energy estimates will be well-resolved.

## R3: Pitfall audit
- MP1 N/A (no threshold-on-null).
- MP2 N/A (activation regime is exactly what's being measured).
- MP3 PASS (25600 ≫ 768).
- MP4 PASS (hypothesis and domain are same model family).
- MP5 PASS (single forward pass; no training).
- MP6 PASS (rank_95 and the ti/td ratio vary freely in [1,768] and [0,1]; verdict logic has a meaningful discriminating range).
- MP7 PASS (see R2 — Δ is measured in MLP output space, not on a renormalized representation; Δ norms and rank are function-output properties, not data-manifold properties).
- MP8 PASS (this IS a functional-rank measurement: SVD on the per-token output delta is exactly the functional-rank concept).

## R4: Verdict logic discriminability
- Confirmed band: rank_95 ≤ 64 in ≥6/12 blocks AND interpretable clusters.
- Refuted band: rank_95 > 400 in all blocks.
- There is a wide inconclusive zone (64-400 in some blocks) which is appropriate for an interpretability item where the quantitative threshold is a proxy for the real claim.
- Semantic interpretability is flagged as "executor prints lists for audit" — plan correctly does not try to programmatically score semantic clusters and instead falls back to Inconclusive if it can't verify them. Honest.

## R5: Pre-flight
- P1/P2/P3/P4 all pass. One forward pass of GPT-2 small on 25600 tokens + 12 × SVD(25600, 768) + 11 × SVD(768, 3072) is well under 30 min on CPU, seconds on GPU.
- Validation asserts (shape, allclose sanity, delta norm scale, rank_95 bounds) are present.
- OOM fallback (token chunking) and slow-SVD fallback (Gram 768×768) are realistic.

## Minor suggestions (non-blocking)
1. Step 5e: `delta_post @ c_proj.weight` is slightly cleaner than `c_proj(delta_post) - c_proj.bias` (no dropout module in the path, no bias arithmetic). Either works; plan already notes equivalence.
2. Consider also reporting the singular value spectrum's top-20 values per block in JSON (plan already does via `sv_energy`). Good.
3. For the cross-block alignment, reporting mean principal cosine across the 11 adjacent block pairs alongside per-pair values would make the cycle-216-comparison cleaner (cycle 216 found 0.649 avg top-5 for column-space alignment). Not required.
4. Sign of Vh[k]: plan already handles by reporting both top-boost and top-suppress (sign-symmetric). Good.

## Conclusion
Plan is mechanically sound, addresses the right question (per-token direction structure of the GELU→ReLU delta), respects the pitfall catalogue, and has sensible fallbacks. The single-block-swap framing is stated explicitly in both plan Step 4 and brief Constraints. APPROVE.
