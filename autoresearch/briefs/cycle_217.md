# Item
231 (P0): Per-token Δ(x) = GELU_FFN(x) − ReLU_FFN(x) — rank, subspace, logit impact.
For 200-sequence OWT calibration batch, all 12 GPT-2 blocks. Compute Δ_l per token; SVD it; project top right singular vectors onto unembedding W_U for vocabulary interpretability and onto next block's W_up for cross-block alignment.

# Context
- Cycle 216 (REFUTED): W_down columns are NOT anti-podal pairs and have no k=32 cluster structure. Random on the sphere. BUT cross-block W_down(l) writing-subspace ↔ c_fc(l+1) reading-subspace has top-5 principal cosines averaging 0.649 — adjacent blocks share directions structurally, not via individual columns.
- This item is the first investigation of where Δ(x) actually points — earlier cycles 184, 188, 190, 193 showed shape change matters but didn't probe the direction structure.
- Plain-language goal: "the bump writes a low-rank token-dependent message into specific vocabulary directions."

# Dependencies
None — does not require item 230.

# Key Files
- HuggingFace `gpt2` model.
- For ReLU comparison activation: replace `gelu_new` with `relu` (or `relu+bias` to match cycle 184/190 — clarify in plan).
- For unembedding projection: `model.lm_head.weight` shape (50257, 768). Top-5 right singular vectors of Δ_l (each shape 768) projected onto W_U → vocab-space scores. Top-k boosted/suppressed tokens.
- For next-block W_up reading: c_fc.weight in block l+1, top-20 left singular vectors as in cycle 216.
- N=200 sequences, SEQ_LEN=128 → 25,600 tokens. Δ matrix shape (25600, 768) per block. SVD → singular values shape (768,).

# Constraints
- Per-token Δ_l(x) means: at the same residual stream input to block l, compute MLP output with GELU vs MLP output with ReLU. ReLU substitution must NOT include the bias correction unless we explicitly want that; use plain ReLU per cycle 184 baseline. (Cycle 190 used "GELU_neg + ReLU_pos hybrid" which is different. The plan should specify pure GELU vs pure ReLU as the cleanest decomposition.)
- Δ is computed AT EACH BLOCK INDEPENDENTLY: the residual stream input to block l is the FLOAT (GELU) trajectory. We're not running a fully-ReLU model; we're swapping just one block's MLP and measuring its single-block Δ. This isolates the direction signal of GELU's bump cleanly.
- Output: per-block rank_95 and rank_99 (smallest k such that cumsum of squared singular values reaches 95%/99% of total energy).
- Token-independent vs token-dependent split: ratio = ||mean(Δ)||² / mean(||Δ||²). 0 = pure noise; 1 = constant offset.
- Top-3 right singular vectors of Δ projected onto W_U: report top-10 boosted/suppressed vocabulary tokens for each.
- Top-3 right singular vectors of Δ_l projected onto top-20 left SVs of c_fc(l+1): cosines of principal angles. Compare to a random 768-dim baseline.
- Confirmed: rank_95(Δ) ≤ 64 in ≥6 of 12 blocks AND top SVs produce interpretable token clusters.
- Refuted: rank_95(Δ) > 400 in all blocks.
- Inconclusive: low rank but no semantic structure (or vice versa).
- Use ≥3 seeds: NOT applicable here — Δ is deterministic given the input; randomness only in OWT batch sampling. Use ONE fixed batch (seed 42 for sampling) but report token count and reproducibility caveat.

# Handoff
Plan agent: design exact computation. Make sure (a) the block input is the FLOAT residual stream at block l (collect once), (b) GELU vs ReLU branches use the SAME inputs and SAME W_fc/W_down, only the activation differs, (c) per-token Δ matrix shape is (n_tokens, 768), (d) SVD via torch.linalg.svd on Δ ∈ R^(n_tokens × 768) — singular values length min(n_tokens, 768) = 768, (e) rank_95/rank_99 from cumulative energy. For interpretability projection, present top-10 boost/suppress per top-3 SVs per block — that's 12 × 3 × 20 = 720 token labels in JSON. The summary markdown should highlight 3-5 most striking blocks. Use HF tokenizer.decode for token strings.
