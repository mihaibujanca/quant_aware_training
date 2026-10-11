# Item
232 (P0): Where does Δ point? — Δ alignment with next block's reading directions.
For each block l, compute alignment of top SVs of CENTRED Δ̃_l = Δ_l − mean(Δ_l) with the next block's W_up rows AND with W_Q/W_K/W_V columns of the attention block sandwiched between l and l+1. Project Δ̃_l onto top PCs of the residual stream at l+1 input.

# Context
- Cycle 217 (item 231) measured raw Δ_l per token. Result corrected to INCONCLUSIVE because Δ is dominated by a per-block CONSTANT BIAS (TI/TD ratio 0.87-0.99, σ₁/σ₀ ≈ 0.03-0.09). Top SV0 was just the mean projected onto unigram-frequency tokens — no per-token signal.
- The audit explicitly recommends: "SVD CENTRED Δ̃ = Δ − mean(Δ) per block (rank_95, rank_99, σ₁, SV vocab projection); extend cross-block alignment to next-block W_Q/W_K/W_V left SVs".
- Cross-block W_down→c_fc alignment (cycle 216) avg top-5 = 0.649; raw Δ→c_fc next-block alignment (cycle 217) avg top-3 = 0.369 vs random 0.168. CENTRED Δ̃ alignment may be different.
- Item 232 in agenda: "principal angles between top-20 Δ directions and top-100 W_up rows of block l+1 are smaller than the random baseline with p < 0.01 in ≥ 8 of 11 cross-block pairs". This is the literal Confirmed criterion. The fix: use Δ̃ (centred) for all SVD operations.

# Dependencies
- Item 231 outputs (cycle 217) — provides the data hooks and structure. We can REUSE the script `experiments/delta_per_token_cycle217.py` as a starting point.
- Audit recommendation explicitly extends 232 to W_Q/W_K/W_V (the attention readers between block l and l+1).

# Key Files
- experiments/delta_per_token_cycle217.py — start point. Reuse data loader, hooks, Δ computation.
- HF GPT-2 attention block: model.transformer.h[l+1].attn.c_attn (Conv1D, weight shape (768, 2304)) — fused QKV. Slice into 768-dim columns: q = c_attn.weight[:, :768], k = c_attn.weight[:, 768:1536], v = c_attn.weight[:, 1536:2304]. Each is shape (768, 768).
- For "rows of W_up_{l+1}": W_up = c_fc.weight in block l+1, shape (768, 3072). The "reading directions in R^768" are the LEFT singular vectors of c_fc.weight. Note: "rows of W_up" in agenda phrasing means the rows after we view it as 768→3072 — those rows are 3072 vectors of length 768. But for principal angle analysis we use the top SVs as in cycle 217.

# Constraints
- Centring: Δ̃_l = Δ_l − mean(Δ_l, dim=0).
- SVD: torch.linalg.svd(Δ̃_l, full_matrices=False). Vh̃[k] is the k-th right singular vector of the centred Δ.
- Compute rank_95, rank_99 of centred Δ̃ (likely much higher than the raw rank_95=1).
- Top σ₁ of centred Δ̃ vs σ₀ of raw Δ — quantify the per-token signal magnitude.
- Cross-block alignment using top-20 right SVs of Δ̃_l:
  (a) vs c_fc(l+1) top-20 LEFT SVs (the W_up writing readers — same as cycle 217)
  (b) vs c_attn(l+1).weight Q/K/V slices, top-20 LEFT SVs each
  (c) vs top-20 PCs of the residual stream at l+1 input (compute via SVD of the captured residual at block l+1's pre-LN1)
- Random baseline: top-20 random unit vectors in R^768, 3 seeds.
- Confirmed if alignment with at least one downstream reader subspace is significantly above random in ≥8 of 11 cross-block pairs (significance via comparing principal cosine sums, with p<0.01 estimated from random baseline std).
- Re-project top-3 right SVs of Δ̃ onto W_U for vocabulary interpretability — this is the corrected version of cycle 217's failed semantic check. If the centred top SVs project onto SEMANTIC clusters of vocab (not just stopwords), we have what 231 was looking for.

# Handoff
Plan agent: design exact computation. Reuse cycle 217 script. New script `experiments/delta_centred_alignment_cycle218.py`. The script must:
1. Repeat the cycle 217 hook setup and Δ computation (reuse code where possible — copy-paste is fine).
2. Compute centred Δ̃ per block.
3. Compute rank_95/rank_99 of Δ̃ per block.
4. SVD Δ̃ → top-20 right SVs.
5. Compute cross-block principal cosines vs (a) c_fc(l+1) left SVs top-20, (b) c_attn(l+1) Q/K/V left SVs top-20 each, (c) residual PCs at l+1 input top-20. Random baseline for each.
6. Project top-3 right SVs of Δ̃ onto W_U → vocab interpretability with the same heuristic as cycle 217 (look for semantic clusters, not just stopwords).
7. Verdict via the agenda criterion: significant alignment in ≥8 of 11 pairs to AT LEAST ONE downstream reader subspace.
8. Output JSON + summary markdown.

Tight scope. ≤30 min wall time.
