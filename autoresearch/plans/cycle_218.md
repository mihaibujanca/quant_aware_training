# Task

Cycle 218 — Item 232 (P0): "Where does Δ point? — Δ alignment with next block's reading directions."
Extends cycle 217. For each block l, SVD the **centred** Δ̃_l = Δ_l − mean(Δ_l) and measure cross-block alignment of its top right singular vectors with the next block's c_fc, c_attn Q/K/V reading subspaces, and the residual-stream PCs at l+1 input. Re-run vocab projection on centred Δ̃.

# Hypothesis

Item 232 verbatim criteria:
- **Confirmed if**: "principal angles between top-20 Δ directions and top-100 W_up rows of block l+1 are smaller than the random baseline (drawn from Gaussian vectors of the same dimension) with p < 0.01 in ≥ 8 of 11 cross-block pairs."
- **Refuted if**: "Δ's top directions are orthogonal to next block's readers (alignment not distinguishable from random)."
- **Inconclusive if**: "weak but present alignment."

**Audit modification (cycle 217 audit, Recommendation 1+2)**: use CENTRED Δ̃ = Δ − mean(Δ) per block for all SVD operations, and extend the downstream reader set to {c_fc(l+1), c_attn(l+1) Q/K/V, residual PCs at l+1 input}. Significance measured as principal-cosine sum exceeding random baseline mean + 3σ (≈ p < 0.01). Verdict fires if ≥ 8 of 11 cross-block pairs are significant against **at least one** downstream reader.

# Pre-flight Results

- **P1 duplicate**: cycle 217 measured raw Δ; this measures centred Δ̃ plus extended Q/K/V/residual readers. NEW.
- **P2 dependency**: 232 depends on 231 (INCONCLUSIVE). Data is usable — we recompute Δ SVD on centred Δ̃ in this cycle. OK.
- **P3 ≤30 min**: single forward pass on GPT-2 small + 12 block SVDs + 11 cross-block alignments. OK.
- **P4 program/prompts**: no changes.

# Steps

1. Create `experiments/delta_centred_alignment_cycle218.py`. Copy data loader, hook setup, and Δ computation from `experiments/delta_per_token_cycle217.py`.
2. For each block l = 0..11:
   a. Compute Δ_l ∈ R^(n_tokens, 768) exactly as in 217 (`(gelu_post − relu_post) @ c_proj.weight`).
   b. `mean_delta_l = Δ_l.mean(0)`; `Δ̃_l = Δ_l − mean_delta_l`.
   c. `U, S, Vh = torch.linalg.svd(Δ̃_l, full_matrices=False)`. Compute `rank_95`, `rank_99`, `σ₀`, `σ₁`.
   d. `V_top20_l = Vh[:20]` ∈ R^(20, 768).
3. For each cross-block pair l = 0..10 → l+1, build downstream reader bases in R^768:
   - **c_fc**: `W_fc = h[l+1].mlp.c_fc.weight` shape (768, 3072). `U_fc, _, _ = svd(W_fc, full_matrices=False)`. `U_fc_top20 = U_fc[:, :20]` ∈ R^(768, 20).
   - **c_attn Q/K/V**: `W_attn = h[l+1].attn.c_attn.weight` shape (768, 2304). Slice `Q = W_attn[:, 0:768]`, `K = W_attn[:, 768:1536]`, `V = W_attn[:, 1536:2304]`. For each, `U_q, _, _ = svd(Q, full_matrices=False)`, take `[:, :20]`. Same for K, V. Assert shapes; if Conv1D layout is transposed, adjust.
   - **Residual PCs**: capture block l+1 pre-LN1 input via `h[l+1].register_forward_pre_hook(lambda m, args: stash[l+1].append(args[0].detach()))`. Flatten to (n_tokens, 768), centre, SVD, take top-20 right SVs → `U_res_top20` ∈ R^(768, 20).
4. Principal cosines per pair per reader:
   - `M = V_top20_l @ U_reader` ∈ R^(20, 20); `cos = torch.linalg.svdvals(M)` (length 20).
   - Record `mean_cos`, `top5_cos_mean`, `cos_sum = cos.sum()`.
5. Random baseline: 3 seeds. For each, draw Gaussian (768, 20), `Q, _ = torch.linalg.qr(G)` → random orthonormal basis. Compute `svdvals(V_top20_l @ Q)` vs each reader's `U_reader`, and cross `Q_V_random @ U_reader`. Record mean and std of `cos_sum` across seeds per (block, reader) — this is the "random replacement of Δ̃ top-20" baseline.
6. Significance per (block l, reader r): `significant[l, r] = (cos_sum[l, r] > random_mean[l, r] + 3 * random_std[l, r])`. Per-pair verdict: `any(significant[l, :])` across readers {c_fc, Q, K, V, residual}.
7. Aggregate verdict: count `n_sig = sum_l any(significant[l, :])` over l = 0..10 (11 pairs). Confirmed if `n_sig ≥ 8`. Refuted if `n_sig == 0`. Inconclusive otherwise.
8. Vocab interpretability on centred Δ̃: project `Vh[:3]` of Δ̃_l onto `lm_head.weight` per block. Print top-10 boost/supp tokens. Heuristic: count blocks where top-10 boost list contains **>3 alphanumeric word tokens that are not stopwords** (exclude `" the"`, `" ,"`, `"."`, `" and"`, `" a"`, `" of"`, `" to"`, `" in"`, `" -"`). Report count; plausible semantic structure if ≥ 3 blocks.
9. Write `autoresearch/results/cycle_218_delta_centred_alignment.json` with per-block {rank_95, rank_99, σ₀, σ₁, mean_delta_norm}, per-pair per-reader {cos_sum, mean_cos, top5_cos_mean, random_mean, random_std, significant}, and top-10 vocab lists.
10. Write `autoresearch/results/cycle_218_summary.md` with verdict, headline numbers (avg rank_95 of Δ̃ vs raw Δ, avg top5_cos_mean per reader, n_sig / 11, semantic-plausible block count), and one paragraph plain-language interpretation.

# Validation

- **Sanity 1**: rank_95(Δ̃) should be **much higher** than rank_95(raw Δ) from cycle 217 `[10, 132, 1, 1, 1, 1, 1, 8, 73, 110, 1, 1]`. Expect tens to hundreds in most blocks (the raw rank_95 = 1 was mean dominance).
- **Sanity 2**: σ₀(Δ̃) ≈ σ₁(raw Δ) from cycle 217 (≈ 200-350 in striking blocks), since removing the mean removes the dominant axis.
- **Sanity 3**: residual capture occurs **BEFORE** `h[l+1].ln_1`, i.e. at the block forward pre-hook, not after LN1. Assert the captured tensor equals the layer input by checking shape `(batch, seq, 768)`.
- **Sanity 4**: `c_attn.weight` is Conv1D layout (768, 2304); slicing `[:, :768]` gives Q columns in R^768. Assert `Q.shape == (768, 768)` and that `Q @ x` for `x.shape == (768,)` produces R^768.
- **Sanity 5**: Random baseline std > 0 (3 seeds); `cos_sum` of an orthonormal basis vs another orthonormal basis is in [0, 20].

# Artifacts

- `experiments/delta_centred_alignment_cycle218.py`
- `autoresearch/results/cycle_218_delta_centred_alignment.json`
- `autoresearch/results/cycle_218_summary.md`
- `autoresearch/results/cycle_218_manifest.txt` (written by execute agent)

# Fallbacks

- If `c_attn.weight` Conv1D layout causes a shape mismatch, assert the shape and transpose slicing accordingly (Conv1D stores weight transposed relative to Linear).
- If `register_forward_pre_hook` on `h[l+1]` captures the wrong tensor (e.g. tuple unpacking issue), hook `h[l+1].ln_1.register_forward_pre_hook` and take `args[0]` — this is still pre-LN1 input since LN1 is the first submodule.
- If random baseline std is suspiciously low (< 1e-3), increase to 10 seeds.
- If c_fc alignment alone satisfies the ≥8/11 criterion, still report all four readers for completeness per the audit recommendation.

# Update bookkeeping

- Move item 232 from `NEXT EXPERIMENTS / P0` to `PAST EXPERIMENTS / Cycle 218` in `autoresearch/agenda.md` (delete from NEXT to avoid the runner bookkeeping bug).
- Append TSV row to `autoresearch/cycles.tsv`.
- Commit: `[autoresearch] cycle 218: centred Δ̃ alignment with downstream readers`.
- Refresh `autoresearch/STATUS.md`.
