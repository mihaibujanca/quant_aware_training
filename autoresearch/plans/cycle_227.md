# Task

Cycle 227 — Item 237 (P0): Cycle 218 specificity control. Measure centred Δ̃_l alignment with c_fc(j) for ALL j > l (not just j=l+1) to test whether the cycle 218 confirmation reflects adjacency-specific inter-block communication or a generic universal reading basis.

# Hypothesis

Verbatim from item 237:

> Cycle 218 showed centred Δ̃_l's top-20 right SVs align with block l+1's c_fc reading subspace at +17 to +33σ above random in 11/11 cross-block pairs, but never tested non-adjacent destinations. Audit (G3) flagged this missing control: if Δ̃_l also aligns with c_fc(j) for j ≠ l+1, the alignment is generic ("Δ̃ leaks into a universal reading basis") rather than specifically inter-block communication. For each l ∈ {0..10}, compute the top-20 right SVs of centred Δ̃_l and measure top-5 principal cosine sums against c_fc(j) for ALL j > l (not just j=l+1). Plot a heatmap (l, j) and compare each row's max-at-j=l+1 against the row's mean over j>l+1. Use the same random baseline (10 seeds Gaussian orthonormal 20×768).

# Pre-flight Results

- P1 duplicate: cycle 218 measured ONLY j=l+1. NEW.
- P2 dependency: 232 / cycle 218 (done). OK.
- P3 ≤30 min: yes — same forward pass as cycle 218 + extended computation. ~1 minute compute.
- P4 program/prompts touched: no.

# Steps

1. Create `experiments/delta_specificity_cycle227.py` based on `experiments/delta_centred_alignment_cycle218.py`. Reuse N_CALIB=200, SEQ_LEN=128, SEED=42, N_RANDOM_SEEDS=10, DEVICE selection, model/tokenizer/data loader, c_fc forward hook, and centred Δ̃_l computation (chunked W_fc/b_fc/W_proj path through GELU−ReLU). Drop block_inputs prehook, c_attn Q/K/V readers, residual PC reader, vocab interpretability — only c_fc(j) reader is needed.
2. For each l ∈ {0..10}:
   - Compute centred Δ̃_l → SVD → cache `Vh_top20[l]` shape (20, 768).
3. For each j ∈ {1..11}:
   - U_j = top-20 left SVs of c_fc(j).weight (shape (768, 20)) via the same `top_left_svs(W, 20)` helper.
4. For each (l, j) with l < j:
   - principal_cos = `torch.linalg.svdvals(Vh_top20[l] @ U_j)`  (length 20)
   - cos_sum_top5 = principal_cos[:5].sum().item()
   - cos_sum_top20 = principal_cos.sum().item()
   - Store both into a dict heatmap[(l, j)] keyed as f"{l}_{j}".
5. Random baseline: 10 seeds × 20-dim Gaussian QR-orthonormal in R^768 (reuse cycle 218's `random_orthonormal(seed)` returning (20, 768)). For each j ∈ {1..11}, for each random V_r, compute `cos = svdvals(V_r @ U_j)` and accumulate cos_sum_top5 and cos_sum_top20. Aggregate per-j mean ± std for both metrics.
6. Per-row analysis (l fixed, j > l):
   - row_max_at_lp1 = cos_sum_top5[(l, l+1)]
   - row_mean_nonadj = mean over j ∈ {l+2..11} of cos_sum_top5[(l, j)]  (skip j=l+1; if l=10 row has no non-adjacent entries → mark as N/A)
   - row_specificity = row_max_at_lp1 − row_mean_nonadj
7. Verdict computation:
   - For each (l, j) with j != l+1, compute `is_above_3sigma = cos_sum_top5[(l,j)] > random_mean_top5[j] + 3*random_std_top5[j]`.
   - n_significant_nonadj = count of (l, j) entries with j != l+1 above 3σ.
   - n_rows_with_quiet_nonadj = number of rows (l ∈ {0..9}; l=10 excluded since no non-adjacent j) where ALL j != l+1 entries lie within mean+3σ of random.
   - median_specificity = median over rows (l ∈ {0..9}) of row_specificity.
   - Verdict:
     - CONFIRMED if median_specificity ≥ 1.0 AND n_rows_with_quiet_nonadj ≥ 7.
     - REFUTED if median_specificity < 0.3.
     - INCONCLUSIVE if 0.3 ≤ median_specificity < 1.0.
8. Save JSON `autoresearch/results/cycle_227_delta_specificity.json` with: cycle/item, config, full (l, j) heatmap (cos_sum_top5 + cos_sum_top20), per-j random baseline (mean/std for top5 + top20), per-row {row_max_at_lp1, row_mean_nonadj, row_specificity, all-quiet-nonadj flag}, n_significant_nonadj, n_rows_with_quiet_nonadj, median_specificity, verdict.
9. Save markdown `autoresearch/results/cycle_227_summary.md`: heatmap as a text table (rows l=0..10, columns j=1..11, blank where j ≤ l), 11 row specificity values, random baseline table, verdict, plain-language summary tying back to cycle 218's CONFIRMED 11/11 adjacent result.

# Validation

- Reproduce cycle 218's adjacent (l, l+1) cos_sum_top20 values (4.42–7.07 range) — sanity check that the centred Δ̃ pipeline matches.
- Random baseline cos_sum_top5 should be O(1.5–2.0) and cos_sum_top20 should match cycle 218's c_fc baseline (~2.72±0.10) — confirm the same QR orthonormal construction.
- Print one heatmap row each for l=0 and l=10 during execution as a quick eyeball check.

# Artifacts

- `experiments/delta_specificity_cycle227.py`
- `autoresearch/results/cycle_227_delta_specificity.json`
- `autoresearch/results/cycle_227_summary.md`
- `autoresearch/results/cycle_227_manifest.txt` (written by execute agent)

# Fallbacks

- If random baseline computation is slow: 5 seeds instead of 10 (still O(50) random samples per j).
- If verdict marginal (median_specificity in [0.8, 1.2] OR n_rows_with_quiet_nonadj in {6, 7, 8}): report exact numbers and recommend manual interpretation in the markdown summary; pick the strict-criterion verdict.
- If cycle 218 sanity check fails (>5% deviation): stop and flag the data-loader/hook mismatch before reporting.

# Update bookkeeping

- Move item 237 from NEXT/P0 to PAST/Cycle 227 in `autoresearch/agenda.md`.
- Append TSV row to `autoresearch/cycles.tsv`.
- Commit with `[autoresearch] cycle 227: ...` message.
- Update `autoresearch/STATUS.md`.
