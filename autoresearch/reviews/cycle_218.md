# Cycle 218 Design Review — Item 232 (P0)

**Verdict: APPROVE**

## R1 — Hypothesis well-posed

The verbatim criterion from the agenda ("top-20 Δ directions vs top-100 W_up rows, p<0.01 in ≥8/11 pairs") is lifted cleanly. The audit-directed extension to {c_fc, Q, K, V, residual PCs} is implemented as an OR over 5 downstream reader subspaces, with a conservative mean+3σ significance threshold (≈p<0.003). Refuted/inconclusive tiers spelled out. Centring is the correct response to cycle 217's TI/TD=0.87-0.99 finding.

## R2 — Measurement validity

- **Centring Δ̃ = Δ − mean(Δ) per block**: correct. Removes the token-independent constant axis that dominated SV0 in 217.
- **Top-20 right SVs of Δ̃ ∈ R^768**: correct. Δ̃ has shape (n_tokens, 768) so `Vh[:20]` lives in R^768, the right space for principal-angle comparison with reading subspaces.
- **HF GPT-2 Conv1D layout**: c_attn.weight is (768, 2304), slicing `[:, :768]` gives Q's 768 columns. Top-20 LEFT SVs of Q ∈ R^(768, 768) are the top variance *reading* directions in R^768. Correct. Sanity 4 explicitly asserts shape.
- **c_fc left SVs**: W_fc is (768, 3072), `U_fc[:, :20]` gives the top-20 reading directions into the MLP's 3072 hidden space. Correct.
- **Residual PC capture**: Plan uses `h[l+1].register_forward_pre_hook`, captures block pre-LN1 input. Correct (MP7-clean: the residual stream is pre-LayerNorm). Fallback to `h[l+1].ln_1` pre-hook is equivalent. Sanity 3 asserts this.
- **Principal cosines via `svdvals(V_top20 @ U_reader)`**: identical to the cycle 217 method; standard principal-angle computation. Correct.
- **Random baseline**: QR-orthogonalized Gaussian → orthonormal basis in R^768, 3 seeds. Matches the "replace top-20 of Δ̃ with random orthonormal basis" semantics of the principal-cosine comparison. Correct.

## R3 — Signal-to-noise of the verdict

With 3 random seeds and mean+3σ, the expected per-(block, reader) false-positive rate under normal approximation is ≈0.003. With 5 reader types per pair and an OR criterion, per-pair FP is ≈5·0.003 = 0.015 (conservative — readers are partially correlated through c_attn, which can inflate but not deflate). Across 11 pairs the expected FP count is ≈0.16, far below the 8/11 threshold. The OR over 5 is **not** too lax; 8/11 is a strong bar.

Potential concern: 3 seeds for random baseline is minimal. The plan's fallback (increase to 10 seeds if std < 1e-3) is adequate. Note that with only 3 seeds, the sample std is a weak estimator of the null std, so mean+3σ is approximate. Acceptable for this triage-level test.

## R4 — Guardrails / pitfalls

- **MP6 (verdict misclassification)**: Addressed. The verdict criteria are aligned with the actual hypothesis (centred Δ per-token alignment), not the mean-dominance artifact that tripped 217.
- **MP7 (post-LN)**: Residual capture is pre-LN1 via forward_pre_hook on the block (or ln_1) — correct side of the LN.
- **MP8 (rep dim vs functional rank)**: SVDs are of the function's weight matrices (c_fc, c_attn) and of the centred Δ output, not of the residual-stream representation cloud. The residual-stream PCs are explicitly a reader basis (comparison target), not a rank claim. Fine.
- **Cycle 217 mean-dominance pitfall**: centring removes it.
- **Counterintuitive-results rule**: if centred Δ̃ rank_95 is still ≤10, that would be surprising and should trigger a second-look before a CONFIRMED label. Sanity 1 anticipates this by comparing against the raw 217 numbers. Good.
- **Vocab interpretability heuristic**: still loose (stopword exclusion list + ≥3 word tokens). The plan explicitly calls it a heuristic/triage; if it fires, the executor should print tokens so the audit can inspect. Acceptable as-is.

## R5 — Reproducibility and scope

- Single forward pass, deterministic seed, ≤30 min. OK.
- Artifacts: experiment script, JSON with per-block/per-pair/per-reader metrics, summary.md, manifest. Adequate.
- Bookkeeping: plan correctly notes to DELETE item 232 from NEXT (runner bug) and adds it to PAST.
- Dependency on 217: data structure reused (not the INCONCLUSIVE verdict), which is legitimate.

## Specific concerns addressed

1. **Is the OR criterion across 5 readers too lax?** No. Even assuming full independence, per-pair FP ≈0.015; expected 11-pair FP count ≈0.16 << 8. Q/K/V correlation inflates this modestly but nowhere near enough to reach 8/11 by chance.
2. **Residual capture**: plan uses `h[l+1].register_forward_pre_hook`, which captures the block input (pre-LN1). Fallback to `h[l+1].ln_1.register_forward_pre_hook` is equivalent. Both pre-LN.
3. **Conv1D slicing**: plan is correct and guarded by Sanity 4.
4. **Vocab heuristic**: improved over cycle 217 but still loose. Accept with the understanding it's triage; interpretability verdict belongs to the audit.

## Issues

None blocking.

## Recommendations (non-blocking)

- Consider bumping the baseline to 10 seeds by default (not only as a fallback). Cost is trivial (10 × QR of 768×20), and it gives a more stable null std for mean+3σ.
- In the JSON, record `cos_sum` per block per reader alongside per-seed random draws (not just mean/std), so the audit can recompute alternative thresholds if needed.
- In the summary.md, explicitly contrast avg rank_95(Δ̃) against cycle 217's `[10, 132, 1, 1, 1, 1, 1, 8, 73, 110, 1, 1]` — this is the headline sanity check and should be front-and-centre.

These are nice-to-haves, not blockers.
