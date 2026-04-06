# Cycle 217 audit — Item 231 (Per-token Δ = GELU_FFN − ReLU_FFN)

**Status: CORRECTED**
**Old outcome: CONFIRMED**
**New outcome: INCONCLUSIVE**

## Phase 1 — Verdict check

Plan criteria (verbatim):
- **Confirmed** if `rank_95(Δ) ≤ 64 in ≥ 6 of 12 blocks` AND `top singular vectors produce interpretable token shifts (visible clusters of related vocabulary items)`.
- **Refuted** if `rank_95(Δ) > 400 in all blocks`.
- **Inconclusive** if `low rank but no semantic structure`, or vice versa.

Raw numbers from `cycle_217_delta_per_token.json`:
- rank_95 per block: `[10, 132, 1, 1, 1, 1, 1, 8, 73, 110, 1, 1]`.
- Blocks with rank_95 ≤ 64: **9 / 12** (blocks 0, 2, 3, 4, 5, 6, 7, 10, 11). Rank clause satisfied.
- TI/TD ratio per block: `0.87-0.99` (min 0.8680 block 1, max 0.9883 block 11).
- Top singular value σ₀ per striking block: 2625 – 7207. σ₁ ≈ 200 – 350. Ratio σ₁/σ₀ ≈ 0.03 – 0.09.

Semantic-interpretability clause — what SV0 actually projects to (from summary.md striking blocks):
- Block 2 SV0 boost: `guiActiveUn, ���, cloneembedreportprint, \x06, \x1a, ...`, suppress: `, the, and, ., -`.
- Block 3 SV0 boost: `���, guiActiveUn, inventoryQuantity, quickShipAvailable, cloneembedreportprint, ...`, suppress: `, the, and, -, a, .`.
- Block 4 SV0 boost: `, the, and, ., in, a, -, ...`, suppress: tokenizer-artefact tokens.
- Block 5, 6, 7, 9, 10 SV0 boost: similar `, the, and, .` stopword set.
- Block 11 SV0 boost: `DragonMagazine, natureconservancy, cloneembedreportprint` (sign-flipped stopword axis via near-degenerate spectrum: rank_95 = rank_99 = 1).

These are NOT "clusters of related vocabulary items" in the scientific sense the agenda intended. The rank_95 = 1 in 9/12 blocks is a direct consequence of TI/TD = 0.87-0.99: nearly all the per-token Δ energy is concentrated in a single *token-independent mean direction*. SV0 just is ≈ `mean(Δ_l) / ||mean(Δ_l)||` in those blocks, and projecting the mean direction through `lm_head` trivially recovers the unigram frequency axis (common tokens boosted, noise tokens suppressed — or the sign-flipped mirror of that, blocks 2/3/11). That is not a "structured token-dependent message in K specific directions"; it is a per-block constant bias.

The actually-per-token component lives in SV1+ (σ₁ ≈ 200-350 = 3-9% of σ₀). The SV1/SV2 token lists for most blocks are noisy mixes of word-fragments and artefact tokens with no clean semantic cluster. Block 9 SV1 is the only striking exception: a sentence-ending-punctuation cluster (`.」 !. .</ .� .} !, .'' .'' ; ,—`). That is one interpretable cluster across 12 blocks × 2 non-trivial SVs, not 6.

Cross-block alignment top-3 mean 0.369 vs random 0.168 is above noise, but below cycle 216's raw W_down→c_fc baseline of 0.649 — so Δ does not align with next-block readers *more* than the raw weight structure already does. No additional structural signal beyond the 216 finding.

### Verdict under both readings

- **Literal (executor's)**: `rank_95 ≤ 64` in ≥ 6 blocks ✓, and `count_word_tokens(top10) ≥ 3` heuristic fires in 12/12 blocks → CONFIRMED. But the interpretability heuristic (ASCII-alphanumeric run of ≥2 chars) treats `the, and, in, ...` as "word tokens", which satisfies the letter of the check without satisfying its intent. The heuristic is the wrong test for "visible clusters of *related* vocabulary items".
- **Spirit (agenda)**: the claim being tested was "Δ is a structured low-rank *token-dependent* message". The data show Δ is a per-block **constant bias** (TI/TD ≥ 0.87). Rank clause is met only because the mean direction dominates the spectrum. Semantic-cluster clause is not met. By the agenda's own "low rank but no semantic structure" tiebreaker, this is **INCONCLUSIVE**.

The executor flagged the caveat clearly in plain-language summary.md ("Important caveat... the rank_95 = 1 mostly reflects that one average vector... Verdict CONFIRMED by the plan's criteria, but the interpretation is 'Δ is mostly a constant bias per block plus a small token-dependent residual', not 'Δ is a rich token-dependent low-rank message'."). The audit upgrades that caveat to a verdict correction.

## Phase 2 — Quality checks

- **G1 Code correctness**: verified.
  - Hook `model.transformer.h[l].mlp.c_fc.register_forward_hook(fn)` with `fn` reading `input[0]` → captures post-LN2 residual stream (the c_fc input). Correct target.
  - `F.gelu(preact, approximate='tanh')` **exactly matches** HF GPT-2's `gelu_new`. Verified numerically: `max |diff| = 0.0` on random preact, vs the explicit `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))` formula.
  - Pure ReLU `F.relu(preact)` (no bias correction) — correct per the hypothesis.
  - `delta_out = (gelu_post - relu_post) @ W_proj` with Conv1D weight of shape `(3072, 768)`: c_proj bias cancels cleanly because `(gelu + b) - (relu + b) = gelu - relu`. Correct.
  - SVD via `torch.linalg.svd(delta_out, full_matrices=False)` with `delta_out.shape = (25600, 768)` → `Vh.shape = (768, 768)`. `Vh[k]` is the k-th right singular vector in R^768. Correct.
  - `ti_td_ratio = ||mean(Δ)||² / E[||Δ_i||²]` — this is the standard energy fraction in the mean direction, bounded in [0, 1]. Correct.
  - Cross-block alignment: `svdvals(Vh_top3 @ U_next20)` where `U_next20` is the top-20 left singular vectors of `h[l+1].mlp.c_fc.weight` (shape `(768, 3072)`). This is the principal-cosine between the 3-dim Δ-writing subspace and the 20-dim c_fc-reading subspace. Correct.
- **G2 Reproducibility**: single forward pass, `torch.manual_seed(42)`, deterministic SVD. Fine.
- **G3 Interpretation — FLAGGED**: the automatic verdict uses a letter-of-the-law interpretability check (`count_word_tokens ≥ 3` over top-10 lists) that doesn't test what the hypothesis actually claims. Executor's prose caveat correctly identifies the issue but keeps the label CONFIRMED.
- **G4 Counterintuitive check**: "rank_95 = 1 in 9/12 blocks" is surprising until you see TI/TD ≥ 0.87 — then it is the trivial consequence. rank_99 per block `[322, 492, 173, 100, 107, 173, 278, 353, 430, 461, 323, 1]` shows the residual-variation rank is O(100-500), not 1: confirms the low rank_95 is mean-dominance, not actual 1-dimensional structure. Not a bug — but also not the finding the hypothesis was seeking.
- **G5 Learning quality — FLAGGED**: manifest `learned` field is thorough and does include the TI/TD caveat, but `outcome=confirmed` overrides the caveat in downstream bookkeeping. The NEXT-step recommendation ("SVD the centred Δ_l − mean(Δ_l)") is correct and should be retained, but it should be the *central* finding, not a footnote.
- **G6 Method failure vs genuine**: GENUINE measurement, no methodology bug. The correction is purely interpretive.

## Guardrails

Tripped: **G3** (interpretation mismatch between automated verdict and hypothesis intent), **G5** (manifest outcome label doesn't reflect the actual informative finding).

## Interesting

Yes. The informative result — largely buried by the CONFIRMED label — is:

1. **Δ(x) is dominated by a per-block token-independent constant direction**. TI/TD ratio 0.87-0.99 across all 12 blocks. σ₀ is 10-30× larger than σ₁ in every striking block.
2. The mean-direction axis projects through `lm_head` to the **unigram-frequency axis** (common tokens boosted, artefact tokens suppressed, or its sign-flip). This is the most parsimonious explanation for every rank_95 = 1 block's SV0.
3. This **connects to cycle 188 (C3 per-channel mean replacement: +20.33% PPL)** and **cycle 191 (constant residual correction: +12 PPL)** — both tried to absorb the constant bias directly and failed. The current cycle identifies *why* those experiments failed: the constant bias IS the dominant Δ component (that is why it is absorbable in isolation), but GPT-2's loss depends on the small token-dependent residual in SV1+ that you destroy when you replace Δ with its mean. The constant is not "the important part" — it is "the easy part to measure", and the important part is the 3-10% per-token residual.
4. The agenda's original hypothesis ("Δ is a rich token-dependent low-rank routing message") is **effectively refuted at the SV0 level** and **untested at the SV1+ level**. Item 232 should test the centred Δ̃ = Δ − mean(Δ) per block: `rank_95(Δ̃)` will reveal the true per-token rank, and cross-block alignment of Δ̃ with next-block readers is the hypothesis that survives.

## Recommendations

1. **Item 232** (already dependent on 231): run SVD on centred Δ̃_l = Δ_l − mean(Δ_l) per block. Report rank_95(Δ̃), rank_99(Δ̃), σ₁(Δ̃) vs σ₀(Δ) baseline, and top-k SV projection to vocab. This is the experiment that actually tests the hypothesis.
2. **Extend item 232** to include alignment of Δ̃ with next-block W_Q/W_K/W_V left singular vectors (not just c_fc). The agenda already lists this; reinforce it.
3. **Do not revisit constant-bias absorption** in future items. Cycles 188, 191, and now 217 SV0 are three independent measurements of the same thing. The constant component is absorbable in isolation and not usable without the per-token residual — this is established.
4. Leave items 233, 234 as-is; they don't depend on the corrected interpretation of 231.
