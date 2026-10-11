# Cycle 237 audit — Item 247 (T3: Bump subspace ablation vs random baseline)

**Status: PASS**
**Old outcome: CONFIRMED**
**New outcome: CONFIRMED**

## Phase 1 — Verdict check

Plan criteria (verbatim from agenda item 247):
- **Confirmed** if joint bump-subspace ablation PPL ≥ 1.5× random-subspace ablation mean PPL, AND at least 3 individual blocks show single-block ablation PPL ≥ 2× their random baseline.
- **Refuted** if bump ablation PPL within 1.1× random baseline.
- **Inconclusive** otherwise.

Headline numbers from `cycle_237_t3_ablation_ppl.json` / `cycle_237_t3_summary.md`:
- `baseline_ppl = 34.514` (wikitext-2 val 200×512 held-out).
- `joint_bump_ppl = 2,344,385.825` vs `joint_random_mean = 101.298 ± 8.137` (3 seeds) → ratio **23,143.38×** (gate ≥1.5).
- Per-block bump/random ratios: `[19.9, 12.9, 10.6, 30.2, 22.1, 25.2, 797.7, 11.2, 1.65, 1.45, 1.51, 928.4]` → **9/12** blocks ≥ 2× (gate ≥3).

CONFIRMED gate: 23,143 ≫ 1.5 ✓ AND 9/12 ≫ 3 ✓. Verdict matches criterion literally. ✓

## Phase 2 — Critical interpretation

23,143× and a PPL of 2.3M look alarming and warrant G4 scrutiny, but the ratio is what the gate measures and it is robust:

1. **Random 10-dim ablation → PPL 101.3** (~2.94× baseline) is a reasonable magnitude for removing 10 dims/block from a 768-dim residual stream across 12 blocks. The joint cascade (10×12 = 120 dim-blocks of removal) of *any* consistent subspace will amplify faster than a single-layer cost.
2. **Bump 10-dim ablation → PPL 2.34M** is 23,000× worse than the same-cardinality random ablation on the same scaffold. The absolute magnitude is large because the bump subspace is the *specific* 10-dim direction where the centred Δ̃ per-token signal lives at every block; ablating it severs the same 10-dim communication channel 12 times in cascade. The ratio, not the magnitude, is the load-bearing number.
3. **Per-block structure is informative, not uniform**: blocks 0–7 show single-block ratios 10–30× random (significantly load-bearing), blocks 6 and 11 show **catastrophic** 797× and 928× single-block ratios, while blocks 8/9/10 are essentially random-like (1.45–1.65×). This is the compression-candidate fingerprint Items 218/235 predicted.
4. **Full-FFN top-10 comparison is a clean control**: joint full-FFN 3.11M (same order as bump 2.34M), per-block catastrophes at blocks **3 and 11** vs bump catastrophes at **6 and 11**. Both subspaces are critical but at *different* blocks, confirming the bump subspace is structurally distinct from raw FFN top PCs. Block 11 is a shared outlier (consistent with cycle 236 T2 depth monotonicity and the cycle 235 audit prediction).
5. **Reconciliation with cycle 236 T2 (INCONCLUSIVE)**: T2 killed the specific `wte @ v` decoder but left the broader "bump writes into *some* downstream subspace" hypothesis alive. T3 confirms the subspace is causally critical for held-out LM performance. T2 was the wrong DECODER test; T3 is the right SUBSPACE test. Both outcomes are coherent.

## Phase 3 — Quality checks

- **G1 Code correctness — clean**: `experiments/bump_t3_ablation_cycle237.py` loads cycle 235 features `V_full = features[f"block_{l}"]["V"]` shape (768, 32), takes first 10 cols to build `V_l ∈ R^{768×10}` and `P_bump_l = V_l @ V_l.T ∈ R^{768×768}`. Hook ablates residual: `new_hs = hs - hs @ P` on block forward output tuple. Random baseline draws `torch.randn(768, 10)` then QR → orthonormal (same construction recipe), 3 seeds `[42, 137, 2024]`. Full-FFN baseline captures `transformer.h[l].mlp` outputs on a fresh 200×512 calibration split of wikitext-2 **train** (disjoint from the val eval set), runs a proper SVD, takes top-10 right singular vectors. Eval is on wikitext-2 **validation**, 200×512 = 102,400 tokens. `eval_ppl` uses standard shift-logits cross-entropy, sum-reduction, divides by total shifted tokens, exponentiates — correct.
- **G2 Reproducibility**: 3 random seeds, deterministic eval, sanity checks reported: `V_0` orthonormal (max `|V_0.T V_0 − I| = 1.07e-6`), `P_0` idempotent (max `|P_0 P_0 − P_0| = 5.07e-7`), baseline PPL identical to 6 decimal places before and after all hooks removed (`|Δ| = 0.00e+00`). No silent fallbacks.
- **G3 Interpretation — clean**: executor framing ("bump directions are the FFN's cascade-critical channels; not compression slack") is correct and explicitly calls out blocks 8–10 as the compression candidates and blocks 6/11 as do-not-compress. Matches the evidence verbatim.
- **G4 Counterintuitive magnitude check — passes**: the 2.3M PPL absolute is large but the comparator (random 10-dim ablation on the same scaffold) sits at 101 PPL, so the 23,143× ratio is measured against a fair same-cardinality control. Both the bump and the full-FFN subspaces are in the millions, which is the expected regime when 12 consistent 10-dim ablations cascade. The ratio bump/full-FFN ≈ 0.75 is telling — the two destroy the cascade at similar magnitudes but at *different* blocks, a sign of genuinely distinct subspace structure.
- **G5 Learning quality**: manifest `learned` field is exhaustive, records the 9/12 gate pass, the 3 random seeds with mean ± std, the full per-block ratio list, the divergent full-FFN block pattern (3, 11 vs bump 6, 11), all sanity checks, the cycle 218 "Δ̃ aligns with c_fc" connection, and explicit block-level compression guidance.
- **G6 Method failure vs genuine — NOT a method failure**: this is a direct causal PPL ablation test on held-out data with a fair random-subspace control and a second orthogonal control (full-FFN top-10). No training, no leakage, no in-distribution overfitting surface. The 23,143× ratio is the real signal.

## Guardrails

Tripped: **none**.

## Interesting

Yes — this is the first strong causal confirmation that the centred Δ̃ top-10 subspace is load-bearing for GPT-2's held-out LM prediction via aggregate PPL ablation, *dramatically* more than any random 10-dim subspace. Block 11 is the shared peak for both bump and full-FFN subspaces (consistent with cycle 236 depth monotonicity). The divergent per-block catastrophe pattern (bump: 6, 11; full-FFN: 3, 11) shows that the bump subspace is structurally distinct from raw FFN output PCs despite comparable joint damage — the two subspaces route through different critical layers. Cycle 236's "T1 wte probe is wrong" is reconciled here: T2 killed the decoder, T3 confirms the subspace.

## Recommendations

1. **Do NOT compress blocks 6 or 11** (single-block catastrophic ratios 797× and 928× — one block alone kills the model).
2. **Blocks 8/9/10 are the realistic k=10 compression candidates** (ratios 1.45–1.65, essentially random-like) — compressing the bump subspace at these middle-late blocks should have minimal PPL cost.
3. **T4 (item 248)** A/B activation patching is the next gating test and will convert this subspace-level confirmation into per-token attribution.
4. Consider a downstream-reader probe: project `V_l` onto later-block `V_{l+k}` bases (or reader subspaces `W_{l+k,fc}`) to identify which block reads the bump subspace, replacing the dead wte@v probe with a logit-lens-style propagation.
5. Investigate why the bump (6, 11) and full-FFN (3, 11) catastrophic block sets differ — this is the direct fingerprint of what makes the bump subspace structurally distinct.
