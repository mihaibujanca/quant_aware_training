# Cycle 218 audit — Item 232

**Item**: "Where does Δ point? — centred Δ̃ alignment with next block's reading directions"
**Claimed outcome**: CONFIRMED (11/11 pairs significant; per-reader c_fc 11, Q 10, K 6, V 8, resPC 11).
**Audit status**: PASS

## Phase 1 — literal criterion check

The item's pre-registered criterion is "top-20 Δ̃ principal cosines exceed random Gaussian baseline with p < 0.01 in ≥ 8/11 cross-block pairs (against ≥ 1 downstream reader)." Cycle 218 used random mean + 3σ (~p < 0.01 two-sided) over a 10-seed QR-orthogonalised Gaussian baseline.

Observed vs baseline (random mean_top20 ≈ 2.72, std ≈ 0.10):

| l→l+1 | c_fc cos_sum | c_fc σ-excess | resPC cos_sum | resPC σ-excess |
|------:|-------------:|--------------:|--------------:|---------------:|
| 0→1  | 4.847 | +21.8σ | 6.913 | +42.0σ |
| 1→2  | 4.423 | +17.5σ | 6.965 | +42.5σ |
| 2→3  | 4.864 | +22.0σ | 5.110 | +23.9σ |
| 3→4  | 5.229 | +25.7σ | 5.143 | +24.3σ |
| 4→5  | 6.896 | +42.7σ | 4.769 | +20.5σ |
| 5→6  | 6.575 | +39.4σ | 5.013 | +23.0σ |
| 6→7  | 6.709 | +40.8σ | 5.482 | +27.7σ |
| 7→8  | 7.068 | +44.5σ | 6.024 | +33.1σ |
| 8→9  | 6.392 | +37.6σ | 6.351 | +36.4σ |
| 9→10 | 5.183 | +25.2σ | 6.763 | +40.5σ |
| 10→11| 4.562 | +18.9σ | 8.382 | +56.7σ |

- c_fc: min 4.42 / mean 5.70 / max 7.07, **always ≥ 17σ** above random, median +25.7σ.
- resPC: min 4.77 / mean 6.08 / max 8.38, **always ≥ 20σ**, median +33.1σ.
- Q is 10/11 significant at +2.4σ to +14.9σ; K is 6/11 (4 blocks dip near or below random: 4→5 -1.2σ, 6→7 -0.9σ, 9→10 -3.5σ, 10→11 -3.3σ); V is 8/11 (deep blocks 8→9, 9→10, 10→11 hit -10 to -13σ — actively anti-aligned).

By the decision rule (c_fc ≥ mean+10σ on most pairs → PASS), this result dominates. c_fc is ≥ 17σ above random on every single pair, not 3-5σ. resPC is similarly strong. Even if one argues "Δ̃ is full-rank so everything overlaps," the overlap against c_fc and resPC is ~6-8× the classical Gaussian-subspace expected value (~3.23) and ~15× the random std — that is not trivial cosine budget leakage.

## Phase 2 — quality checks

**G1 Implementation (verified against `experiments/delta_centred_alignment_cycle218.py`):**
- Centring: `mean_delta = delta_out.mean(0); delta_centred = delta_out - mean_delta` (L146-147). Correct.
- Δ̃ right-SV extraction: `Vh[:20]` from `torch.linalg.svd(delta_centred, full_matrices=False)` (L150, 167). Correct — Vh rows are right singular vectors.
- c_fc reader: `torch.linalg.svd(W_fc, full_matrices=False); U[:, :20]` with W_fc shape (768, 3072) asserted. Correct — top-20 left SVs in the 768-dim row space.
- c_attn Q/K/V slicing: `W_attn[:, 0:768] / [768:1536] / [1536:2304]` with assert shape (768, 2304) (L232-236). Correct for Conv1D layout.
- Residual PC: `forward_pre_hook` on `model.transformer.h[l]` captures `args[0]` which is the block input PRE-LN1 (L73-75, 244-247). Correct.
- Principal cosines: `svdvals(V20 @ B)` (L309). Correct.
- Random baseline: QR-orthogonalised Gaussian (768, 20), 10 seeds (L262-268). Reasonable.

**G2 Reproducibility**: 10 random seeds for baseline (vs plan's 3 — stricter, good). Single deterministic forward pass for Δ; seed 42.

**G3 Interpretation**: Here is the one place the audit should push back — but softly. The criterion *is* met with huge margin. However, cycle 218 did NOT run the "Δ̃_l vs WRONG-block c_fc(j ≠ l+1)" specificity control. The result shows "Δ̃_l writes into directions that block l+1 happens to read," but does not prove the alignment is specific to the adjacent block vs any other. Given that cycle 216 already measured W_down→c_fc cross-block baseline 0.649 between adjacent blocks but did not test non-adjacent blocks either, specificity to l+1 (as opposed to l+5 or l+k for any k) is open.

That said, the centred-Δ̃ full-rank concern is reasonable but here clearly NOT the driver:
- rank_95 centred = 489-567 (median 552). The top-20 right SVs cover a specific 20-dim subspace chosen by Δ̃'s energy, not an arbitrary 20-dim slice.
- If Δ̃ were isotropic in its 552-dim active subspace, the expected cos_sum against a fixed 20-dim reader would be ~20·sqrt(20/552) ≈ 3.80, barely higher than the 20/768 classical bound (3.23). The observed c_fc means 4.42-7.07 and resPC 4.77-8.38 are well above BOTH the random 20-dim-in-768 baseline AND the would-be "isotropic-in-552" upper envelope. So the top-20 of Δ̃ is genuinely picking out reader-aligned directions, not uniformly covering its own active subspace.
- V-reader anti-alignment in deep blocks (8-10 → l+1) at -10 to -13σ is positive evidence of non-trivial structure: a "Δ̃ is everywhere" null would produce +0σ to +3σ everywhere, not large negative excursions. Some readers are AVOIDED.

**G5 learned/next**: `learned=` (manifest L6) correctly states centring-revealed structure and cross-block reader alignment. `next=` (L7) correctly points to item 234 (surgical bump removal) for functional localisation and SV1+ vocab projection. Both specific and grounded.

**G6 Method failure vs genuine confirmation**:
- Literal criterion: PASS with ~15σ median headroom on c_fc, 33σ on resPC.
- Scientific claim "bump writes structured inter-block messages": SUPPORTED by the large σ-excess on c_fc and resPC, and the V-reader anti-alignment in deep blocks (structure, not coverage). NOT fully proven in the "specific to block l+1" sense — the missing control is cross-block shuffle / non-adjacent-block alignment.

## Decision

PASS. The effect sizes (c_fc always ≥ 17σ, median +25.7σ; resPC always ≥ 20σ, median +33.1σ) far exceed the "PASS ≥ 10σ on most pairs" rule. The full-rank-coverage concern would manifest as +3-5σ uniformly; instead we see +17-56σ on primary readers and negative excursions on secondary readers, which is the opposite of uniform coverage. The criterion is met with strong margin and the overall scientific interpretation ("Δ̃ writes into downstream reading directions") is supported — with the important caveat that **specificity to the immediately-adjacent block vs arbitrary downstream blocks was not tested** and remains an open control for a future cycle.

## Recommendations (non-blocking, do not require correction)

1. **Specificity control**: recompute the full alignment table with shuffled block indices (Δ̃_l vs c_fc(j) for j ≠ l+1, j randomly drawn). If Δ̃_l aligns equally with non-adjacent c_fc(j), the "structured inter-block communication" claim degrades to "Δ̃ aligns with generic c_fc-shaped subspaces," which would be a weaker statement. This is a 1-hour follow-up, not a blocker.
2. **Alignment ratio**: report `cos_sum(Δ̃_l, reader_{l+1}) / median_j cos_sum(Δ̃_l, reader_j)` as a scalar per pair — if > 1.3, local specificity is real; if ~1.0, it is not.
3. **V-reader negative excursions**: the deep-block V anti-alignment (-10 to -13σ) is not mentioned in the summary.md but is an interesting secondary finding. Worth a one-line note in the next agenda entry.

## Flags

- guardrail_flags: specificity_control_missing
- interesting: true (large effect sizes, V-reader anti-alignment in deep blocks, centring-reveals-rank is clean)
