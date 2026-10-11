# GELU Bump Investigation: Deep Audit

**Date**: 2026-04-01
**Scope**: Audit of `docs/gelu_bump_investigation_brief.md` against all underlying experiment data (cycles 102-193)
**Status**: Analysis complete. Tie-breaker experiments run (cycles 188-193). Contradictions resolved.

---

## 1. Verdict on the brief

The raw numbers are accurate. Every claim was verified against the result JSONs:

- Cycle 173 C3: 12/12 blocks within +2% — confirmed. Max is block 4 at +1.4%.
- Cycle 181 all-12: PPL 42.6 (+88%) — confirmed.
- Cycle 182 iterative: PPL 37.0 (+63%) — confirmed.
- Cycle 109 b_energy: 87-98% — confirmed.
- Cycle 102 full swap: PPL 1874 (+8320%) — confirmed.
- Cycle 165 learned negativity: z_mean shift ~1.0 from random init — confirmed.

The facts are correct. The framing is wrong.

---

## 2. The central confusion: two operations called the same thing

The brief frames a contradiction: "per-block bias replacement works (+2%), but composition fails (+63-88%)." This conflates two fundamentally different operations.

### Cycle 173 C3 — what actually happens

```
For each token t, for each channel c where z_tc < 0:
    output_c = mu_neg_c    (per-channel conditional mean, 3072 values)
For channels where z_tc >= 0:
    output_c = GELU(z_tc)  (preserved exactly)
```

- **3072-dimensional** per-channel replacement
- **Token-dependent** mask (which channels get replaced varies per token)
- **Preserves GELU shape** for positive pre-activations
- Cost: +0.4% to +1.4% per block

### Cycle 181 — what actually happens

```
For ALL tokens, ALL channels:
    activation = ReLU(z)   (shape change for ALL channels, including 0 < z < 2)
    bias += W_down @ E[GELU(z) * 1[z<0]]   (768-dim constant, same for all tokens)
```

- **768-dimensional** projected bias (3072 → 768 through W_down)
- **Token-independent** (same correction regardless of input)
- **Destroys GELU shape** for positive pre-activations (ReLU ≠ GELU near z=0)
- Cost: +0.8% to +5.9% per block

These differ in three ways: granularity (3072 vs 768), data-dependence (token-varying mask vs constant), and positive-regime shape (preserved vs destroyed). The brief treats the +2% C3 result as evidence that "the bias works," then tests something quite different in cycle 181 and calls the failure a "composition mystery."

---

## 3. Three error sources, decomposed

Using data from cycles 173 and 181, we can isolate three distinct contributions per block:

| Block | E_shape (C1 vs C0) | E_replacement (C3 vs C0) | E_projection (181 vs C3) | Shape/Total |
|------:|-------------------:|------------------------:|------------------------:|------------:|
| 0     | **+233.2%**        | +0.5%                   | +2.9%                   | 68x         |
| 1     | +4.6%              | +1.2%                   | +0.5%                   | 2.6x        |
| 2     | +10.2%             | +1.0%                   | +0.9%                   | 5.4x        |
| 3     | +26.3%             | +1.2%                   | +1.2%                   | 11x         |
| 4     | **+34.0%**         | +1.4%                   | +4.5%                   | 5.8x        |
| 5     | +24.0%             | +1.3%                   | +1.9%                   | 7.6x        |
| 6     | +17.1%             | +1.0%                   | +1.7%                   | 6.3x        |
| 7     | +9.0%              | +0.7%                   | +0.7%                   | 6.5x        |
| 8     | +9.1%              | +0.8%                   | +0.6%                   | 6.5x        |
| 9     | +5.0%              | +0.6%                   | +0.2%                   | 6.1x        |
| 10    | +4.0%              | +0.4%                   | +0.4%                   | 5.1x        |
| 11    | +20.0%             | +0.7%                   | +2.3%                   | 6.5x        |

- **E_shape** = cost of GELU→ReLU (no compensation). Dominates at 3-68x the replacement cost.
- **E_replacement** = cost of per-channel mean replacement (C3). Small: +0.4-1.4%.
- **E_projection** = additional cost of 3072→768 projection + constant bias. Adds +0.2-4.5%.

The GELU→ReLU positive-regime shape change is the dominant error source. The brief barely mentions it.

---

## 4. Norm measurements per regime (cycle 109)

Cycle 109 split each block's GELU output by sign(z), projected each half through W_down, measured L2 norms averaged over tokens:

| Block | ||pos|| | ||neg|| | ||output|| | (||pos||+||neg||-||out||) / (||pos||+||neg||) |
|------:|--------:|--------:|----------:|------:|
| 0     | 29.0    | 16.8    | 26.0      | 43%   |
| 1     | 15.7    | 13.2    | 11.5      | 60%   |
| 2     | 30.1    | 27.9    | 14.7      | 75%   |
| 3     | 34.3    | 34.8    | 14.2      | 80%   |
| 4     | 33.9    | 35.0    | 14.6      | 79%   |
| 5     | 35.3    | 34.3    | 17.2      | 75%   |
| 6     | 34.0    | 31.1    | 20.1      | 69%   |
| 7     | 36.0    | 30.8    | 23.9      | 64%   |
| 8     | 36.9    | 30.1    | 29.2      | 57%   |
| 9     | 43.6    | 32.3    | 36.9      | 51%   |
| 10    | 82.3    | 55.1    | 68.3      | 50%   |
| 11    | 164.7   | 124.3   | 80.5      | 72%   |

The last column is a scalar summary: how much smaller is the sum than the parts. In blocks 3-5 it's ~80%. **Caution**: these are L2 norms of 768-dim vectors averaged over tokens. The norms tell us the components are large relative to their sum. They do NOT tell us the geometry of how these 768-dim vectors relate — "cancellation" or "opposing directions" would require per-dimension analysis that wasn't done. See section 7 for rank data that complicates any simple "opposing directions" reading.

Correlation with cycle 181 cascade damage (marginal PPL when adding each block to the progressive swap):

| Block added | Marginal PPL | Norm ratio |
|:------------|------------:|-----------:|
| 0           | +0.78       | 43%        |
| 1           | +0.37       | 60%        |
| 4           | +3.47       | 79%        |
| 5           | +2.82       | 75%        |
| 6           | +3.04       | 69%        |
| 11          | +4.02       | 72%        |

Blocks with high norm ratios AND high absolute norms (blocks 4-6, 11) produce the biggest cascade jumps. Block 11's norms are enormous (165 + 124, output 80.5), so even small relative perturbations produce large absolute shifts in the residual stream.

---

## 5. The s_ratio sleight-of-hand

The brief claims "87-98% token-independent" for the negative regime, citing b_energy. Misleading.

**b_energy** = fraction of negative-regime variance explained by the mean. Within-regime decomposition.

**s_ratio** = ||s_leak|| / ||output||. Token-dependent negative signal relative to total block output:

| Block | b_energy | s_ratio | s as % of neg |
|------:|---------:|--------:|--------------:|
| 0     | 91%      | **20%** | 31%           |
| 1     | 87%      | **48%** | 42%           |
| 3     | 97%      | **45%** | 18%           |
| 4     | 97%      | **44%** | 18%           |
| 11    | 99%      | **18%** | 12%           |

The token-dependent negative signal is 18-48% of the total output norm. The 87-98% figure describes the decomposition *within* the negative regime, but the negative regime itself is 1-2.4x the output norm. 3% of a 2.4x-norm signal is still 7% of the output.

The brief uses b_energy (flattering) instead of s_ratio (alarming) when claiming the negative regime is "just a bias."

---

## 6. The interaction paradox (C5 is sub-additive)

Cycle 173 C5 (replace both regimes with means): damage is **sub-additive**.

| Block | C1 (kill neg) | C2 (kill pos) | Expected additive | C5 (actual) | Ratio |
|------:|--------------:|--------------:|------------------:|------------:|------:|
| 0     | 75.6          | 220,233       | 220,286           | 1,339       | 0.006 |
| 3     | 28.7          | 32.6          | 38.5              | 24.3        | 0.63  |
| 4     | 30.4          | 36.1          | 43.8              | 24.3        | 0.55  |
| 11    | 27.2          | 42.9          | 47.4              | 27.1        | 0.57  |

Replacing BOTH regimes with their means is less damaging than killing either one alone. Block 0: killing positive gives 220K PPL, killing negative gives 76. But replacing both with means gives only 1339.

This means the two regimes are not independent — C1's damage (zeroing neg) and C2's damage (zeroing pos) don't add. Replacing both with constants (C5) loses only the token-dependent variation, which is much cheaper than removing either regime's mean contribution entirely. The regimes' mean contributions are jointly calibrated.

---

## 7. The rank anomaly nobody discusses (cycle 174)

Cycle 174 (refuted for its original hypothesis) contains a buried finding:

| Block | rank_neg (95% energy) | rank_pos (95% energy) |
|------:|----------------------:|----------------------:|
| 0     | 406                   | 348                   |
| **1** | **453**               | **21**                |
| **2** | **441**               | **1**                 |
| 3     | 462                   | 343                   |

Block 2's positive-regime token-dependent signal is **rank-1**. Block 1's is rank-21. The negative regime is always high-rank (~450).

Implications the brief doesn't draw:
- Early blocks' positive regime is highly structured (near rank-1): essentially the same function scaled by one coefficient per token
- The negative regime is diffuse (rank ~450): varies across many independent directions
- By block 3, positive rank jumps to 343 — structure changes dramatically with depth
- Neither regime aligns with the gradient (cos ≈ 0.03 for both). Both compute in directions the Jacobian doesn't measure.

---

## 8. The mu_neg norm growth under iterative swap

Cycle 182: negative-regime norms grow when upstream blocks are swapped to ReLU.

| Block | Precomputed (float upstream) | Iterative (ReLU upstream) | Drift |
|------:|-----------------------------:|--------------------------:|------:|
| 0     | 14.9                         | 14.9                      | 0%    |
| 1     | 11.6                         | 13.1                      | +14%  |
| 4     | 33.9                         | 35.2                      | +4%   |
| 10    | 52.7                         | 56.5                      | +7%   |
| 11    | 120.3                        | 134.9                     | +12%  |

ReLU pushes more of the distribution into the negative regime. The bias correction chases a moving target. Iterative only recovers 5.6 PPL (37.0 vs 42.6) despite tracking the drift — drift correction is necessary but not sufficient. The remaining +63% comes from the shape change (E_shape), not bias miscalibration.

---

## 9. The superadditivity of composition

If cycle 181's per-block errors composed independently (multiplicative in PPL):
- Sum of log-PPL deltas: 0.28
- Predicted PPL: 30.1 (+33%)
- Actual PPL: 42.6 (+88%)
- **Superadditivity factor: 2.7x**

Per-block errors are positively correlated — confirmed by cycle 189 (cos(e_l, e_{l-1}) = 0.72). Each block's 768-dim error vector in the residual stream points in a similar direction to the previous block's error, causing pile-up.

One candidate mechanism (untested): the Jacobian mismatch. GELU'(0) = 0.5 (smooth), ReLU'(0) is 0 or 1 (discontinuous). This changes how perturbations propagate through each block, potentially creating a systematic directional bias. But this is speculation — the Jacobian comparison experiment (proposed but not yet run) would test it.

---

## 10. Contradiction inventory

### Contradiction 1: "Bias works per-block" vs "Bias fails at composition"

**Status: RESOLVED** (cycle 188). C3 all-12 blocks = PPL 27.3 (+20%). Much better than ReLU+bias (+88%). The brief's "composition mystery" was mostly the GELU→ReLU shape change, not composition of constant replacements. But +20% is still significant — each block's +1-2% C3 error compounds mildly superadditively.

### Contradiction 2: "Negative regime is just a bias (87-98%)" vs "Removing it is catastrophic"

**Status: RESOLVED.** Not contradictory. The negative regime IS mostly a constant, but that constant is 1-2.4x the output norm. "Just a bias" implies small; it's enormous. The 87-98% describes within-regime variance decomposition, not magnitude. s_ratio (18-48%) is the right metric.

### Contradiction 3: "Positive channels carry 99.9% of computation" vs "Both regimes essential"

**Status: RESOLVED.** L2 norm ≠ functional importance. Killing either regime is catastrophic (C1: +4-233%, C2: +6-970K%). Both are necessary. The "99.9%" claim was an L2 norm comparison that doesn't measure functional importance.

### Contradiction 4: Sub-additive within blocks, super-additive across blocks

**Status: RESOLVED** (cycle 189). Within a single block, C5 is sub-additive because replacing both regimes with means preserves their joint calibration — you lose token-dependent variation but keep the mean structure. Across blocks, cycle 189 measured cos(e_l, e_{l-1}) = 0.72 (averaged over layers), where e_l is the 768-dim residual stream error at layer l. The high cosine means successive layers' errors point in similar directions in the residual stream, causing pile-up rather than random-walk accumulation. This explains the 2.7x superadditivity.

### Contradiction 5: Iterative bias barely helps despite correcting distribution shift

**Status: RESOLVED** (cycles 190, 193). The full decomposition:

| Intervention | What changes | PPL | Delta |
|---|---|---|---|
| Baseline | Nothing | 22.7 | 0% |
| C3 (188) | Neg → per-channel mean | 27.3 | +20% |
| Hybrid (190) | Pos: GELU→ReLU shape | 29.0 | +28% |
| C3+Hybrid (193) | Both | 41.4 | +83% |
| ReLU+bias (181) | Both + projection loss | 42.6 | +88% |

Additive prediction: +20% + +28% = +48%. Actual combined: +83%. **Interaction: +35% excess.** The two interventions are not independent — doing both costs far more than the sum of doing each alone. Why the interaction is so large is not explained by the norm data (see section 11).

The 3072→768 projection loss (C3+Hybrid vs ReLU+bias) adds only ~5%. Almost all of cycle 181's damage comes from the two perturbations interacting, not from information loss in the projection.

---

## 11. State of understanding

### What we can measure

Both activations are z scaled by a gate: GELU(z) = z·Φ(z), ReLU(z) = z·1[z>0]. Both continuous. The gates differ for |z| < ~3: Φ(z) transitions smoothly (0.16 at z=-1, 0.50 at z=0, 0.84 at z=1), while 1[z>0] jumps. For z < 0, GELU contributes a small negative output (max 0.17); ReLU contributes 0. For 0 < z < 2, GELU attenuates (Φ(z) < 1); ReLU passes z fully.

GPT-2 learns z_mean ≈ -1 per channel (random init gives z_mean ≈ 0). ~80-90% of channel×token pairs have z < 0. Swapping GELU→ReLU zeroes out those channels AND changes the gate for the positive channels. Both effects are measured (section 3 decomposition, cycles 188/190).

### What we tried to explain and got wrong

We attempted to interpret the section 4 norm data (||pos|| ≈ ||neg|| ≈ 35, ||output|| ≈ 14) as "the FFN computes via cancellation of two opposing signals." This is a scalar narrative projected onto 768-dimensional vectors. The norm ratio tells us the components are large relative to their sum. It does not tell us the components "oppose each other" or that the network "computes a difference" — those are 1D interpretations of a 768D measurement.

The narrative directly contradicts section 7's rank data: in blocks 1-2, the positive signal varies across tokens along ~1-21 directions while the negative signal varies along ~450 directions. Signals in different subspaces of R^768 cannot be "cancelling" in any geometric sense. The norm coincidence (||pos|| ≈ ||neg||) needs a different explanation.

The +35% interaction term (section 10, contradiction 5) between C3 and hybrid interventions is real but unexplained. We attributed it to "destroying both sides of a cancellation" — which is the same bad narrative. The actual mechanism producing the superadditive interaction in 768 dimensions is not understood.

### What actually holds up

Established by construction (not interpretation):
- The gate difference: GELU(z) = z·Φ(z) vs ReLU(z) = z·1[z>0]. The functions disagree for |z| < ~3.
- GPT-2 learns to place most pre-activations in the disagreement zone (z_mean ≈ -1, cycle 165).
- The per-block error decomposition (section 3): shape change dominates replacement cost by 3-68x.
- C3 all-12 costs +20%, hybrid all-12 costs +28%, both together +83% — the interaction is +35% above additive (cycles 188, 190, 193).
- Cascade errors are directionally aligned in the residual stream: cos(e_l, e_{l-1}) = 0.72 (cycle 189).
- The two regimes are jointly calibrated — their mean contributions are not independent (section 6, C5 sub-additivity).

### What remains unexplained

1. **Why does GPT-2 learn z_mean ≈ -1?** Training moves pre-activations into the GELU bump zone. The computational purpose is unknown.
2. **What does the 768-dim structure of pos and neg look like?** We have norms and ranks. We don't have a geometric description of how these 768-dim vectors relate across tokens, blocks, or to the residual stream's functionally important directions.
3. **Why is the C3 × hybrid interaction so large (+35%)?** The two interventions are not independent — but the mechanism in 768 dimensions is not captured by any scalar proxy we've measured.
4. **Why does the positive signal in early blocks have such low rank?** Block 2 positive = rank-1, negative = rank-441. This structural asymmetry is unexplained.

### What a good explanation would look like

A satisfactory answer would let you predict: given W_up, W_down, and the input distribution, what does the GELU bump compute that ReLU cannot? Not in terms of scalar norms, but in terms of the 768-dim output vector's relationship to downstream processing (attention, next FFN block, unembedding). It would explain why training converges on z_mean ≈ -1, why the positive and negative regimes have different rank structures, and why perturbing both is worse than the sum of perturbing each.

---

## Data sources

All result JSONs in `autoresearch/results/`, experiment scripts in `experiments/`. See `docs/gelu_bump_investigation_brief.md` for the complete reference table.

Tie-breaker experiments: cycles 188 (C3 composition), 189 (error direction), 190 (shape isolation), 193 (combined C3+hybrid).

---

## 12. Postscript: what was wrong with this document

**Added 2026-04-08, after the cycle 280 rank-collapse finding and the parameter-golf cross-project comparison forced a reframing.**

Sections 1–11 above are factually correct but framed wrong. The framing problem is the same one the rescue note (`/Users/mihai/.claude/memory/handoff/quant_aware_training.md`, 2026-04-02) flagged for the 190 prior cycles, and which I failed to fully internalize when writing this audit. This section names the mistake, gives the geometrically correct framing, and explains why every numerical finding above is a special case of a more general object that I should have been measuring all along.

### The mistake: studying "the bump" as the object

Throughout sections 1–11 the object of study is "the negative-zone GELU bump" — a chunk of the activation function defined by the numerical condition `z < 0`. Cycle 173 measures C3 (replace negatives with mean). Cycle 181 measures full ReLU+bias swap. Cycle 188 measures progressive C3 across blocks. Cycle 217 measures Δ̃ = (GELU − ReLU) @ W_down. Cycle 218 measures Δ̃ alignment with downstream readers. The implicit framing throughout: *the bump is a thing, and we are studying what it does*.

This is the wrong unit of analysis. The bump is not a thing; it is a numerical accident of where the GELU function happens to differ from a piecewise-linear baseline. The actual object — the one that has geometric content and that transfers across activation functions — is **the curvature residual function**:

```
δ_σ(z) = σ(z) − σ_lin(z)
```

For any pair of activations (σ, σ_lin), δ_σ is a function R → R. It has a shape, a support, a magnitude, a sign pattern. For (GELU, ReLU) the support is roughly z ∈ (−3, 3) with a dip around z = −1; that dip is what we called "the bump." For (ReLU², ReLU) the support is z > 0 with a different sign profile; for (LeakyReLU², ReLU²) the support is z < 0 with shape `0.25 z²`. **All three are decomposed identically.** The cycles in this document studied one specific shape and treated it as a privileged object. They should have studied the shape-agnostic geometric machinery and parameterized over (σ, σ_lin) at the end.

### The geometrically correct object

Per channel, the FFN block output decomposes:

```
f_block(x) = W_proj · σ(W_fc · x + b_fc) = Σ_c σ(z_c(x)) · w_c
```

where `w_c` is the c-th column of W_proj and `z_c(x)` is the c-th pre-activation. **The columns w_c are activation-independent**; only the per-channel scalar weights `σ(z_c)` change across activation functions.

The contribution of σ's nonlinearity, relative to a linear baseline σ_lin, is:

```
Δy(x) := f_σ(x) − f_{σ_lin}(x) = W_proj · δ_σ(z(x)) = Σ_c δ_σ(z_c(x)) · w_c
```

This is a per-token vector in residual-stream R^768. It is the *only* thing the activation function contributes to the block output beyond its linear part. Every measurement in this document — and every measurement in cycles 102–264 — is a property of `Δy(x)` as a function of x, for the specific case `(σ, σ_lin) = (GELU, ReLU)`.

The geometric questions you can ask about Δy are activation-pair-agnostic:

1. **Rank**: SVD `[Δy(x_t)]_t` over a calib slice. Report `rank_95`. (Cycle 217 did this for Δ̃; got rank_95 ≤ 64 in 9/12 blocks. The framing was "rank of the bump"; the correct framing is "rank of the curvature residual.")
2. **Subspace alignment**: principal angles between top-k left singular vectors of `[Δy(x_t)]_t` and downstream readers (next block's W_fc, Q, K, V, unembedding rows). Cycle 218 did this for Δ̃; CONFIRMED on c_fc 11/11 blocks.
3. **Per-token magnitude**: distribution of `‖Δy(x)‖_G` under the pullback metric. Section 4 of this document approximated this by reporting `‖pos‖`, `‖neg‖`, `‖output‖` separately and computing scalar ratios — that was the "scalar narrative" mistake explicitly called out in section 11.
4. **Pullback κ contribution**: patch the block with σ_lin (with optional first-moment correction), recompute the residual stream trajectory under G = U^T U, measure the per-layer κ shift. Cycle 258 did this for (GELU, ReLU+b_leak) and found a +47% κ kink at layer 4.
5. **Energy decomposition vs a frozen low-rank subspace**: project Δy onto col(A) for some W_proj column subspace of interest (e.g., the cycle 280 frozen rank-2 factor at block 1), report kept vs discarded energy fraction.

These five measurements **are** the geometric content of the GELU vs ReLU comparison. Sections 3–9 of this document are special cases of measurement 1, 2, and 3 — sometimes computed in dimension-reducing ways that lose the geometric content (the L2-norm summaries in section 4 are the worst offenders).

### What the section-by-section findings actually mean

Re-read with the right object in mind:

| Section | What it measured | What it actually measures geometrically |
|---|---|---|
| 3 | Per-block error decomposition (E_shape, E_replacement, E_projection) | The PPL impact of three different ways of replacing `δ_σ(z)` with a coarser approximation: zero (kill), per-channel mean (constant), per-channel mean projected through W_proj (constant in residual stream). Each is a successively coarser approximation of the rank-1 truncation of Δy. The "shape change dominates by 3-68×" finding is restating that Δy has rank ≫ 1 — most of its content does not survive a constant approximation. |
| 4 | L2 norms of pos and neg parts of the output | A scalar projection of the per-token vector `Σ_{c: z_c>0} σ(z_c) w_c` (and analogously for negative). Throws away every directional signal in R^768. Section 11 already flagged this as the "scalar narrative" failure; the geometric replacement is "compute the column-space of the positive-active set across tokens, compute its principal angle with the negative-active column-space." That measurement was done partially in cycle 261 (REFUTED in shape but the data was real). |
| 5 | b_energy vs s_ratio | b_energy is the fraction of within-regime variance explained by the per-channel mean — a scalar property of the conditional distribution `δ_σ(z) | z<0`. s_ratio compares that scalar to total output norm. Both throw away direction. The geometric replacement: SVD of the per-token negative-zone contribution `Σ_{c: z_c(x)<0} δ_σ(z_c) w_c` over tokens, report the spectrum. |
| 6 | C5 sub-additivity within blocks (replacing both regimes is cheaper than replacing either alone) | This is the most interesting numerical finding in the document, and it has a direct geometric reading I missed: the two replacements (kill positive, kill negative) are operations on *non-orthogonal* subspaces of Δy. The non-orthogonality is what creates the sub-additivity. The right measurement: what is the angle between `Δy_pos = Σ_{z_c>0} δ_σ(z_c) w_c` and `Δy_neg = Σ_{z_c<0} δ_σ(z_c) w_c` across tokens? If small, the two operations interfere and C5 < C1 + C2. If large, they're independent and C5 ≈ C1 + C2. The 0.55× ratio at block 4 means the two subspaces are correlated, and the *angle* between them is the missing scalar. This is testable in ~5 minutes. |
| 7 | Rank anomaly (block 2 positive rank-1, negative rank-441) | This is already a geometric measurement, but on the wrong object: it measures the rank of the *activation* matrix on the positive-z subset, not the rank of `Δy_pos` (the residual-stream contribution). Cycle 174's "block 2 pos = rank 1" probably means the *output direction* is rank-1, not that the activation values are rank-1. The right experiment is to recompute on `Σ_{z_c>0} σ(z_c) w_c` and see if rank 1 transfers. |
| 8 | μ_neg norm growth under iterative swap | Tracks a scalar `‖μ_neg^(l)‖` per block under sequential swap. The geometric content is "where does the residual stream live after l blocks of patched activations, and how does that location change the per-channel pre-activation distribution at block l+1?" Scalar norm tracking is a 1D shadow of a 768-dim trajectory. Item 270 (the per-token d_l measurement) does this correctly. |
| 9 | Superadditivity, cos(e_l, e_{l−1}) = 0.72 | Aggregated cosine of per-block error vectors. The aggregation is the loss: per-token cos can be very different from average cos. Item 270 measured the per-token version; it's the right quantity. |
| 10 | Contradiction inventory | Mostly resolved correctly. Contradiction 5 (C3 × hybrid interaction +35% above additive) is the key open one and has the same geometric reading as section 6's C5 sub-additivity: the two interventions act on non-orthogonal subspaces and the interaction term is the cosine of the subspaces. Same ~5-minute experiment resolves it. |
| 11 | "What we tried to explain and got wrong" | Correctly identifies that the pos/neg cancellation narrative was a 1D projection of a 768D measurement. Does not name the right object (Δy) or the right framework (δ_σ as a function family). |

### What the corrected framework predicts about activation pairs other than (GELU, ReLU)

Because Δy is parameterized by `(σ, σ_lin)`, every measurement in this document has an analog for any other activation pair. Specifically, for the parameter-golf project that uses ReLU² (`max(0, z)²`):

- **The (ReLU², ReLU) curvature residual** is `δ(z) = relu(z)² − relu(z) = relu(z) · (relu(z) − 1)`. Support: z > 0. Sign: negative on (0, 1), zero at 0 and 1, positive for z > 1.
- **The (LeakyReLU², ReLU²) curvature residual** is `δ(z) = 0.25 z² for z < 0, else 0`. Support: z < 0. Quadratic shape.

These have *different* shapes from the GELU bump but the *same* geometric machinery applies. Items 312 and 313 in `autoresearch/agenda.md` formalize this: a single parameterized script `experiments/curvature_residual_geometric.py` taking `(σ, σ_lin, model, blocks, A_factor_optional)` as arguments, computing all five geometric measurements above, and producing a JSON per tuple. Cycles 217, 218, 235, 258, and 311 are five hand-coded special cases of that one script.

The cross-project doc `parameter-golf/docs/curvature_residual_geometric.md` proposes running the same framework on parameter-golf checkpoints. The headline question — "is the rank-2 collapse from cycle 280 a generic phenomenon or a GELU artifact?" — is a 5-minute experiment under the corrected framework (PG-1 in that doc). The leaderboard question — "why does LeakyReLU² beat ReLU² on the parameter-golf leaderboard?" — is PG-4, also a single application of the unified framework with `(σ, σ_lin) = (LeakyReLU², ReLU²)`.

### What this means for the cycle 102–264 corpus

About 25 cycles of "GELU bump investigation" cycles (102, 173, 181, 182, 184, 188, 189, 190, 191, 193, 217, 218, 220, 228, 234, 235, 237, 238, 239, 246, 253, 258, 263, 264) are not "wrong" in the sense of producing incorrect numbers. They produced correct numbers about a specific instance of a more general object. The mistake was not computing the general object — they computed properties of `Δy` for `(σ, σ_lin) = (GELU, ReLU)` only. With the geometric framework in place, those cycles are best read as **the (GELU, ReLU) row of a table whose other rows have not been filled in yet**.

The 200-cycle investment was not wasted, but it was scoped to one shape when the right scope was the function family. The corrected framework in items 312/313 fills in the missing rows in ~30 minutes total of additional compute.

### What I should have written instead of sections 1–11

A document organized by geometric measurement (rank of Δy, subspace alignment, pullback κ, energy decomposition), parameterized by activation pair, with (GELU, ReLU) as the first row of every table and (ReLU², ReLU), (LeakyReLU², ReLU²), and other pairs as additional rows to fill in. The "GELU bump" would not appear as an organizing principle; it would appear as a description of one particular δ_σ shape. The cycle 280 rank-collapse finding would be cited as a structural property of the trained model — a property of W_proj's column geometry that constrains where any Δy can go — and would be derived independently of which σ generated it.

I am leaving sections 1–11 in place because the data audit is correct and the contradictions resolved in section 10 are still valid. But the framing of "what does the GELU bump compute" should be replaced everywhere by "what is the geometric structure of Δy = W_proj · δ_σ(z(x)), and how does it depend on the activation pair (σ, σ_lin)?". The first version of that question was the wrong question.

The rescue note was right at session start. I half-corrected in section 11 by removing the bad scalar narrative; I did not fully correct by replacing it with the geometric framework. This postscript does that.
