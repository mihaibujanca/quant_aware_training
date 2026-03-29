verdict: APPROVE
round: 1

## R1: Criteria faithfulness
Agenda criteria quoted verbatim. Verdict logic (step 7) maps: top3_frac >= 0.70 in >= 8 layers
-> CONFIRMED; max_frac <= 0.15 in >= 8 layers -> REFUTED; else INCONCLUSIVE. Matches agenda. Pass.

## R2: Measurement soundness
Per-head amplification = ||Attn_h(LN(x+delta)) - Attn_h(LN(x))||_2 / ||delta||_2.
This is the right quantity: it measures how much each head individually amplifies the
residual-stream perturbation from FFN compression. The denominator normalizes across
transitions so different block-L compression errors don't distort the heatmap. Pass.

## R3: Known pitfalls
- **MP7 (post-LayerNorm):** The plan feeds perturbed input through LN_1 of block L+1,
  which is correct — we're measuring the *functional* amplification through LN+Attn, not
  geometry of post-LN representations. The perturbation delta is measured pre-LN (residual
  stream). LN is part of the function being measured, not a confound. Not triggered.
- **MP1 (trivial threshold):** Under uniform distribution, top-3 = 3/12 = 25%. Confirmed
  threshold is 70%, refuted is max <= 15% (~1.8x uniform share). Both well-separated
  from null. Not triggered.
- **MP8 (representation vs functional rank):** We measure functional amplification
  (output perturbation), not representation dimension. Not triggered.
- **MP5 (insufficient capacity):** Single measurement, not a comparison of approaches. N/A.
- **MP2, MP3, MP4, MP6:** Not applicable (no fitting, no detection threshold, no domain transfer, metric varies).

## R4: Cost and feasibility
12 FFN compressions (~10 min) + 11 forward passes with hooks (~3 min). Well within budget.
Reuses derive_ffn_compression infra. Single seed for per-head measurement is fine — the
quantity is deterministic given a fixed compression; variance comes only from data sampling.

## R5: Verdict logic soundness
Binary thresholds (70% vs 15%) with clear gap. The inconclusive zone (15-70% or mixed
across layers) is well-defined. 11 transitions measured against 8-layer threshold. Sound.

## Advisory

**A1 (LayerNorm amplification/suppression).** LN can amplify or suppress perturbations
depending on the perturbation direction relative to the current activation vector. The
measured per-head amplification includes LN's effect, not just the head's linear map.
This is actually desirable (LN is part of the real forward pass), but worth noting that
"head amplification" is really "LN+head amplification." Not blocking.

**A2 (Single compression seed).** The plan uses one FFN compression per block (no seed
sweep). The amplification measurement is deterministic given the compression, but the
compression itself has minor seed dependence (lstsq+Adam). For a census/heatmap this
is acceptable — the question is structural (which heads), not about precise magnitudes.
If results are borderline, re-run with 3 seeds. Not blocking.
