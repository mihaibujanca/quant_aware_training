# Cycle 110 Plan: Channel-Level Bump Anatomy — Compression Relevance (GPT-2 FFN)

**Item:** 121
**Script:** `experiments/bump_compression_relevance_gpt2.py` (new)

## Pre-flights

**P1 -- Null hypothesis:** Bump fraction and compression importance are orthogonal (|ρ|<0.1).

**P2 -- Criteria (verbatim from agenda):**
- Confirmed if: Spearman(bump_fraction, importance) > 0.3 in >=8/12 blocks
- Refuted if: |Spearman| < 0.1 in >=8/12 blocks
- Inconclusive if: moderate or sign-varying

**P3 -- Novelty:** Cycle 108 computed Spearman(σ_j, importance) but σ is not bump_fraction. This cycle uses per-channel bump_fraction = fraction of tokens with |z_j|<2 (the actual bump zone membership).

**P4 -- Confounds:** Importance formula includes `mean(|GELU'(z)|)` and `std(GELU(z))`, both driven by bump-zone membership. A naive correlation is partially tautological. **Mitigation:** compute THREE importance variants: (A) full formula, (B) `||w_out_j||` only (weight-only, zero tautology), (C) `std(GELU(z_j)) * ||w_out_j||` (activation magnitude, mild tautology). Report Spearman for all three. Apply agenda criteria to variant B (weight-only). Report A and C for context.

**P5 -- Cost:** Single forward pass on 4096 tokens, 12 blocks × 3072 channels. <3 min on MPS.

**P6 -- Data:** `data/openwebtext_sample.txt`, same 4096 tokens as cycles 103/108.

**P7 -- Abort:** Baseline PPL outside [25, 65]. NaN in pre-activations.

## Implementation

1. **Load GPT-2 + data.** 4096 tokens from OWT sample. Single deterministic pass.

2. **Collect pre-activations.** Hook `mlp.c_fc` output for all 12 blocks. Store z (N, 3072).

3. **Per-channel bump fraction.** For each block, each channel j:
   `bump_frac_j = (|z_j| < 2.0).float().mean()` — fraction of tokens in bump zone.

4. **Importance scores — three variants per block:**
   - (A) Full: `std(GELU(z)) * ||w_out|| * (|mean(GELU(z))| + |mean(GELU'(z))| + 1e-6)`
   - (B) Weight-only: `||w_out_j||` (c_proj row norms, no activation dependence)
   - (C) Activation-weight: `std(GELU(z_j)) * ||w_out_j||`

5. **Spearman correlations.** For each block (12 blocks), compute:
   - ρ_A = Spearman(bump_frac, importance_A)
   - ρ_B = Spearman(bump_frac, importance_B)   ← **primary metric for verdict**
   - ρ_C = Spearman(bump_frac, importance_C)

6. **Bump-critical channels.** Per block: channels in top quartile of BOTH
   bump_frac AND importance_B. Count them. Measure their aggregate contribution
   to block output: `||h_j * w_out_j|| / ||h * w_out||` summed over bump-critical j.

7. **Per-token output variance.** For bump-critical channels, compute
   `var_j = Var_tokens(GELU(z_j))`. If median var_j < 0.01: near-constant
   (replaceable by bias). If > 0.01: token-dependent gating.

8. **Verdict logic.**
   ```
   blocks_above = sum(rho_B[i] > 0.3 for i in range(12))
   blocks_near_zero = sum(abs(rho_B[i]) < 0.1 for i in range(12))

   if blocks_above >= 8:    CONFIRMED
   elif blocks_near_zero >= 8: REFUTED
   else:                       INCONCLUSIVE
   ```

9. **Output:**
   - JSON: `autoresearch/results/cycle_110_bump_relevance.json`
     Per-block: ρ_A, ρ_B, ρ_C, bump-critical count, aggregate contribution, median var.
   - Plot: 3-panel bar chart (ρ_A, ρ_B, ρ_C per block). `..._bump_relevance.png`.
