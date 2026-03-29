# Audit — Cycle 110

**Item**: 121 — Channel-level bump anatomy: compression relevance — GPT-2 FFN [INTERPRETABILITY]
**Verdict**: INCONCLUSIVE
**Audit status**: PASS

## G1: Code quality

**Conv1D handling**: Line 220-222 uses `model.transformer.h[b].mlp.c_proj.weight` without `.T`. GPT-2 Conv1D(nf=768, nx=3072) stores weight as (3072, 768). Row j = output projection for channel j. Correct — verified empirically.

**GELU variant**: Lines 101-103 use `F.gelu(z, approximate='tanh')`, matching GPT-2's tanh-approximate GELU. Correct.

**GELU derivative**: Lines 132-133 use symmetric finite difference `(GELU(z+eps) - GELU(z-eps)) / 2*eps` with eps=1e-4. Clean, avoids autograd overhead. Result is `.abs().mean(dim=0)` — taking mean of absolute derivative per channel. Correct.

**Importance variants**: Three variants implemented (lines 140-143):
- A (full): `std(h_j) * ||w_out_j|| * mean(|GELU'(z_j)|)` — includes tautological coupling
- B (weight-only): `||w_out_j||` alone — no activation coupling
- C (activation-weight): `std(h_j) * ||w_out_j||` — partial coupling

Design is sound. Variant B as verdict metric avoids the tautological artifact in variant A.

**Hook placement**: Line 84 hooks `mlp.c_fc`, capturing its output (pre-GELU z). Correct.

**Data**: Deterministic loading from `data/openwebtext_sample.txt`, no seeds needed. 4096 tokens, 128 seq_len. PPL guard at lines 200-201 (20-65 range).

## G2: Determinism

No seeds or random operations. Data loading is deterministic (fixed file, sequential slicing). Model weights are frozen. PASS.

## G3: Verdict criteria

Criteria: CONFIRMED if rho_B > 0.3 in >=8/12, REFUTED if |rho_B| < 0.1 in >=8/12.

Results: 0 blocks with rho_B > 0.3. 2 blocks with |rho_B| < 0.1 (blocks 0 and 2).

Neither threshold met. INCONCLUSIVE is the correct verdict.

However, the result is more informative than "inconclusive" suggests. 8 of 12 blocks have rho_B < -0.2, and blocks 8-11 show rho_B in [-0.49, -0.60]. The true story is a **depth-dependent sign flip**: weakly positive in early blocks, strongly negative in deep blocks. This is a structured anti-correlation, not noise.

## G4: Tautological artifact

The rho_A finding (mean +0.79, range [+0.66, +0.87]) is correctly flagged as tautological. Importance variant A includes `mean(|GELU'(z_j)|)` which is mechanically coupled to bump fraction — channels spending more time in the bump zone have higher mean |GELU'| by construction. This inflates rho_A and would produce a false "confirmed" if used as the verdict metric.

The experiment design correctly anticipated and isolated this artifact. Good.

## G5: What did we learn?

1. **Bump channels are anti-important for compression in deep layers.** Weight-only importance (||w_out_j||) is negatively correlated with bump fraction in blocks 3-11. The channels living in the nonlinear GELU zone have *small* output projections.

2. **Bump-critical channel count drops sharply with depth** (355 at block 0 to 38 at block 9). In deeper layers, few channels are both high-bump and high-importance simultaneously.

3. **The tautological coupling in standard importance scores is severe.** rho_A = +0.79 mean vs rho_B = -0.28 mean — opposite signs. Any importance metric including GELU sensitivity will overestimate bump channel importance.

4. **For compression**: bump zone nonlinearity is geometrically interesting but orthogonal to what matters for output reconstruction. Compressing bump-zone channels preferentially would be a mistake — they contribute less to the output.

## G6: Is weight-only importance the wrong proxy?

`||w_out_j||` measures how much channel j *can* influence the output, regardless of activation patterns. This is a genuine non-tautological proxy, but it ignores activation statistics entirely.

Variant C (rho_C, mean +0.45) includes `std(GELU(z_j))` which partially reintroduces coupling: channels with high bump fraction tend to have variable GELU outputs (the nonlinear zone amplifies input variance). The positive rho_C could be partly tautological via this path.

A better non-tautological proxy might be: **actual output contribution** = `mean(|h_j|) * ||w_out_j||` (which the code computes for `contrib` at line 164 but doesn't use for correlation). This captures both the typical activation magnitude and the output projection, without referencing GELU derivative or bump fraction. This would be a useful follow-up.

Another option: **Fisher information** or **gradient-based importance** on a held-out loss, which avoids any structural coupling to bump geometry.

## Flags

- **FLAG: Anti-correlation is the real finding.** The INCONCLUSIVE verdict undersells the result. The depth-dependent negative rho_B is systematic and informative. Consider whether this warrants a follow-up item framed as "bump channels have low output weight norms in deep layers."
- **FLAG: Variant C (rho_C) may have residual tautological coupling** via std(GELU(z)) — not as clean as rho_B.
