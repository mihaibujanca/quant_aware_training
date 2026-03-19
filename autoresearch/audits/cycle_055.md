# Audit — Cycle 055

**Item 71**: Jacobian Frobenius std as corpus-sensitivity predictor — GPT-2 FFN
**Verdict**: CONFIRMED — agrees with evidence

## Guardrail Checks

| Check | Status | Notes |
|-------|--------|-------|
| G1: Implementation correctness | PASS | `ffn_forward`, `compute_jacobian`, k-NN (k=5, n=128), and P1 aggregation are identical to cycle 048. Shakespeare P1 loaded from cycle 048 JSON, not recomputed. |
| G2: Reproducibility | PASS | Fixed seed=42, deterministic CPU, versioned WikiText-2 dataset. Single-run is adequate — Spearman is over 12 block identities, not random samples. |
| G3: Results interpretation | PASS | Scatter shows genuine monotonic correspondence across full range. Inner-block rho (0.939) virtually unchanged from full (0.944) — no leverage-point inflation. WikiText-2 P1 magnitudes are systematically lower than Shakespeare, but block ranking is preserved. |
| G4: Counterintuitive check | N/A | Result is expected; corpus-invariance of an intrinsic geometric property is the natural outcome. |

## Key Observation

WikiText-2 P1 values are uniformly lower than Shakespeare (all 12 points below y=x line). This means P1 absolute magnitude IS corpus-dependent — only the rank ordering is invariant. Any downstream use must compare P1 within a single corpus run, not across corpora.

## Status: PASS — no flags
