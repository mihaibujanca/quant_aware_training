# Cycle 135 Review — Curved Inference Census (Item 145)

Item 145. Reviewer: design-review agent.

## R1: Premise validity — OK with caveat

Manson 2025 demonstrates salience-curvature anticorrelation on LLaMA-3.2-3b (28 layers, r=-0.89). GPT-2 has 12 layers, yielding only 11 curvature points. With N=11, Spearman rho=0.7 gives p~0.02 — statistically meaningful but fragile. One outlier layer can swing rho by 0.2. The plan correctly identifies this but the confirmation threshold (|rho| >= 0.7) may be too aggressive for N=11. Acceptable since the agenda set the threshold, not the plan.

## R2: Measurement soundness — OK

The `aleph/pullback_geometry.py` module implements all needed functions correctly: `compute_salience` uses `pullback_norm(delta, U)`, `compute_curvature` uses the 3-point central difference formula with the proper cross-term `sqrt(||a||^2_G ||v||^2_G - <a,v>^2_G) / ||v||^3_G`. The plan's cross-check (step in Validation) of manual `U @ delta` vs library output is good practice.

## R3: Index alignment — MINOR FIX NEEDED

The plan notes curvature has 11 values (layers 1-11) and PPL-delta has 12 values (blocks 0-11). Step 7 says "use PPL_delta[1:12] aligned with kappa." This is correct for correlation, but the salience-curvature alignment in step 4 says "use S[1:12] to match kappa[0:11]" — salience has 12 values for layers 0-11 (inter-layer deltas), and curvature has 11 values for layers 1-11. S[1:] gives S for layers 1-11, which is 11 values matching kappa. This is correct. No fix needed, but the plan text is slightly confusing — consider a comment in the experiment script clarifying the index semantics.

## R4: Single-block swap PPL — OK

The plan recomputes swap PPL for all 12 blocks rather than reusing prior data from specific blocks. This is correct: prior cycles only tested subsets [1,4,5,7] and used different calibration data. A consistent all-12-block measurement with the same 200 sequences and 3 seeds is needed for valid correlation.

## R5: ArXiv reference check — OK

ArXiv 2507.21107 (July 2025) is already surveyed at `autoresearch/surveys/curved_inference_geometric_interpretability.md`. The survey confirms the paper exists and has been read. The pullback metric framework, the r=-0.89 anticorrelation on LLaMA, and the methodological caveat about UMAP are all documented.

## Summary

One advisory (no code change needed):
1. Add a comment in the experiment script clarifying index alignment between salience (12 values, layers 0-11), curvature (11 values, layers 1-11), and PPL-delta (12 values, blocks 0-11).

No blocking issues.

**Verdict: ACCEPT.**
