# Task
Survey tropical geometry literature for neural network compression relevance. Extract: representation schema, ReLU→tropical conversion algorithm, parameter efficiency comparison.

# Hypothesis
- **Confirmed if:** tropical representations offer provably or empirically fewer parameters for the same piecewise linear function, with a computable conversion
- **Refuted if:** tropical representation is equally or more expensive, or conversion is intractable
- **Inconclusive if:** theoretical advantage exists but no practical algorithm is available

# Pre-flight Results
- P1 PASS: No tropical geometry work in PAST EXPERIMENTS (only Balestriero spline survey in cycle 001).
- P2 PASS: No dependencies — standalone literature survey.
- P3 PASS: Clear confirmed/refuted/inconclusive criteria; single-cycle literature task.
- P4 PASS: No modifications to program.md or prompts needed.

# Steps
1. **Fetch Zhang et al. 2018** — WebFetch `https://arxiv.org/abs/1810.05228` ("Tropical Geometry of Deep Neural Networks"). Extract: (a) tropical polynomial definition for ReLU networks, (b) duality between network parameters and tropical hypersurface, (c) any conversion algorithm (weights → tropical monomials), (d) parameter count of tropical form vs weight matrices.

2. **Fetch Maragos et al. 2021** — WebSearch `"Tropical Geometry and Machine Learning" Maragos 2021 arxiv`, then WebFetch the paper page. Extract: (a) morphological / tropical representations of DNNs, (b) any practical algorithms or implementations, (c) complexity results.

3. **Search for follow-up work** — WebSearch `tropical geometry ReLU network compression piecewise linear 2022 2023 2024 2025`. Check if anyone has built practical tools or measured parameter savings. Note any code repos.

4. **Compare to MASO framework** — Read `autoresearch/surveys/balestriero_spline_theory.md` Techniques 1-3 (already known from cycle 001). Key comparison: tropical polynomial monomial count vs MASO affine map dictionary size for equivalent PWL functions. Does tropical form compress the dictionary or just re-encode it?

5. **Write survey** — Create `autoresearch/surveys/tropical_geometry.md` with sections: (a) Core representations (tropical polynomial, Newton polytope, tropical hypersurface), (b) Conversion algorithm + complexity (ReLU weights → tropical form), (c) Parameter count comparison (tropical monomials vs weight matrices, for networks of width w and depth d), (d) Comparison to MASO affine dictionary, (e) Outcome verdict with justification.

6. **Update agenda** — In `autoresearch/agenda.md`, move item 2 from NEXT EXPERIMENTS to PAST EXPERIMENTS under Cycle 002 with outcome, commit hash placeholder, and artifact path `autoresearch/surveys/tropical_geometry.md`.

7. **Commit** — `[autoresearch] cycle 002: tropical geometry survey`

# Validation
- Survey must answer all three focus questions from the brief: (a) parameter efficiency, (b) conversion algorithm, (c) complexity.
- Outcome must be one of confirmed/refuted/inconclusive with specific justification referencing paper results.
- If "confirmed": must cite a concrete algorithm with polynomial complexity AND a parameter count comparison showing savings.
- If "inconclusive": must state exactly what is missing (e.g., "algorithm exists but exponential in depth").

# Artifacts
- `autoresearch/surveys/tropical_geometry.md` — main deliverable
- Updated `autoresearch/agenda.md` — item 2 moved to PAST EXPERIMENTS

# Fallbacks
- If Zhang 2018 arxiv is inaccessible via WebFetch, search for the ICLR 2018 published version or any mirrors. The paper's core results (Theorem 1: ReLU network = tropical rational map) are widely cited and summarized in secondary sources.
- If no practical conversion algorithm exists in either paper, explicitly mark outcome as "inconclusive" and note that tropical theory characterizes but does not compress. Flag item 8 (minimal network construction) as the more promising synthesis route.
- If parameter comparison is not given explicitly in the papers, derive it: a depth-d width-w ReLU network has O(dw²) weight parameters; tropical form has up to O(w^d) monomials. If monomial count is exponential in depth, that alone refutes parameter efficiency.
