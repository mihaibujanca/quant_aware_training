# Item
2. Survey: Tropical geometry and neural networks

# Context
P0 priority. Tropical geometry as alternative representation for piecewise linear functions computed by ReLU networks. Previous cycle (001) confirmed MASO theory actionable for activation census and affine map extraction. This survey tests if tropical polynomials offer parameter efficiency gains.

# Dependencies
None. Standalone literature survey. Item 3 (linear region census) is independent and can run in parallel.

# Key Files
- Target papers: Zhang et al. "Tropical Geometry of Deep Neural Networks" (2018), Maragos et al. "Tropical Geometry and Machine Learning" (2021)
- Existing artifact: `autoresearch/surveys/balestriero_spline_theory.md` (cycle 001 context)

# Constraints
- Focus on three questions: (a) parameter efficiency vs. weight matrices, (b) algorithmic conversion from ReLU networks, (c) computational complexity
- "Confirmed" requires both theoretical advantage AND computable algorithm
- "Inconclusive" acceptable if theory exists but algorithm unavailable

# Handoff
Read Zhang et al. (2018) and Maragos et al. (2021). Extract: tropical polynomial representation schema, conversion algorithm (ReLU → tropical), parameter count comparison on standard test cases (2D moons/spirals networks). Compare to MASO effective affine map dictionary. Write `autoresearch/surveys/tropical_geometry.md` with (a) core representations, (b) conversion algorithm + complexity, (c) empirical parameter counts vs. weight matrices, (d) outcome (confirmed/refuted/inconclusive).
