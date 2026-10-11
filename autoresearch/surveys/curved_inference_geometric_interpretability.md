# Curved Inference & Geometric Interpretability of LLM Residual Streams

**Source:** Rob Manson (robman.fyi), independent researcher
**Key paper:** "Curved Inference: Concern-Sensitive Geometry in Large Language Model Residual Streams" (arXiv:2507.21107, July 2025; also published in *Artificial Intelligence and Applications*, 2026)
**Code:** github.com/robman (curved-inference benchmarks)
**Related:** FRESH model (Functionalist & Representationalist Emergent Self Hypothesis) — broader cognitive framework

---

## Core Technical Contribution

### The Pullback Semantic Metric

The central insight: measure geometry in the residual stream using a metric derived from the model's own output semantics, not raw Euclidean distance.

**Pullback metric:** G = U⊤U, where U is the unembedding matrix.

This induces an inner product ⟨u,v⟩_G = u⊤Gv in residual space ℝ^d where:
- Distances correspond to shifts in output token probabilities
- Directions correspond to latent semantic operations
- The geometry is invariant to coordinate rotation

All measurements (curvature, salience, norms) are computed under this metric, ensuring they reflect token-aligned semantic geometry rather than coordinate artifacts.

### Residual Stream as Geometric Trajectory

The residual stream x^(ℓ+1) = x^(ℓ) + Attention(x^(ℓ)) + MLP(x^(ℓ)) forms a trajectory through ℝ^d. Attention and MLP outputs are delta vectors — they cause curvature. The residual stream is the curve.

**Key finding:** Only the residual stream shows coherent geometric signal. Attention outputs and MLP outputs individually do not — they are forces, not trajectories.

### Two Primary Metrics

1. **Curvature κ_i** (second-order): How sharply the trajectory reorients between layers. Computed via discrete 3-point central differences under the pullback metric G:

   κ_i = √(‖a_i‖²_G · ‖v_i‖²_G − ⟨a_i, v_i⟩²_G) / ‖v_i‖³_G

   where v_i = velocity (first derivative), a_i = acceleration (second derivative), both estimated via non-uniform central differences to handle unequal step sizes.

2. **Salience S(t)** (first-order): How far the representation moves between layers: S(t) = ‖x_{t+1} − x_t‖_G. Captures semantic velocity/effort.

### Two-Layer Geometric View of LLMs

- **Latent geometry**: encoded in E (embedding) and U (unembedding) — static conceptual structure from pretraining
- **Contextual geometry**: expressed in residual stream x — dynamic meaning during inference

Curved Inference measures how latent potential is bent into contextual meaning.

---

## Experimental Results

**Models:** Gemma3-1b (26 layers, d=2048), LLaMA3.2-3b (28 layers, d=3072)
**Prompts:** 20 matched sets across 7 semantic domains (emotional, moral, perspective, logical, identity, environmental, nonsense), each with neutral + 4 concern-shifted variants

### Key Findings

1. **Curvature exists by default** — even neutral prompts produce structured curvature in the residual stream
2. **Concern-shifted tokens initiate curvature changes** — Δκ heatmaps show minimal delta until the concern token, then colored cells spread across layers
3. **Concern strength controls curvature scale** — stronger concern → proportionally harder bending at the same token-layer positions (semantically grounded, not generic noise)
4. **Curvature persists with turbulence** — effects ripple forward, fluctuating (spike → fade → re-emerge), suggesting layered integration
5. **Salience and curvature are anticorrelated** (r = −0.89 in LLaMA) — representational trade-off: models prioritize either distance or reorientation, not both ("representational budget")
6. **Architecture-dependent expression** — LLaMA: early, strong, sustained curvature (layer ~3, persists through mid-depth); Gemma: weak, shallow, localized (layers 5-6)

### Statistical Significance

LLaMA shows statistically significant scaling: curvature p=0.006, salience p=0.016 for moderate→strong positive concern shifts. Gemma shows the effect but without significance.

### Critical Methodological Point

**UMAP destroys curvature geometry.** Earlier versions using UMAP projections produced median angle correlations of 0.47 (Gemma) and −0.25 (LLaMA) — far below fidelity threshold. This led to mischaracterizing LLaMA's curvature as "late and steep" when it's actually early and sustained. **Native-space analysis is essential.**

---

## Connection to Our Work

### Direct Relevance: Pullback Metric for Quantization Geometry

The pullback metric G = U⊤U is directly applicable to measuring quantization error in a semantically meaningful way. Instead of measuring ‖W − W_q‖_F (Frobenius, coordinate-dependent), we could measure quantization distortion under a task-relevant pullback metric:

- For intermediate layers: pullback from the next layer's weight matrix (or from the final output)
- For the full network: pullback from the unembedding/classification head

This would give **semantically-weighted quantization error** — distortions in directions that matter for output get penalized more than distortions in semantically irrelevant directions. This is essentially what our topk+lstsq correction already does implicitly (it corrects in the directions that matter for the next layer), but the pullback metric formalizes it.

### Curvature as Compression Signal

The curvature/salience framework suggests:
- **Low-curvature layers** = trajectory is straight = layer is doing linear-ish work = highly compressible
- **High-curvature layers** = trajectory bends sharply = nonlinear reorientation happening = compression is riskier
- **Salience-curvature anticorrelation** = there's a budget — could we redistribute quantization precision based on this budget?

### The "Semantic Lens" Model

Attention and MLP as lenses that bend trajectories is a useful frame for understanding why quantization error propagates differently through different layer types:
- Attention (relational lens): redistributes information based on relative relevance — quantization error gets amplified or suppressed depending on attention patterns
- MLP (nonlinear lens): sharpens/redirects — the GELU nonlinear zone analysis we did (bump-dominated vs compression-important channels) is directly measuring properties of this lens

### Residual Stream = Error Propagation Path

Our cascade analysis (cycle 104: attention-driven 70%+ in 8/11 transitions) is measuring the same thing from the quantization side: how errors propagate through the residual trajectory. Curved Inference provides the semantic metric to evaluate which of those errors actually matter.

---

## Limitations & Open Questions

- Only tested on small models (1-3B) with 26-28 layers
- 20 hand-curated prompts — small dataset
- No causal validation (curvature correlates with concern but causality not established)
- No null baselines (scrambled/synonym-swapped prompts)
- Torsion, energy, intrinsic curvature unexplored
- 3-point finite differences may be noisy for long sequences
- Author acknowledges these and has clear future work plan

## Other Work from This Research Program

- **FRESH model**: Consciousness as traversal through a constrained manifold in role-space. Theoretical framework connecting to the empirical work in Curved Inference. More philosophical than technical but provides the motivation.
- **PRISM**: Experimental platform for geometric analysis of LLM representations (details limited)
- **Inference Geometry: The External Curvature of Meaning** (academia.edu) — appears to be a companion/precursor paper
- Blog posts applying the framework to Apple's "Illusion of Thinking" paper, Anthropic's interpretability work
