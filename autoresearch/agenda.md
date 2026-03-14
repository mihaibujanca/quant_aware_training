# Research Agenda — Geometric Compression

## NEXT EXPERIMENTS

### P0


3. **Linear region census on trained networks**
   - For trained MLPs on spirals and moons (2D), count distinct linear regions activated by the data. Method: forward pass each data point, record binary ReLU activation pattern (on/off per unit per layer), count unique patterns. Also compute theoretical maximum (product of 2^width per layer). Compare.
   - Use existing models in `aleph/models.py`. Implement census in `experiments/region_census.py`.
   - Type: empirical
   - Confirmed if: active regions < theoretical_max / 10 (≥10x gap), demonstrating compression opportunity
   - Refuted if: active regions > theoretical_max / 2 (network uses most patterns)
   - Inconclusive if: gap exists but active regions still number in thousands (merging opportunity unclear)

### P1

4. **Partition extraction and function reconstruction**
   - For a small trained 2D MLP (moons, architecture 2→16→16→1), extract the complete piecewise linear function: for each unique activation pattern found in item 3, compute the corresponding affine map (W_eff, b_eff) by composing masked weight matrices. Reconstruct the function from these (pattern, affine_map) pairs. Compare to original on held-out data.
   - Depends on: item 3 (needs activation pattern enumeration)
   - Type: implementation
   - Confirmed if: reconstruction matches original to <1% relative error on held-out points
   - Refuted if: extraction is numerically unstable or reconstruction error >10%
   - Inconclusive if: works for the small network but unclear how to scale

5. **Affine region similarity and merging potential**
   - Given activation patterns and their affine maps from item 4, measure pairwise similarity between adjacent regions (cosine similarity of effective weight vectors, bias difference). Quantify: what fraction of regions could be merged (similarity > 0.95) with bounded error? Actually merge them and measure accuracy impact.
   - Depends on: item 4
   - Type: empirical
   - Confirmed if: ≥30% of regions mergeable with <5% accuracy degradation on held-out data
   - Refuted if: merging any 10% of regions causes >10% accuracy loss
   - Inconclusive if: mergeability is highly architecture/task-dependent

6. **Layer-wise effective rank on data distribution**
   - For each layer of trained MLPs (spirals depth=12, moons depth=4), compute effective rank of weight matrices restricted to the data distribution. Method: forward pass, collect pre-activation vectors at each layer, compute SVD, measure rank at 95% and 99% energy thresholds. Low effective rank = layer is compressible.
   - Type: empirical
   - Confirmed if: ≥50% of layers have effective rank < 50% of their nominal width
   - Refuted if: all layers operate near full rank on the data distribution
   - Inconclusive if: rank reduction exists but only in first/last layers (not interior)

### P2

8. **Minimal network construction from known partition**
   - Given the extracted partition from items 4-5 (set of affine maps over convex regions), construct the smallest ReLU MLP that implements the same piecewise linear function. Use the theoretical result that a PWL function with K pieces in R^n can be represented by a network with O(K) neurons. Implement this construction for 2D inputs.
   - Depends on: items 4, 5
   - Type: implementation
   - Confirmed if: constructed network has fewer parameters than original AND matches function to <1% error
   - Refuted if: construction requires same or more parameters than original
   - Inconclusive if: parameter reduction exists but is <20% (marginal)

## PAST EXPERIMENTS

### Cycle 001

1. **Survey: Balestriero's piecewise linear / spline theory of deep learning**
   - Type: literature
   - Outcome: **confirmed** — 6 concrete, implementable techniques identified (≥3 required)
   - Cycle: 001
   - Commit: (see cycle_001_manifest.txt)
   - Artifact: `autoresearch/surveys/balestriero_spline_theory.md`
   - Key finding: Activation pattern census (Technique 1) and affine map extraction (Technique 2) are directly actionable from MASO theory; SplineCam (Technique 3) provides exact 2D enumeration. Gap: analysis-to-synthesis route for minimal network construction remains open.

### Cycle 003

7. **Survey: Non-distillation model compression methods (2023-2026)**
   - Type: literature
   - Outcome: **confirmed** — found ≥2 papers with working no-SGD compression methods
   - Cycle: 003
   - Commit: 66021d0
   - Artifact: `autoresearch/surveys/literature_map.md`
   - Key findings:
     - **Serra et al. NeurIPS 2021 (2102.07804)**: ReLU stability criterion — MIP-based dead/always-active neuron detection → 56% connection removal, 183x faster than prior art. No SGD required.
     - **Berzins 2023 (2306.07212)**: GPU-accelerated partition extraction + §5.4 pruning demo → 78% parameter reduction (2,50,50,1 → 2,25,19,1) via dead + always-active removal. Constructive, no SGD.
     - **3-Phase algorithm** identified: Phase 1 (dead neuron removal) + Phase 2 (always-active collapse to ≤D neurons/layer) = solvable analytically. Phase 3 (merging genuinely unstable neurons) = open problem.
     - Depth reduction ruled out: Arora et al. (1611.01491) proves exponential neuron cost for depth reduction → target width compression only.

### Cycle 002

2. **Survey: Tropical geometry and neural networks**
   - Type: literature
   - Outcome: **inconclusive** — tropical polynomial form is exponentially large (intractable for deep nets); practical TropNNC algorithm uses zonotope clustering but never builds a compact tropical form
   - Cycle: 002
   - Commit: TBD
   - Artifact: `autoresearch/results/cycle_002_summary.md`
   - Key finding: Tropical geometry = dead end for direct compression. One salvageable idea: neurons-as-zonotope-generators motivates clustering-based pruning (overlaps with MASO region merging). Do NOT pursue tropical algebra further.
