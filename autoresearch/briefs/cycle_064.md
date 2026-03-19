# Cycle 064 — Item 42: Topology-Preserving Distillation (Toy ReLU)

**Status**: Eligible (no dependencies)
**Type**: Implementation + empirical
**Target Dataset**: Moons / Spirals (2D → 32×12 → 1)

## Objective
Train a small student network using both output loss + topological distillation loss. Test whether topology preservation (measured by persistent homology / persistent diagram distance) improves generalization vs. output-only distillation at matched architecture.

## Success Criteria
- **Confirmed**: Topo-distilled student ≥5pp higher test accuracy than output-only student, AND topology loss converges (PD distance decreases).
- **Refuted**: Topology loss oscillates/diverges OR topo-distilled ≤ output-only student.
- **Inconclusive**: Loss converges but accuracy gain < 3pp (signal real but weak).

## Approach
1. Train teacher (float32) on toy task.
2. Extract teacher persistent diagrams (each layer / full network).
3. Student network with combined loss: λ·L_output + (1−λ)·L_topo.
4. Sweep λ ∈ [0, 1]; measure accuracy and PD convergence.
5. Baseline: output-only (λ=1) and topology-only (λ=0) ablations.

## Key Risk
Topology loss is high-dimensional and noisy; gradient signal may be too weak to optimize effectively. Requires careful loss weighting + smoothing.
