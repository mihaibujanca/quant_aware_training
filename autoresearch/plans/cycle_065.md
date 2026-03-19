# Cycle 065 Plan — Topological Truncation: Betti-Guided Depth Reduction (rev 1)

**Item**: 43 (MOONSHOT) | **Script**: `experiments/topo_truncation_betti.py`

## Pre-flight
- P1: Betti numbers stabilize at layer T → post-T layers replaceable by affine map.
- P2: Refuted if Betti never stabilizes or affine replacement loses >30%.
- P3: Cycle 047 REFUTED monotonic decrease (L4 spike). This tests stabilization (weaker).
- P4: Toy only (spirals + moons). CPU, ~5 min. No GPT-2.
- P5: JSON results + 2-panel plot. P6: Single script. P7: CPU-only, <2 min.

## Design

### Data & Architecture
- Spirals: n=2000, 3 turns, scale=2.0, noise=0.05, seeds={42,123,7}
- Moons: n=2000, noise=0.1, seeds={42,123,7}
- Network: 2→32×4→1. Adam lr=1e-3, 2000 epochs, BCE. Verify ≥90% before proceeding.

### Step 1: Betti Per Layer
- Hook post-ReLU activations L1–L4 (32D). Subsample 500 points (stratified).
- Ripser: maxdim=1, thresh=diameter of point cloud (full, not /2).
- Record β₀, β₁ diagrams and total persistence TP_k = Σ(death−birth) per dim k.

### Step 2: Stabilization via Total Persistence
- Criterion (threshold-free): layer i is stable if for all j ≥ i:
  |TP_k(j) − TP_k(i)| / TP_k(i) < 0.10 for k ∈ {0,1}.
- T = earliest such i. If no T: REFUTED. Report per-seed.

### Step 3: Affine Replacement
- For each T ∈ {1,2,3}: fit W_aff,b_aff = lstsq(A_T, output_float).
- Truncated preds: sigmoid(A_T @ W_aff + b_aff). Report accuracy vs float.

### Step 4: Multi-Seed Validation
- Run Steps 1–3 on all 3 seeds for BOTH datasets (6 runs total).
- T must agree across ≥2/3 seeds per dataset for confirmation.

## Success / Refutation
- **Confirmed**: T ≤ L−2 AND affine loss ≤ 5% on BOTH datasets, ≥2/3 seeds.
- **Refuted**: Betti never stabilizes OR affine loss >30%.
- **Inconclusive**: stabilizes but affine loss 5–30%.

## Outputs
- `autoresearch/results/cycle_065_topo_truncation.json`
- `autoresearch/results/cycle_065_topo_truncation.png`
