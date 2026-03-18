# Cycle 017 Plan — Region Volume Pareto (Toy ReLU)

## Task
Quantify whether the region-volume distribution of a trained spirals ReLU network has a Pareto structure:
a small fraction of (large) regions should contain most correctly-classified test points.

## Hypothesis
Training concentrates test-set coverage into a small number of large regions. The top 20% of regions
by volume will cover ≥70% of correctly-classified test points.

## Pre-flight
- P1 (not duplicate?): Item 34 has never been run. ✓
- P2 (deps met?): No dependencies. ✓
- P3 (concrete?): Exact architecture, dataset, fallback strategy all specified. ✓
- P4 (no program.md changes?): Pure experiment. ✓

## Steps

1. **Train spirals MLP** — 2→32→32→1, seed=42, n=2000, 3 turns, scale=2.0.
   Same hyperparams as cycle 012: Adam lr=0.001, epochs=2000 (or until ≥97% train acc).

2. **Split data** — 1000 train / 1000 test (same seed split).

3. **Extract activation patterns** — for each test point, record the binary ReLU activation vector
   across all hidden neurons (concatenated across all layers). Each unique binary vector = one region.

4. **Compute A_ω per region** — A_ω = W_L · diag(q^L) · … · W_1 · diag(q^1).
   One A_ω per distinct pattern. Collect all patterns observed on test set.

5. **Volume** — count how many test points fall in each region → volume = count / 1000.

6. **Sort** regions descending by volume.

7. **Cumulative accuracy curve** — iterate: keep regions[0..k]. For each test point:
   - If its pattern is in kept set: classify using A_ω (output = A_ω @ x + b_ω, sign = class).
   - Else: find nearest kept region by Hamming distance on activation vector, use that A_ω.
   Record accuracy at each k. Also record fraction of all test points whose true pattern is kept (coverage).

8. **Gini coefficient** — compute on the volume array.

9. **20% threshold check** — at k = ceil(0.2 × N_regions), what fraction of correctly-classified points
   (under float model) are in the kept regions?

10. **Plot** — 3-panel figure:
    - Panel 1: region volume distribution (log scale, sorted descending).
    - Panel 2: cumulative accuracy vs fraction of regions kept.
    - Panel 3: Lorenz curve of volume distribution (for Gini visualization).

## Validation
- float model accuracy ≥ 97% on test set (else retrain longer).
- Number of distinct patterns on 1000 test points: expected 50–500 (cycle 012 saw ~375 on 200 pts).
- Gini > 0.3 → Pareto structure present; < 0.3 → refuted.
- Top-20% coverage ≥ 70% → confirmed.

## Artifacts
- `experiments/region_volume_pareto.py`
- `autoresearch/results/cycle_017_pareto.json` — metrics dict
- `autoresearch/results/cycle_017_pareto.png` — 3-panel figure
- `autoresearch/results/cycle_017_manifest.txt`

## Fallbacks
- If too many patterns (>5000): subsample to 500 test points.
- If Hamming-distance fallback is slow (many patterns): use most-populated region as universal fallback.
- If accuracy under A_ω classification is poor even with true pattern (A_ω has wrong bias): add bias term
  b_ω = output_L_affine_constant (constant part of the piecewise-affine output map).
