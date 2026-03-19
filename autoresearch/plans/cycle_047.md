# Task
Persistent homology of activation manifold through depth in a trained toy ReLU spirals network (item 27). Measure Betti number evolution layer-by-layer and correlate with single-affine residual r from cycle 010.

# Hypothesis
- **Confirmed if**: total Betti complexity decreases monotonically through at least 80% of layers AND layers with the largest Betti drop have the lowest single-affine residual r from experiment 20.
- **Refuted if**: Betti complexity does not decrease or is the same for trained and randomly-initialized networks.
- **Inconclusive if**: topology simplifies overall but the relationship to compression ease is not consistent.

# Pre-flight Results
- P1 PASS: no persistent homology experiment in PAST EXPERIMENTS.
- P2 PASS: depends on item 20 (cycle 010, CONFIRMED, r values: L1=0.430, L2=0.050, L3=0.033, L4=0.024).
- P3 PASS: concrete — run ripser on 4 layers, compute Betti, correlate with known r values.
- P4 PASS: no prompt/program changes needed.
- P5 (null baseline): trained network compared against (a) random-init network (architectural null) AND (b) isotropic Gaussian point cloud (500 pts, matched dim and scale) to establish "no topological structure" baseline for total persistence lifetime.
- P6 OK: ReLU network, standard PH pipeline. PCA used only for visualization, NOT for ripser input.
- P7 (power): n=4 layers is small for Spearman. Address by reframing second criterion as rank-1 prediction (see below).

# Steps
1. `pip install ripser scikit-tda` in the project venv.
2. Train spirals network: 2->32x4->1, 2000 train, scale=2.0, lr=0.001 (cycle 010/019 convention). Verify test acc >= 95%.
3. Collect activations at L1-L4 on 500 test points. Store full-dimensional (32D) activations.
4. **Run ripser on full 32D activations** (NOT PCA-reduced). The 500x500 distance matrix is the bottleneck, not ambient dimension. Use max filtration radius = 95th percentile of pairwise distances (prevents noise-driven features at large scales). Compute H0 and H1 persistence diagrams at each layer.
5. Compute total Betti complexity per layer: sum of persistence lifetimes (birth-death intervals) across all H0 and H1 features with lifetime > noise threshold (1% of max filtration radius).
6. **Null baselines**: (a) random-init network — same architecture, collect activations, run ripser. (b) Isotropic Gaussian null — 500 points in 32D, scale-matched to mean activation norm at each layer. Report total persistence lifetime for both nulls alongside trained network.
7. Check monotonicity criterion: Betti complexity decreases in >= 80% of consecutive layer pairs (i.e., >= 3 of 3 transitions for 4 layers).
8. **Correlation with r**: reframe as rank-1 prediction — the layer with the LARGEST Betti drop should be the layer with the LOWEST r (from cycle 010). Report Spearman(Betti_drop_rank, r_rank) as supplementary, noting n=4 means p<0.05 requires |rho|=1.0 exactly.
9. Generate 3-panel figure: (a) persistence diagrams at each layer (2x2 grid), (b) total Betti complexity vs depth (trained, random-init, Gaussian null), (c) PCA-2D visualization of activations at each layer (visualization only, NOT used for PH computation).
10. Save results to `autoresearch/results/cycle_047_phom_toy.json` and `autoresearch/results/cycle_047_phom_toy.png`.
11. Update `autoresearch/agenda.md`: move item 27 from NEXT to PAST EXPERIMENTS with commit hash and artifact paths.
12. Commit: `[autoresearch] cycle 047: persistent homology through depth — toy ReLU`

# Validation
- Task accuracy >= 95% before any PH analysis (undertrained model = meaningless topology).
- Trained network total persistence lifetime differs from BOTH null baselines (random-init AND Gaussian) by > 2x at input or output layer — confirms PH is measuring real structure, not artifacts.
- Persistence diagrams have finite features (ripser converged, no NaN/inf).
- r values used match cycle 010 exactly: L1=0.430, L2=0.050, L3=0.033, L4=0.024.

# Artifacts
- `autoresearch/results/cycle_047_phom_toy.json` — per-layer Betti complexity, persistence lifetimes, null baseline values, rank-1 prediction result
- `autoresearch/results/cycle_047_phom_toy.png` — 3-panel figure
- Experiment script (inline or standalone)

# Fallbacks
- If `ripser` pip install fails: use `gudhi` (alternative PH library, `pip install gudhi`). API differs but computes same Betti numbers.
- If 500-point distance matrix is too slow (>5 min): subsample to 200 points, note reduced statistical power.
- If all Betti complexity values are near-zero (trivial topology at all layers): report as "refuted — no topological structure detected" and check whether the 32D embedding is too sparse for PH at this sample size.
