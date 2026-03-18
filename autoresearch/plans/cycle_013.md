# Cycle 013 Plan: RTG Spectral Coarsening (Item 24)

## Task
Build a Hamming-1 adjacency graph (RTG) of observed activation patterns on a 2->32->32->1 spirals MLP, compute the graph Laplacian's Fiedler vector, run K-way spectral clustering (K=2,4,8,16), merge patterns per cluster via support-weighted affine map averaging, and compare accuracy to a random-merging baseline.

## Hypothesis
RTG spectral structure carries a merging signal that k-means on raw affine maps missed (cycle 012). Hamming-1 adjacency encodes which regions are geometric neighbors; spectral clustering should group functionally similar regions.

## Pre-flight Results
- **P1 (duplicate):** PASS. No past experiment uses RTG adjacency or spectral clustering. Cycle 012 used k-means on Euclidean affine-map vectors -- a different method.
- **P2 (dependency):** PASS. Item 24 depends on nothing (agenda: "Depends on: none").
- **P3 (concrete):** PASS. Single experiment, clear metrics, one script.
- **P4 (program change):** PASS. No modifications to `program.md` or prompts needed.

## Criteria (verbatim from agenda)
> Confirmed if: RTG spectral coarsening achieves >= 3x region reduction with < 5% accuracy drop AND outperforms random region merging by >= 5 percentage points in accuracy at the same K.
> Refuted if: spectral coarsening does not outperform random merging, suggesting the RTG eigenvectors carry no useful merging signal.
> Inconclusive if: spectral coarsening outperforms random on spirals but the benefit disappears for the 100D embedded version, indicating the advantage is geometry-dependent.

## Steps
1. **Train network.** Reuse cycle 012 setup: 2->32->32->1 spirals (n=2000, noise=0, 3 turns, scale=2.0, seed=42), Adam lr=0.01, 3000 epochs, early stop at 99% train acc. Verify test acc >= 90%.
2. **Extract patterns.** On test set (1000 samples): `_forward_hidden` -> `_pack_activation_codes` -> unique patterns. Record per-pattern support sizes and affine maps via `_effective_affine_from_mask`.
3. **Build RTG.** Each pattern is a 64-bit binary vector (32+32 neurons). For all pairs, check Hamming distance == 1. Use `scipy.spatial.distance.cdist(patterns, patterns, 'hamming') * 64 == 1` for efficiency. Store as sparse adjacency matrix.
4. **Spectral clustering.** Compute degree matrix D, Laplacian L = D - A (unnormalized). Use `scipy.sparse.linalg.eigsh(L, k=max_K, which='SM')` for smallest eigenvalues. For each K in {2, 4, 8, 16}: take first K eigenvectors, run sklearn KMeans on the embedded coordinates.
5. **Merge & evaluate (spectral).** For each K: assign each pattern to its cluster, compute cluster-centroid affine map as support-weighted average of member affine maps (weight = number of test samples in that pattern). Evaluate test accuracy using merged maps.
6. **Random baseline.** For each K: randomly assign N_patterns into K groups (5 random seeds), merge via same weighted-average procedure, report mean +/- std accuracy.
7. **Verdict.** Compare spectral vs random at each K. Check 3x reduction (K <= N_patterns/3) with <5% accuracy drop, and >=5pp advantage over random.

## Validation
- Sanity: K=N_patterns affine accuracy must match float accuracy (replicates cycle 012 check).
- RTG stats: report N_patterns, number of edges, graph connectivity (number of connected components), mean degree.
- If RTG is mostly disconnected (>80% isolated nodes), note this as a structural limitation.

## Artifacts
- `experiments/rtg_spectral_toy.py` -- self-contained experiment script
- `autoresearch/results/cycle_013_rtg_spectral.json` -- per-K accuracy for spectral and random
- `autoresearch/results/cycle_013_rtg_spectral.png` -- accuracy vs K curve (spectral vs random)

## Fallbacks
- If RTG is too sparse for meaningful spectral clustering (most patterns isolated), try Hamming-2 adjacency as a relaxation and note this deviation.
- If scipy eigsh fails on the Laplacian (e.g., singular), use normalized Laplacian (L_sym = D^{-1/2} L D^{-1/2}) instead.
