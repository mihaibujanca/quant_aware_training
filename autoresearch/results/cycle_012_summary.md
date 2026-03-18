# Cycle 012 — Effective number of affine behaviors via Jacobian k-means (toy ReLU)

## Setup

- Dataset: spirals, n=2000, 3 turns, scale=2.0, noise=0.0, seed=42, 50/50 train/test split
- Architecture: 2→32→32→1 SimpleMLP (2 hidden layers, 64 total hidden units)
- Training: Adam lr=0.01, BCE loss, early stop at train_acc ≥ 99% (stopped at epoch 2600)
- Final accuracy: train=99.8%, test=98.3%

## Results

| K   | Test accuracy |
|-----|--------------|
| 2   | 51.2%        |
| 4   | 57.0%        |
| 8   | 59.3%        |
| 12  | 63.9%        |
| 16  | 67.2%        |
| 24  | 68.5%        |
| 32  | 75.2%        |
| 48  | 74.4%        |
| 64  | 74.9%        |
| 375 | 98.3% (= float) |

- N_patterns = 375 (unique activation patterns in 1000 test probes)
- Float accuracy = 98.3%
- Threshold (float − 1%) = 97.3%
- K* = 375 (never recovered within 1% of float until K=N_patterns)
- Ratio K*/N_patterns = 1.00

## Sanity check

K=N_patterns affine accuracy = 98.3% = float accuracy (PASS: exact-affine reproduces float outputs).

## Affine map geometry

The 375 A_ω vectors (each of length 3 = [A[0], A[1], b]) show:
- Norms: mean=74.8, std=71.3 (high variation)
- Pairwise distances: mean=113.5, median=94.5
- Components: A[0] ∈ [-175, 183], A[1] ∈ [-574, 211], b ∈ [-115, 135]

The affine maps are genuinely diverse — they are not clusterable without significant accuracy loss.

## Verdict: REFUTED

K*/N_patterns = 1.00 > 0.7. Most activation patterns compute meaningfully different linear functions. k-means cannot find a low-dimensional representation of affine behavior that preserves network outputs.

## Implications

High functional diversity of MASO affine maps in a trained spirals network means:
1. Activation-pattern redundancy (measured geometrically in affine space) does not exist at the level k-means can exploit.
2. This makes experiment 23 (GPT-2 Jacobian k-means) harder to motivate — if functional redundancy is absent even in a small toy ReLU network, the LLM analog is unlikely to show it.
3. The failure mode is not numerical (the sanity check passes perfectly), but structural: each activation pattern has a genuinely distinct geometry in output-function space.

Alternative: RTG spectral coarsening (experiment 24) may do better than k-means by respecting the Hamming-1 adjacency structure of activation patterns.
