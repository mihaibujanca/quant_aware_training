# Task
Item 99: FFN output manifold geometry -- GPT-2 FFN. For blocks [7,5,4,1], measure hidden activation eigenspectrum and test whether Johnson-Lindenstrauss explains why random+lstsq matches SVD+lstsq.

# Hypothesis
- Confirmed if: isotropy ratio > 0.1 for >=3/4 blocks (near-spherical, JL explains lstsq dominance).
- Refuted if: isotropy ratio < 0.01 (steep spectrum) but lstsq still works for random bases (something beyond JL).
- Inconclusive if: moderate isotropy (0.01-0.1).

# Pre-flight Results
- P1: Not duplicate. Cycles 071/077 measured reconstruction error for random vs SVD bases but never examined the spectrum that explains *why* they match.
- P2: No dependencies. Item 99 says "Depends on: none".
- P3: Concrete -- single script, 4 blocks, deterministic eigendecomposition + 10 random projections for JL stats.
- P4: No program.md changes needed.
- P5 (null baseline): Under Marchenko-Pastur (iid Gaussian n=16384, d=3072), isotropy ratio ~ n/d = 5.3 and spectrum is flat. If real spectrum matches MP, hidden activations are structureless noise -- isotropy is trivial. The interesting case is spectrum steeper than MP but isotropy ratio still > 0.1 (structured but spread).
- P6 (methods): Eigendecomposition of sample covariance (3072x3072, symmetric) is exact. JL distortion measured as max_pairs |d_proj/d_orig - 1| over 10K random token pairs. 10 random projection draws give variance estimate.
- P7 (power): 16384 tokens >> 3072 dims (gamma=0.19) ensures well-conditioned sample covariance with moderate MP distortion. 10K pairs x 10 draws = 100K distortion samples per block. Adequate for distinguishing JL-predicted epsilon from tighter distortion.

# Steps
Script: `experiments/ffn_manifold_geometry_gpt2.py`

1. **Setup.** Load GPT-2 (float32), OWT corpus, seq_len=128, 128 sequences = 16384 tokens. Blocks = [7,5,4,1]. k=512 (6x compression from 3072).

2. **Collect activations.** For each block, hook (a) after ln_2 to get x_ln (16384x768) — the input to W_up before any FFN computation, and (b) after GELU to get h (16384x3072). Also hook after W_down to get y (16384x768). Center h and x_ln (subtract mean).

3. **LayerNorm isotropy diagnostic.** For each block, compute sample covariance of x_ln (768x768). Eigendecompose. Record all 768 eigenvalues. Compute isotropy ratio = lambda_512/lambda_1 for x_ln. This distinguishes inherited isotropy (from LayerNorm normalization propagated through W_up) from learned structure in the FFN. If x_ln is already isotropic in 768D, measured h-isotropy may be an artifact of LayerNorm, not a property of FFN computation.

4. **Hidden activation eigenspectrum.** Compute sample covariance C = h^T h / (n-1). Eigendecompose (torch.linalg.eigh). Record all 3072 eigenvalues. Compute isotropy ratio = lambda_512 / lambda_1. Plot normalized spectrum + MP density overlay.

5. **Marchenko-Pastur comparison.** For gamma = d/n = 3072/16384 = 0.19, MP support = [(1-sqrt(gamma))^2, (1+sqrt(gamma))^2] * sigma^2 (sigma^2 = mean eigenvalue). Compute KS statistic between empirical spectrum and MP.

6. **Projection condition numbers.** For each block, build 3 projection matrices P (3072 -> 512): (a) top-512 eigenvectors, (b) top-512 right singular vectors of h, (c) Gaussian random (10 draws). Compute lstsq condition number = cond(P @ h^T) for each. Report mean +/- std for random.

7. **JL distortion.** Sample 10K random token pairs. For each projection type, compute relative distance distortion |d_proj/d_orig - 1|. Report max, 95th percentile, mean. Compare to JL bound epsilon = sqrt(8 * log(16384) / 512) ~ 0.39.

8. **Save results.** JSON: `autoresearch/results/cycle_086_manifold_geometry.json`. Plots: `autoresearch/results/cycle_086_manifold_geometry.png` (5-panel: 4 blocks with h-spectrum + MP overlay + isotropy ratio annotation, plus 1 panel comparing x_ln vs h isotropy ratios across blocks).

9. **Bookkeeping.** Move item 99 from NEXT to PAST in agenda.md with verdict and artifact paths.

# Validation
- Sample covariance must be PSD (all eigenvalues >= 0). Negative eigenvalues indicate numerical error.
- Isotropy ratio must be in [0, 1]. Values > 1 indicate indexing bug (lambda_k > lambda_1).
- Random projection condition numbers should be close to SVD condition numbers if spectrum is flat.
- JL distortion should be <= epsilon_JL for all pairs if JL bound holds. If max distortion > 2 * epsilon_JL, check projection normalization.
- x_ln isotropy ratio should be compared to h isotropy ratio: if x_ln already isotropic and h also isotropic, inherited structure is the more parsimonious explanation.

# Artifacts
- `experiments/ffn_manifold_geometry_gpt2.py`
- `autoresearch/results/cycle_086_manifold_geometry.json`
- `autoresearch/results/cycle_086_manifold_geometry.png`

# Fallbacks
- If 16384 tokens cause OOM -> reduce to 8192 (gamma=0.37, still much better than original 0.75).
- If eigendecomposition OOM on 3072x3072 -> use randomized SVD (torch.svd_lowrank, rank=768) and report only top-768 spectrum.
- If JL distortion wildly exceeds bound -> verify projection columns are orthonormalized (QR after Gaussian draw).
