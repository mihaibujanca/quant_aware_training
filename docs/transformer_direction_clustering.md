# Experiment 1: Direction Clustering in GPT-2 MLP Blocks

## Objective

Test whether the rows of MLP up-projection matrices in a trained GPT-2 model cluster into significantly fewer effective directions than the full 3072, which would validate the "scaffolding hypothesis" — that gradient-trained networks carry representational overhead removable by reparameterization to compact smooth functions (KANs).

## Setup

- **Model:** GPT-2 small (124M params), pretrained from HuggingFace (`gpt2`)
- **Architecture:** 12 layers, d_model=768, d_ff=3072, GeLU activation
- **Target:** MLP up-projection `c_fc.weight` (Conv1D, shape 768×3072 — each column is a direction vector in ℝ^768) and down-projection `c_proj.weight` (shape 3072×768 — each row is an output direction)
- **Baseline:** 3072 random unit vectors in ℝ^768

## Methods

1. **Pairwise cosine similarity** — full 3072×3072 matrix per layer, compared to random baseline
2. **PCA effective dimensionality** — number of principal components for 90/95/99% variance
3. **Hierarchical clustering** — average-linkage on cosine distance, cluster counts at similarity thresholds 0.5–0.9
4. **Joint input+output clustering** — combined distance `α·d_cos(up) + (1-α)·d_cos(down)` with α=0.5
5. **Within-cluster bias analysis** — collective GeLU response for largest clusters in most compressible layer

## Results

### Pairwise Cosine Similarity

| Layer | mean\|cos\| | std | frac>0.3 | frac>0.5 |
|-------|-------------|------|----------|----------|
| 0 | 0.0431 | 0.0352 | 0.05% | 0.01% |
| 1 | 0.0375 | 0.0350 | 0.12% | 0.02% |
| 2 | 0.0382 | 0.0330 | 0.04% | 0.00% |
| 3 | 0.0380 | 0.0311 | 0.03% | 0.00% |
| 4 | 0.0376 | 0.0327 | 0.06% | 0.01% |
| 5 | 0.0386 | 0.0322 | 0.03% | 0.00% |
| 6 | 0.0386 | 0.0318 | 0.03% | 0.00% |
| 7 | 0.0392 | 0.0311 | 0.02% | 0.00% |
| 8 | 0.0385 | 0.0303 | 0.01% | 0.00% |
| 9 | 0.0390 | 0.0304 | 0.01% | 0.00% |
| 10 | 0.0402 | 0.0313 | 0.02% | 0.00% |
| 11 | 0.0428 | 0.0331 | 0.02% | 0.00% |
| **RAND** | **0.0288** | **0.0218** | **0.00%** | **0.00%** |

Direction pairs are near-orthogonal. Mean |cos sim| is 0.038–0.043, only marginally above the random baseline of 0.029.

### PCA Effective Dimensionality

| Layer | dim@90% | dim@95% | dim@99% | compress@95% |
|-------|---------|---------|---------|--------------|
| 0 | 532 | 618 | 719 | 20.1% |
| 1 | 556 | 639 | 728 | 20.8% |
| 2 | 540 | 627 | 726 | 20.4% |
| 3 | 521 | 610 | 718 | 19.9% |
| 4 | 515 | 606 | 718 | 19.7% |
| 5 | 515 | 607 | 718 | 19.8% |
| 6 | 517 | 610 | 720 | 19.9% |
| 7 | 515 | 606 | 714 | 19.7% |
| 8 | 521 | 610 | 717 | 19.9% |
| 9 | 515 | 605 | 713 | 19.7% |
| 10 | 515 | 606 | 715 | 19.7% |
| 11 | 513 | 606 | 717 | 19.7% |
| **RAND** | **585** | **661** | **742** | **21.5%** |

GPT-2 needs ~606 PCs for 95% variance vs. 661 for random — marginally more concentrated but not meaningfully so. Directions span most of ℝ^768.

### Hierarchical Clustering

| Layer | cos>0.5 | cos>0.6 | cos>0.7 | cos>0.8 | cos>0.9 | compress@0.7 |
|-------|---------|---------|---------|---------|---------|--------------|
| 0 | 2952 | 2988 | 3022 | 3053 | 3064 | 98.4% |
| 1 | 2914 | 2968 | 3016 | 3054 | 3071 | 98.2% |
| 2 | 2993 | 3024 | 3045 | 3063 | 3068 | 99.1% |
| 3 | 3023 | 3049 | 3062 | 3067 | 3071 | 99.7% |
| 4 | 2985 | 3018 | 3042 | 3063 | 3072 | 99.0% |
| 5 | 3033 | 3054 | 3064 | 3070 | 3072 | 99.7% |
| 6 | 3027 | 3056 | 3064 | 3070 | 3072 | 99.7% |
| 7 | 3049 | 3062 | 3068 | 3071 | 3072 | 99.9% |
| 8 | 3050 | 3064 | 3068 | 3071 | 3072 | 99.9% |
| 9 | 3049 | 3060 | 3063 | 3069 | 3072 | 99.7% |
| 10 | 3038 | 3051 | 3060 | 3071 | 3072 | 99.6% |
| 11 | 3025 | 3053 | 3063 | 3071 | 3072 | 99.7% |

At cos>0.7 threshold: only 4–56 neurons merge out of 3072. Virtually no clustering.

### Joint Input+Output Clustering

| Layer | input-only@0.7 | joint@0.7 | ratio |
|-------|----------------|-----------|-------|
| 0 | 3022 | 3056 | 1.01× |
| 1 | 3016 | 3061 | 1.01× |
| 2 | 3045 | 3070 | 1.01× |
| 3 | 3062 | 3071 | 1.00× |
| 4 | 3042 | 3071 | 1.01× |
| 5 | 3064 | 3072 | 1.00× |
| 6 | 3064 | 3072 | 1.00× |
| 7 | 3068 | 3072 | 1.00× |
| 8 | 3068 | 3072 | 1.00× |
| 9 | 3063 | 3072 | 1.00× |
| 10 | 3060 | 3072 | 1.00× |
| 11 | 3063 | 3072 | 1.00× |

Accounting for output directions makes clustering even sparser.

## Summary

| Layer | PCA@95% | PCA% | Clusters@0.7 | Compress | Signal |
|-------|---------|------|--------------|----------|--------|
| 0 | 618 | 20.1% | 3022 | 98.4% | WEAK |
| 1 | 639 | 20.8% | 3016 | 98.2% | WEAK |
| 2 | 627 | 20.4% | 3045 | 99.1% | WEAK |
| 3 | 610 | 19.9% | 3062 | 99.7% | WEAK |
| 4 | 606 | 19.7% | 3042 | 99.0% | WEAK |
| 5 | 607 | 19.8% | 3064 | 99.7% | WEAK |
| 6 | 610 | 19.9% | 3064 | 99.7% | WEAK |
| 7 | 606 | 19.7% | 3068 | 99.9% | WEAK |
| 8 | 610 | 19.9% | 3068 | 99.9% | WEAK |
| 9 | 605 | 19.7% | 3063 | 99.7% | WEAK |
| 10 | 606 | 19.7% | 3060 | 99.6% | WEAK |
| 11 | 606 | 19.7% | 3063 | 99.7% | WEAK |

- Best compression (single layer): 98.2% = 1.0× parameter reduction
- Average compression across layers: 99.4% = 1.0× average

## Verdict

**NEGATIVE.** The 3072 up-projection direction vectors in GPT-2 MLP blocks are near-uniformly distributed across ℝ^768. There is no meaningful clustering — the scaffolding hypothesis as formulated (direction redundancy in up-projections) is not supported for this architecture.
