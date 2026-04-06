# Cycle 216 — W_down Column Geometry Census (Item 230)
**Verdict: REFUTED**

## Summary

GPT-2 small c_proj.weight in each of the 12 blocks has shape (3072, 768). We treat the 3072 rows as column vectors of W_down in R^768 and measure pairwise cosine structure, near-anti-podal pair counts at thresholds {-0.95, -0.9, -0.85, -0.8}, top-20 singular values, and KMeans k=32 cluster quality. We also compute principal angles between block l's W_down right-singular (writing) subspace and block l+1's c_fc left-singular (reading) subspace. A random baseline samples Gaussian unit vectors in R^768 with seeds [42, 137, 2024].

**Blocks 1-5 anti-podal fraction at cos<-0.9:** min=0.00%, max=0.26%. **Random baseline (cos<-0.9):** count_mean=0.00 (fraction=0.0000%), count_std=0.00. **Random baseline (cos<-0.8):** count_mean=0.00 (fraction=0.0000%).

**Cross-block alignment headline:** across the 11 consecutive pairs, mean of top-5 principal-angle cosines = 0.649, mean of all 20 cosines = 0.336. (Cosine=1 means the subspaces are identical; 0 means orthogonal.)

## Per-block table

| block | %twin@-0.9 | %twin@-0.8 | within_cos | between_cos | rank_95 | xblock_top5 |
|------:|----------:|----------:|-----------:|------------:|--------:|------------:|
| 0 |   1.17% |   3.32% | +0.047 | -0.005 |  682 | 0.665 |
| 1 |   0.26% |   0.91% | +0.059 | -0.018 |  686 | 0.650 |
| 2 |   0.13% |   0.65% | +0.047 | -0.020 |  684 | 0.704 |
| 3 |   0.07% |   0.26% | +0.060 | -0.013 |  682 | 0.694 |
| 4 |   0.00% |   0.85% | +0.068 | -0.006 |  687 | 0.778 |
| 5 |   0.00% |   0.20% | +0.055 | -0.009 |  685 | 0.755 |
| 6 |   0.13% |   0.39% | +0.048 | -0.010 |  684 | 0.720 |
| 7 |   0.00% |   0.26% | +0.037 | -0.016 |  683 | 0.710 |
| 8 |   0.07% |   0.20% | +0.038 | -0.013 |  682 | 0.651 |
| 9 |   0.00% |   0.65% | +0.032 | -0.023 |  681 | 0.489 |
| 10 |   0.07% |   0.78% | +0.031 | +0.019 |  684 | 0.327 |
| 11 |   0.13% |   1.63% | +0.049 | +0.004 |  690 | — |

## Per-block one-sentence claims

- Block 0: 1.2% of columns have an anti-podal twin (cos<-0.9); 682 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.05/-0.00.
- Block 1: 0.3% of columns have an anti-podal twin (cos<-0.9); 686 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.06/-0.02.
- Block 2: 0.1% of columns have an anti-podal twin (cos<-0.9); 684 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.05/-0.02.
- Block 3: 0.1% of columns have an anti-podal twin (cos<-0.9); 682 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.06/-0.01.
- Block 4: 0.0% of columns have an anti-podal twin (cos<-0.9); 687 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.07/-0.01.
- Block 5: 0.0% of columns have an anti-podal twin (cos<-0.9); 685 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.06/-0.01.
- Block 6: 0.1% of columns have an anti-podal twin (cos<-0.9); 684 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.05/-0.01.
- Block 7: 0.0% of columns have an anti-podal twin (cos<-0.9); 683 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.04/-0.02.
- Block 8: 0.1% of columns have an anti-podal twin (cos<-0.9); 682 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.04/-0.01.
- Block 9: 0.0% of columns have an anti-podal twin (cos<-0.9); 681 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.03/-0.02.
- Block 10: 0.1% of columns have an anti-podal twin (cos<-0.9); 684 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.03/0.02.
- Block 11: 0.1% of columns have an anti-podal twin (cos<-0.9); 690 singular values capture 95% of W_down energy; KMeans k=32 within/between = 0.05/0.00.

## Test claim and verdict

> "GPT-2 W_down contains enough anti-podal column pairs that GELU's bump effectively doubles channel capacity via bidirectional pushes."

**Verdict: REFUTED**. Confirmed required (a) ≥20% anti-podal fraction at cos<-0.9 in every block in {1..5}, (b) within-cluster cos>0.5 in every block in {1..5}, and (c) between-centroid cos<0.2 in every block in {1..5}. Refuted required <5% anti-podal fraction at cos<-0.9 in ALL 12 blocks AND within-cluster cos<0.2 in ALL 12 blocks.
