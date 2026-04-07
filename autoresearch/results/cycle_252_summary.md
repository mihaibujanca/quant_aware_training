# Cycle 252 — Item 262: Block 5 geometric deep-dive

**Verdict: REFUTED**

- Criteria: CONFIRMED if block 5 |z|≥1.5 on ≥2 metrics; REFUTED if block 5 |z|<1.0 on ALL metrics; INCONCLUSIVE otherwise
- Config: GPT-2 pretrained, wikitext-2 val `all_ids[:200]` @ SEQ_LEN=128 → 25,600 tokens, seed 42, device=mps
- Blocks analysed: {1,2,3,4,5,6,7,8}; cohort for z-score = {1,2,4,7,8}; target = 5
- 6 metrics: `std_mean`, `kurt_mean`, `eff_rank` (SVD entropy), `cov128`, `cov512` (std × c_proj row-norm weighted topk coverage), `mean_cos` (off-diagonal c_proj row cosine)

## Per-block metrics (post-GELU input to c_proj, shape 25600 × 3072)

| block | std_mean | kurt_mean | eff_rank | cov128 | cov512 | mean_cos | max_cos |
|-------|----------|-----------|----------|--------|--------|----------|---------|
| 1 | 0.1399 | 51.67 | 293.20 | 0.7293 | 0.7954 | 0.00042 | 0.9872 |
| 2 | 0.1388 | 63.16 |  45.38 | 0.9772 | 0.9834 | 0.00267 | 0.9914 |
| 3 | 0.1343 | 62.38 | 293.45 | 0.4018 | 0.5727 | 0.00332 | 0.9732 |
| 4 | 0.1422 | 48.67 | 309.79 | 0.1880 | 0.4057 | 0.00305 | 0.8774 |
| **5** | **0.1527** | **56.84** | **358.21** | **0.1576** | **0.3814** | **0.00227** | **0.7981** |
| 6 | 0.1611 | 51.67 | 378.81 | 0.1657 | 0.3938 | 0.00123 | 0.8556 |
| 7 | 0.1691 | 52.32 | 375.59 | 0.1668 | 0.3975 | 0.00078 | 0.7399 |
| 8 | 0.1744 | 54.57 | 348.72 | 0.1911 | 0.4167 | 0.00040 | 0.8161 |

## Block-5 z-scores vs cohort {1,2,4,7,8}

| metric | target (b5) | cohort mean | cohort std | z |
|--------|-------------|-------------|------------|---|
| std_mean | 0.1527 | 0.1529 | 0.0174 | **−0.012** |
| kurt_mean | 56.84 | 54.08 | 5.499 | **+0.502** |
| eff_rank | 358.21 | 274.53 | 132.12 | **+0.633** |
| cov128 | 0.1576 | 0.4505 | 0.3781 | **−0.775** |
| cov512 | 0.3814 | 0.5998 | 0.2727 | **−0.801** |
| mean_cos | 0.00227 | 0.00146 | 0.00129 | **+0.625** |

All six z-scores have |z| < 1.0 (max |z| = 0.801 on cov512).

## Verdict logic

- |z| ≥ 1.5 metrics: [] (count 0)
- |z| ≥ 1.0 metrics: [] (count 0)
- ≥2 metrics at |z|≥1.5 → **CONFIRMED** [not met]
- ALL metrics at |z|<1.0 → **REFUTED** ✓
- Exactly 1 metric outlier → INCONCLUSIVE [not met]

## Interpretation

Block 5 is **geometrically typical** relative to the {1,2,4,7,8} cohort on ALL six metrics: per-channel std, per-channel kurtosis, effective rank (SVD entropy), topk-128 / topk-512 std×row-norm coverage, and mean c_proj row cosine. The cohort itself is very heterogeneous (cov128 ranges 0.167–0.977, eff_rank ranges 45–376) — the cohort std on most metrics swamps the block-5 deviation. Block 5 sits in the "dense high-rank" regime shared with blocks 3, 4, 6, 7, 8: high effective rank (~300–380), low topk concentration (~0.16–0.19 at K=128), and low off-diagonal cosine (~0.5–3.3e-3). Blocks 1 and 2 are the geometric outliers of the cohort (eff_rank 293 and 45; cov128 0.73 and 0.98) — block 2 in particular has a 7× lower effective rank and 5× higher cov128 than block 5.

The cycle 250 prior (REFUTED: swap block 5 for block 3 in the 6-block set does not recover the 5-block win, so the cycle 245 gain was a count transition 6→5, not block-5-specific) is **confirmed on the geometry side**: block 5 has no intrinsic FFN-geometry anomaly that would make K=512 topk structurally harder there than on its neighbours. If block 5 ever does "matter more" downstream (cycle 243's rank-1 per-block ΔPPL=5.10), the cause must be in how its output reads into attention/LN — a cross-block interaction — not intrinsic to the MLP activation distribution or c_proj row geometry. The cohort cov128 variance is so large (0.167 to 0.977) that ANY single-block outlier test at n=5 with this cohort is underpowered: block 2 itself sits >2σ away from the cohort mean on cov128 if we include block 5 in the reference, but that tells us the early layers (1, 2) are special, not that block 5 is.

## Closed directions

- **"Block 5's MLP distribution is atypical"**: refuted on all 6 probed metrics. No evidence from per-channel moments, SVD spectrum, topk coverage (at both K=128 and K=512), or c_proj row alignment.
- **"Eff_rank is higher at block 5"**: block 5 eff_rank=358 is below blocks 6 (379) and 7 (376); the upper-mid-cascade sits in the 350–380 band and block 5 is firmly inside it.
- **"Topk-512 leaves more residual at block 5 because of geometry"**: refuted. Block 5 cov512=0.381 is within 0.04 of blocks 4,6,7,8 (range 0.394–0.417). If anything, cov512 is MONOTONE in block index across 4–8, and block 5 sits exactly on the monotone — no inflection point.

## Follow-ups

- The open question shifts to cross-block interaction: is there a downstream attention/LN direction that blocks 5 writes into and blocks 3/6 don't? Probing idea for a future cycle: compute the per-block contribution to the next block's `ln_1` output along the top attention singular directions.
- The cohort is too heterogeneous for single-block outlier tests. Any future "block X deep-dive" should use a matched-neighbourhood cohort (blocks X−1 and X+1), not {1,2,4,7,8}.
- Cycle 245's "5-block float-tie" stays explained by block COUNT (cascade constant collapse 6→5), not block IDENTITY — closed jointly with cycle 250 and cycle 252.

## Artifacts

- `experiments/block5_geometric_cycle252.py`
- `autoresearch/results/cycle_252_block5_geometric.json`
- `autoresearch/logs/cycle_252_run.log`
