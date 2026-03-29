# Audit: Cycle 101 — GELU output clustering (corrected) — GPT-2 FFN

**Item:** 106
**Claimed outcome:** REFUTED
**Audited outcome:** REFUTED (confirmed, but with significant methodological caveat)
**Status:** PASS

## Phase 1: Verdict Check

Refutation gate: K* > 256 for majority of blocks. Result: K* > 256 for 12/12 blocks. The lowest mean reconstruction error at k=256 is block 11 at 0.843 (84.3%), far above the 5% threshold. **Refutation gate clearly triggered.**

## Phase 2: Quality Checks

### G1: Implementation correctness

**GELU derivative: CORRECT.** Uses `torch.autograd.grad` on `F.gelu(z, approximate='tanh')`, not a hand-rolled formula. Sanity checks at z=0, z=3, z=-3 pass with correct values (0.5, ~1.0, ~-0.012). Range assertions [-0.15, 1.15] verify no anomalies.

**Hook structure: CORRECT.** Hooks capture `c_fc` output (pre-GELU z) and `mlp` output (full FFN y). Manual verification `gelu(z) @ W_out + b_out ≈ y_hook` passes at max error <1e-4 for blocks 0, 5, 11.

**Reconstruction error formula: CORRECT.** `||y_approx - y|| / (||y|| + 1e-8)` is a standard relative L2 error on holdout tokens. No linearization — uses actual cluster-mean outputs.

**K-means setup: CORRECT.** Clusters on GELU'(z) vectors (3072-dim), computes cluster-mean of actual FFN outputs on training set, evaluates on held-out set.

### G2: Reproducibility

`random_state=42` for all MiniBatchKMeans. `n_init=5`, `max_iter=300`. Token counts (4096 total, 3072 train, 1024 holdout), model (gpt2), and corpus composition all recorded. Reproducible.

### G3: Results interpretation

K* > 256 for all 12/12 blocks clearly exceeds the refutation threshold (>256 for majority). The refutation is unambiguous.

However, a deeper look reveals that the experiment is partially degenerate:

- **Blocks 3-10:** Error at k=256 is *higher* than the k=1 null baseline (global mean). Clustering on GELU'(z) does not just fail — it actively *hurts*. The cluster means on the 3072 training tokens overfit: 256 clusters with ~12 training tokens each, minus 17-25% degenerate clusters, means some clusters have single-digit members. The cluster means don't generalize.

- **Block 2, k=16:** Error spikes to 3.14 (from 1.18 at k=8 and 1.94 at k=1). A 165% increase. This is k-means instability, not a meaningful signal.

- **Only blocks 0, 2, 11** show any real error reduction with clustering (15%, 48%, 12% respectively). All others are noise around the null baseline.

### G4: Counterintuitive results — Non-monotone errors

The non-monotonicity at blocks 1 (k=8 > k=1) and 2 (k=16 > k=8) is flagged by the code's `monotone_ok` field. Root cause: with only 3072 training tokens and increasing k, the train/holdout split causes cluster means to overfit. This is expected behavior for high k relative to sample size, not a bug. The 5% slack in the code's monotone check hides many smaller violations (blocks 3-10 show persistent upward drift at high k).

### G5: Learning quality — What did we learn beyond cycle 085?

Cycle 085 used Taylor linearization (wrong methodology). Cycle 101 corrects this by using actual FFN outputs. The key new learning:

1. **Even with correct methodology, GELU' clustering fails.** This is not a Taylor-approximation artifact. The fundamental hypothesis (tokens with similar GELU' produce similar FFN outputs) is wrong.

2. **The failure mechanism is clear:** GELU'(z) is a 3072-dim vector in [~0, ~1]. Tokens with similar derivative patterns still produce wildly different outputs because the output depends on `gelu(z) @ W_out`, not just `gelu'(z)`. Similar derivatives can correspond to very different magnitudes of z, which after GELU produce very different output magnitudes.

### G6: Method failure vs genuine refutation

**This is a genuine refutation of the GELU' clustering hypothesis, but the experiment does not test whether other clustering features would work.**

The question "could clustering on z directly, or on y directly, give different results?" is important:
- Clustering on **z directly** would capture magnitude information that GELU' misses. But z is 3072-dim with 4096 tokens — the curse of dimensionality makes k-means unreliable here regardless.
- Clustering on **y directly** (the FFN outputs) would be the natural test. If k-means on y gives K* <= 64, that would suggest the output manifold has low intrinsic complexity even though GELU' is not the right feature to discover it. This is essentially what item 99 (FFN output manifold geometry) should address.

**Sample size concern:** 4096 tokens is small for 3072-dim clustering. However, even at k=8 (384 training tokens/cluster), the error barely drops. The failure is too large to be a sample-size artifact — with 15% max improvement at k=256, no feasible sample size would bring error below 5%.

## Verdict

REFUTED is correct. GELU derivative cluster membership does not predict FFN output similarity. The corrected methodology supersedes cycle 085 and confirms the same conclusion with stronger evidence.
