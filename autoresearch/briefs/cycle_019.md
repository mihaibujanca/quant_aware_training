# Cycle 019 Brief — Activation Pattern Entropy Cascade (Toy ReLU)

## Item
37. Activation pattern entropy cascade — toy ReLU

## Hypothesis
If deep layers add little new partition structure (low conditional entropy given earlier layers),
those layers are redundant and safe to compress. The entropy cascade H(L1), H(L2|L1),
H(L3|L1,L2), H(L4|L1,L2,L3) gives a per-layer information budget for compression.

## Method
- Train spirals ReLU MLP: 2→32→32→32→32→1, 4 hidden layers
- Extract binary activation masks on test set (N=500 points)
- Compute marginal entropy H(Li) for each layer
- Compute joint entropy via concatenated binary patterns; derive conditional entropy
  H(Li | L1…L_{i-1}) = H(L1…Li) − H(L1…L_{i-1})
- Compare conditional entropy to single-affine residual r from cycle 010

## Criteria
- Confirmed: ≥1 layer has conditional entropy ≤50% of layer 1 marginal AND that layer has low r
- Refuted: all conditional entropies uniformly high
- Inconclusive: entropy varies but no correlation with compression signals

## Key risk
4×32 = 128-bit concatenated pattern; with N=500 samples, most joint patterns will be unique
→ naïve joint entropy will hit log(N) ceiling. Must use empirical estimate carefully.
