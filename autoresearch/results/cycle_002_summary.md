# Cycle 002: Tropical Geometry and Neural Networks

## Papers Reviewed

1. **"Tropical Geometry of Deep Neural Networks"** — Zhang et al. (2018), arXiv:1805.07091
2. **"Revisiting Tropical Polynomial Division: Theory, Algorithms and Application to Neural Networks"** — Maragos et al. (2023), arXiv:2306.15157
3. **"TropNNC: Structured Neural Network Compression Using Tropical Geometry"** — Maragos et al. (2024), arXiv:2409.03945

---

## Key Results

### Zhang et al. 2018 — Foundational Theory

**Main theorem:** A ReLU network with integer weights is equivalent to a tropical rational map (ratio of two tropical polynomials). Each linear region of the network corresponds to a vertex/face of a polytope in the tropical sense.

**On compactness:** The tropical polynomial has *one monomial per linear region*. Since the number of linear regions grows exponentially with depth/width, the tropical polynomial representation is **not more compact** — it is exponentially larger than the weight matrix representation. This paper is a theoretical equivalence result, not a compression result.

**Expressivity bonus:** Deeper networks are exponentially more expressive than shallow ones — a result derivable from the tropical representation but not useful for compression.

### Maragos 2023 — Tropical Polynomial Division

**Goal:** Simplify a network polynomial p(x) by dividing it by a "simpler" divisor d(x), obtaining a quotient q(x) such that p ≈ d · q.

**Exact algorithm:** Polyhedral geometry (epigraph intersections, convex hulls). Complexity is **exponential** in input dimension. Only usable for tiny examples.

**Approximate algorithm:** Alternating optimization with linear programs — feasible, but convergence not guaranteed. Competitive at *large* compression ratios (very few parameters remaining).

**Limitation:** Simplification here means fewer maxout neurons, but reconstruction requires converting back from maxout to ReLU, which may expand parameters again. No clean conversion for deep ReLU networks.

### TropNNC 2024 — Practical Compression

**Core insight:** Each neuron in a ReLU layer corresponds to a generator of a zonotope (polytope). The network output is a tropical rational function. Similar generators (neurons) produce similar outputs — so cluster them and replace each cluster with a representative.

**Algorithm:**
1. Extract generator matrix from weight matrix (each row = one generator)
2. K-Means cluster generators into K clusters
3. Replace cluster with representative = sum of generators in cluster
4. For multi-output: simultaneous zonotope approximation to ensure shared directions

**This is NOT a conversion to tropical polynomial form.** It uses the zonotope interpretation as *motivation* for K-Means pruning. The actual computation stays in standard weight space.

**Experimental results:**
- VGG/CIFAR-10: 14.7M → 2.75M params (5.3× reduction), 93.64% → 91.71% accuracy. Competitive with ThiNet and CUP.
- ResNet18/ImageNet: modest gains only (10.98M from 11.69M, accuracy drops significantly to 41.65%).
- Requires **no training data** — weights-only method.

**Limitations:** ReLU only. No skip connections. No attention. Works best on VGG-style flat feedforward stacks. "Substantial advantage with VGG remains somewhat unexplained."

---

## Assessment Against Criteria

**Confirmed if:** tropical representations offer provably or empirically fewer parameters for the same piecewise linear function, with a computable conversion.

**Refuted if:** tropical representation is equally or more expensive, or conversion is intractable.

**Inconclusive if:** theoretical advantage exists but no practical algorithm is available.

**Verdict: INCONCLUSIVE**

The tropical polynomial representation itself (Zhang 2018) is *more expensive* — one monomial per linear region, exponentially many for deep networks. The exact division algorithm (Maragos 2023) is intractable. TropNNC achieves real compression (5× on VGG) but the algorithm is K-Means clustering with zonotope geometry as motivation — it never actually constructs a tropical polynomial. The compression benefit comes from geometric insight, not from a compact tropical form.

The tropical framing does, however, yield one implementable idea: **neurons as zonotope generators**, which naturally motivates clustering-based pruning without SGD.

---

## Implementable Techniques

1. **Zonotope generator clustering (from TropNNC):** Treat each row of a weight matrix as a generator vector. K-Means cluster generators; replace cluster with sum. No training data required. ~1 day to implement for linear layers.
   - Limitation: multi-layer interactions ignored; each layer compressed independently.

2. **Tropical factorization (approximate division, Maragos 2023):** Factor the network polynomial as d · q where q has fewer terms. Feasible only for small/shallow networks. More theoretically principled than K-Means but harder to implement.

3. **Expressivity analysis via linear region counting:** The tropical view confirms that the linear region count (≡ tropical monomial count) grows exponentially with depth. This is consistent with Balestriero's results from cycle 001 — both frameworks agree on the same compression opportunity.

---

## Connection to Research Directions

- **D2 (Tropical Geometry):** Mostly dead. The full tropical representation is intractable for deep networks. TropNNC's practical method is a geometric heuristic, not a tropical algebra computation.
- **D3 (Partition-Based Compression):** TropNNC's zonotope view confirms that neurons can be merged by geometric similarity. This is the same idea as region merging (item 5 in agenda) but motivated differently.
- **D1 (Balestriero):** Both frameworks converge on the same picture — the compression opportunity is in the gap between used and theoretical regions. Balestriero's MASO theory provides a more direct computational pathway.

**Main takeaway:** Tropical geometry provides elegant theory but intractable exact algorithms. The practical payoff is one geometric intuition (neurons as zonotope generators → cluster them), which overlaps with what MASO already gives us. Pursuing tropical geometry further is low-value. Items 3-5 (region census, partition extraction, merging) using MASO-based methods are the better path.
