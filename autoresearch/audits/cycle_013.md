# Audit — Cycle 013

## Item
24 — RTG spectral coarsening: Fiedler-based region merging — toy ReLU

## Phase 1: Verdict Check

Claimed outcome: **inconclusive** — spectral beats random by up to 9.4pp but never achieves < 5% accuracy drop from float.

Agenda criteria:
- Confirmed: ≥3× region reduction with <5% accuracy drop AND outperforms random by ≥5pp.
- Refuted: spectral does not outperform random merging.
- Inconclusive: spectral outperforms on spirals but benefit disappears for 100D embedded version.

**Assessment**: The outcome doesn't exactly match any criterion. Spectral DOES outperform random (by 9.4pp at K=8 — clearing the 5pp threshold), so it's not refuted. But it never achieves <5% accuracy drop (best = 60% vs 95.5% float = 37% drop), so it's not confirmed either. The agenda's inconclusive criterion mentions 100D embedding, which wasn't tested.

The verdict of inconclusive is reasonable but for a different reason than the agenda specifies. The real issue is RTG graph fragmentation (79 components, 330 patterns, mean degree 1.7). The Hamming-1 graph is too sparse to provide useful spectral structure.

## Phase 2: Quality Checks

### G1: Methodology
- Float accuracy 95.5% (above 90% threshold) — model is trained.
- Sanity check: K=N_patterns gives 100% accuracy — correct.
- Random baseline: 10 seeds, mean±std reported — good practice.
- RTG construction: Hamming-1 adjacency is correct per agenda specification.
- Spectral clustering: scipy eigsh on Laplacian, k-means on eigenvectors — standard approach.

### G2: Reproducibility
Single seed for network, 10 seeds for random baseline. Acceptable.

### G3: Tautology Check
Not tautological. The fragmented RTG is a genuine finding about the topology of activation pattern space.

### G4: Scope
Within scope. The 100D embedded version mentioned in the inconclusive criterion was not tested, but the primary result (fragmented graph preventing useful compression) is informative.

### G5: Counterintuitive
The sparse Hamming-1 graph is somewhat surprising — one might expect nearby activation patterns to be more connected. In fact, a 64-bit binary vector has 64 possible Hamming-1 neighbors, but most aren't realized in the observed pattern set. This highlights that activation patterns are sparsely distributed in binary space.

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: none
reason: Spectral signal is real (beats random significantly) but graph fragmentation prevents useful compression. The inconclusive outcome correctly reflects that the method shows promise but fails in practice on this architecture. Item 25 (GPT-2 analog) uses k-NN Jacobian graph instead of Hamming-1, which avoids the sparsity problem — still worth pursuing.
recommendations: Item 25 uses continuous Jacobian distance instead of binary Hamming, so it doesn't suffer from RTG sparsity. The deeper insight: Hamming-1 adjacency is too strict for networks with many neurons. Future toy experiments could use Hamming-d (d>1) thresholds.
