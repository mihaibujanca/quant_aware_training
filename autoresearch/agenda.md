# Research Agenda — Transformer-First Geometric Compression

This queue supersedes the old toy-model-first plan. The goal is to answer the right transformer FFN compression questions, in the right order, with distillation as a control and not the mainline.

## NEXT EXPERIMENTS

### P0

10. **Single-block swap localization curve**
   - For a serious teacher (`nanochat` if available, else proxy `gpt2`), compress one FFN block at a time using the current analytic method and a matched-budget distillation baseline. Sweep a small budget ladder and depth-stratified block subset. Measure both local block metrics and full-model metrics after the single-block replacement.
   - Type: empirical
   - Confirmed if: at least 3 tested blocks achieve `>=4x` FFN parameter reduction with the analytic method while keeping relative perplexity increase `<=10%` and staying within `25%` of the distilled baseline on the same swap.
   - Refuted if: the analytic method causes relative perplexity increase `>25%` for more than `80%` of tested single-block swaps at `>=4x` reduction.
   - Inconclusive if: results vary strongly by depth with no stable pattern.
   - Depends on: 9

11. **Stable and near-linear channel census in pretrained FFNs**
   - Implement the exact or near-exact pruning census implied by Kumar/Serra/Berzins for transformer FFNs: dead channels, always-positive channels, always-negative channels, and near-linear channels under the actual probe distribution. Report fractions per block and test whether these fractions are robust enough to justify an exact-pruning baseline.
   - Type: empirical
   - Confirmed if: stable or near-linear channels account for at least `5%` of hidden width in a majority of tested FFN blocks.
   - Refuted if: stable or near-linear channels stay below `1%` in more than `80%` of tested blocks.
   - Inconclusive if: the fraction is nontrivial but highly dependent on the chosen probe corpus.
   - Depends on: 9

12. **Progressive multi-block accumulation curve**
   - Use item 10 to rank blocks by single-block swap damage, then compress blocks cumulatively in three orders: easiest-first, hardest-first, and random. This tests whether no-training compression failure is mostly local, mostly cumulative, or strongly order-dependent.
   - Type: empirical
   - Confirmed if: easiest-first cumulative compression materially outperforms hardest-first and random, e.g. `>=25%` lower perplexity or KL at the same number of swapped blocks.
   - Refuted if: cumulative degradation is nearly identical across orders, implying that single-block localization is not informative.
   - Inconclusive if: order helps only at very small block counts and disappears quickly.
   - Depends on: 10

13. **Probe-corpus robustness of geometry signals**
   - Compare the FFN geometry atlas across disjoint probe subsets and, if available, across a proxy corpus vs a more teacher-like corpus. Measure robustness of stable-channel masks, channel rankings, Jacobian spectra, and selected hidden-unit sets. The aim is to determine whether the compression signal is genuinely teacher-specific or just probe-noise.
   - Type: empirical
   - Confirmed if: for a majority of tested blocks, score rank correlation is `>=0.8` and selected-set overlap is `>=50%` across disjoint probes.
   - Refuted if: selected-set overlap drops below `20%` and score correlations stay below `0.5` for most blocks.
   - Inconclusive if: early and late blocks behave very differently.
   - Depends on: 9, 11

### P1

14. **Rank-only baseline versus geometry versus distillation**
   - Add a matched-budget SVD / low-rank FFN baseline and compare it directly to the current geometric method and distillation on the same single-block swaps and budgets. This asks whether the current "geometric" method is actually better than a plain matrix-factorization baseline.
   - Type: empirical
   - Confirmed if: the geometric method beats the rank-only baseline on full-model metrics in more than `50%` of tested block-budget pairs at equal FFN parameter count.
   - Refuted if: rank-only beats the geometric method in more than `80%` of tested block-budget pairs.
   - Inconclusive if: wins trade back and forth by block or budget with no consistent edge.
   - Depends on: 10

15. **Centroid and Jacobian affinity as merge signals**
   - Evaluate whether data-conditioned similarity measures from the centroid-affinity / local-map literature predict safe compression better than raw weight cosine. For each block, compare weight similarity, centroid similarity, Jacobian-spectrum similarity, and any combined score against actual swap damage under controlled budgeted compressions.
   - Type: empirical
   - Confirmed if: a centroid/Jacobian-based score improves Spearman correlation with swap error by at least `0.15` over raw weight cosine on a majority of tested blocks.
   - Refuted if: raw weight cosine is just as good or better across the board.
   - Inconclusive if: the better signal changes by block depth or corpus.
   - Depends on: 10, 11

16. **Exact prune plus linear collapse baseline**
   - Implement the strongest exact baseline from the literature: stable-channel deletion and always-active / near-linear collapse before any heuristic merging. Compare exact-prune-only, exact-prune-plus-least-squares-refit, and the current geometric method on single-block swaps. This is the cleanest test of how much "free" no-SGD compression exists.
   - Type: implementation
   - Confirmed if: exact or near-exact pruning alone yields at least `1.1x` FFN parameter reduction with `<=1%` relative perplexity increase on the tested single-block swaps.
   - Refuted if: exact or near-exact pruning yields negligible savings or immediately breaks block behavior.
   - Inconclusive if: savings exist only under loose thresholding that is hard to justify.
   - Depends on: 11

17. **Local-chart analytic compression inside one FFN block**
   - Replace the current global top-k analytic compression with a local-chart construction: cluster FFN inputs on the probe distribution, derive a small set of local charts, and fit the compressed block analytically or semi-analytically per chart. Test this on representative easy, medium, and hard blocks from item 10.
   - Type: implementation
   - Confirmed if: local-chart compression improves full-model single-block metrics by at least `20%` over the global analytic method at equal budget on at least `3` representative blocks.
   - Refuted if: local-chart compression fails to improve over the global analytic baseline or is less stable.
   - Inconclusive if: it helps one class of blocks but not the others.
   - Depends on: 10, 13, 15

18. **Learned compressor generalization on held-out blocks**
   - Train the descriptor-autoencoder / hypernetwork compressor on a subset of blocks and evaluate on held-out blocks from the same model family. If exact nanochat checkpoints at multiple depths are available, also test cross-checkpoint generalization. The point is to determine whether the learned track discovers reusable structure or merely memorizes block identities.
   - Type: empirical
   - Confirmed if: on held-out blocks, the learned compressor beats the current geometric baseline in a majority of cases and stays within `25%` of its in-distribution performance.
   - Refuted if: held-out performance collapses relative to both the geometric baseline and in-distribution performance.
   - Inconclusive if: generalization works only within narrow depth bands or only within one checkpoint.
   - Depends on: 10

## PAST EXPERIMENTS

### Cycle 001

1. **Survey: Balestriero's piecewise linear / spline theory of deep learning**
   - Type: literature
   - Outcome: **confirmed** — 6 concrete, implementable techniques identified (≥3 required)
   - Cycle: 001
   - Commit: (see cycle_001_manifest.txt)
   - Artifact: `autoresearch/surveys/balestriero_spline_theory.md`
   - Key finding: Activation pattern census (Technique 1) and affine map extraction (Technique 2) are directly actionable from MASO theory; SplineCam (Technique 3) provides exact 2D enumeration. Gap: analysis-to-synthesis route for minimal network construction remains open.

### Cycle 003

7. **Survey: Non-distillation model compression methods (2023-2026)**
   - Type: literature
   - Outcome: **confirmed** — found ≥2 papers with working no-SGD compression methods
   - Cycle: 003
   - Commit: 66021d0
   - Artifact: `autoresearch/surveys/literature_map.md`
   - Key findings:
     - **Serra et al. NeurIPS 2021 (2102.07804)**: ReLU stability criterion — MIP-based dead/always-active neuron detection → 56% connection removal, 183x faster than prior art. No SGD required.
     - **Berzins 2023 (2306.07212)**: GPU-accelerated partition extraction + §5.4 pruning demo → 78% parameter reduction (2,50,50,1 → 2,25,19,1) via dead + always-active removal. Constructive, no SGD.
     - **3-Phase algorithm** identified: Phase 1 (dead neuron removal) + Phase 2 (always-active collapse to ≤D neurons/layer) = solvable analytically. Phase 3 (merging genuinely unstable neurons) = open problem.
     - Depth reduction ruled out: Arora et al. (1611.01491) proves exponential neuron cost for depth reduction → target width compression only.

### Cycle 002

2. **Survey: Tropical geometry and neural networks**
   - Type: literature
   - Outcome: **inconclusive** — tropical polynomial form is exponentially large (intractable for deep nets); practical TropNNC algorithm uses zonotope clustering but never builds a compact tropical form
   - Cycle: 002
   - Commit: TBD
   - Artifact: `autoresearch/results/cycle_002_summary.md`
   - Key finding: Tropical geometry = dead end for direct compression. One salvageable idea: neurons-as-zonotope-generators motivates clustering-based pruning (overlaps with MASO region merging). Do NOT pursue tropical algebra further.

### Cycle 004

9. **Benchmark lock-in and asset audit**
   - Type: analysis
   - Outcome: **confirmed** — exact-open nanochat benchmark and explicit `gpt2` proxy benchmark are both now documented, with concrete path conventions and acquisition procedures
   - Cycle: 004
   - Commit: TBD
   - Artifact: `autoresearch/results/cycle_004_summary.md`
   - Key finding: the exact-open target is `karpathy/nanochat` base checkpoints plus `climbmix-400b-shuffle`, but those assets are not present locally; the only benchmark runnable immediately on this machine is the cached `gpt2` proxy path, which must remain clearly labeled proxy-only.
