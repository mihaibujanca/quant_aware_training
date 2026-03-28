# Research Status — 68 Cycles (001–068)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 13 |
| Refuted | 20 |
| Inconclusive | 26 |
| Crash | 1 |

## Latest Cycle
- **Cycle 068** (item 80): FFN Jacobian effective rank census — GPT-2 FFN — INCONCLUSIVE
  - 10/12 blocks have rank_99 ≤ 500 (low-rank confirmed), but Block 1 outlier at 588 (near random baseline 592)
  - Spearman(rank_99, P1_frob_std) = -0.497 (below +0.5 threshold, wrong sign)
  - Rank does NOT correlate with compression damage (rho=-0.33, p=0.30)
  - Low rank is real but does not predict compressibility

## Final Answer (from cycle 061, still valid)

**Research question**: "Given a trained transformer LM and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN reparameterization that preserves geometry without SGD?"

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6×** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Supporting Findings
- **Per-block is essentially free**: single-block 6× gives sub-baseline PPL (lstsq regularization)
- **Stacking is the sole bottleneck**: +30% at k=4 vs sub-baseline per-block (cycles 057-058)
- **Cascaded re-probing doesn't help**: error propagates through attention/LN (cycle 060)
- **Boundary corrections partially help**: +5pp gain, but far less than gradient polish (cycle 063)
- **Gradient polish is the answer**: 500 steps = 84% relative reduction of stacking error (cycle 061)
- **Compression-difficulty oracle**: P1 Frobenius std rho=0.84, corpus-invariant rho=0.94 (cycles 048, 055)
- **Channel selection is data-hungry**: Jaccard@256=0.517, needs >2K tokens (cycle 067)
- **FFN Jacobian rank**: 10/12 blocks rank_99 ≤ 500 (low-rank), but rank does not predict compressibility (cycle 068)

### Dead Ends
- Topology-based methods (PH, Betti, topological distillation): 4 experiments, 0 useful results
- Learned/distilled compressors: geometric baseline dominates (cycles 030, 049)
- Cascaded re-probing: negligible improvement (cycle 060)
- Variable per-block ratios: breaks correction step (cycle 050)

## Remaining NEXT Items
- **P0**: Items 81–84 (spectral analysis, depend on 80 — now complete) — 81 next (subspace alignment)
- **P0**: Item 64 (corpus-diverse probing) — no deps, ready to run
- **P1**: Item 66 (mixed-corpus refit) — blocked on item 64
- **P1**: Item 28 (persistent homology GPT-2) — unblocked
- **P1**: Item 35 (region volume Pareto GPT-2) — unblocked
- **P2**: Items 85–88 (moonshots) — unblocked
- **Closed**: Items 23 (superseded by 80), 41, 44 (refuted deps) — OBSOLETE, do not run
