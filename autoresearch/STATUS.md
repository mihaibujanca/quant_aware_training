# Research Status — 69 Cycles (001–069)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 13 |
| Refuted | 20 |
| Inconclusive | 27 |
| Crash | 1 |

## Latest Cycle
- **Cycle 069** (item 81): Jacobian subspace alignment — GPT-2 FFN — INCONCLUSIVE (audit-corrected: implementation bug)
  - Jacobians computed at block input instead of FFN input (post-attention, post-ln_2). All results unreliable.
  - Must re-run with register_forward_hook on mlp.c_fc (matching cycle 068 methodology)

## Final Answer (from cycle 061, still valid)

**Research question**: "Given a trained transformer LM and a corpus drawn from the data distribution it was trained on, how do we construct a much smaller FFN reparameterization that preserves geometry without SGD?"

**Answer**: Pure no-SGD compression works per-block but fails when stacking. The practical recipe requires minimal gradient polish:

### The Recipe (cycle 061)
1. **topk + lstsq** → analytic single-block FFN compression (sub-baseline PPL per block)
2. **Stack k=4 blocks at 6x** → +30% PPL unpolished
3. **500 Adam steps** (lr=1e-4, 4096 tokens, W_down only) → **+4.9% PPL on OpenWebText**

### Key Supporting Findings
- **Per-block is essentially free**: single-block 6x gives sub-baseline PPL (lstsq regularization)
- **Stacking is the sole bottleneck**: +30% at k=4 vs sub-baseline per-block (cycles 057-058)
- **Gradient polish is the answer**: 500 steps = 84% relative reduction of stacking error (cycle 061)
- **Compression-difficulty oracle**: P1 Frobenius std rho=0.84, corpus-invariant rho=0.94 (cycles 048, 055)
- **FFN Jacobian rank**: 10/12 blocks rank_99 <= 500 (low-rank), but rank does not predict compressibility (cycle 068)
- **Jacobian subspace alignment**: cycle 069 invalid (implementation bug), needs re-run

### Dead Ends
- Topology-based methods (PH, Betti, topological distillation): 4 experiments, 0 useful results
- Learned/distilled compressors: geometric baseline dominates (cycles 030, 049)
- Cascaded re-probing: negligible improvement (cycle 060)
- Variable per-block ratios: breaks correction step (cycle 050)

## Remaining NEXT Items
- **P0**: Items 82–84 (spectral analysis, depend on 80 — complete) — run next
- **P0**: Item 64 (corpus-diverse probing) — no deps, ready
- **P1**: Item 66 (mixed-corpus refit) — blocked on 64
- **P1**: Item 28 (persistent homology GPT-2) — unblocked
- **P1**: Item 35 (region volume Pareto GPT-2) — unblocked
- **P2**: Items 85–88 (moonshots) — unblocked
- **Note**: Item 81 needs re-run with correct hook methodology (should be added back to NEXT)
