# Cycle 092: Adam Polish Dissection

**Item 107** — Adam polish dissection: what does the iterative polish actually learn?

## Context

Cycle 061 (CONFIRMED): gradient polish reduces stacking degradation from +30.2% to +18.4% at 500 Adam steps (39% improvement). Cycle 082 (GN retry): Newton approximations fail, but steepest descent confirms gradient signal is valid (+14.3% improvement). The question: **is Adam learning to correct lstsq approximation error, or discovering genuinely new structure?**

## Hypothesis

Polish learns primarily residual correction (>70% energy in lstsq residual subspace), indicating low-rank correction opportunity. If true, suggests analytic post-lstsq correction is viable.

## Experiment

- Blocks [7,5,4,1] at 6× compression; topk+lstsq baseline
- Polish via 500 Adam steps (lr=1e-4, W_down only)
- **Snapshot at steps**: {0, 50, 100, 200, 500}
- Per snapshot compute:
  1. SVD of delta = W_polished − W_lstsq (rank-95%-energy, top-5 singular values)
  2. Residual vs orthogonal projection (% energy in lstsq residual subspace)
  3. Per-token reconstruction error stratified by GELU activation sparsity
  4. Single-block + stacked 4-block PPL at each checkpoint

## Success Criteria

- **Confirmed**: rank₉₅(delta) ≤ 16 AND >70% energy in lstsq residual → low-rank correction
- **Refuted**: rank₉₅(delta) > 64 OR >50% orthogonal energy → full-rank discovery
- **Inconclusive**: moderate rank (16–64) or mixed energy split

## Key Files

- `experiments/gradient_polish_scaling_gpt2.py` — cycle 061 baseline (adapt for snapshots)
- `autoresearch/results/cycle_061_stacking_sweep.json` — +4.9% polished baseline
