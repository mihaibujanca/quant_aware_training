# Item
149 — Layer-local pullback metric via rest-of-network Jacobian — GPT-2 FFN (P2, analysis)

# Context
- Cycle 136 (item 146): global pullback G = U^T U is constant rescaling of Frobenius (ratio ~34-35 across blocks 0-10). Frobenius rho=0.797, G-weighted rho=0.734. G adds no information at FFN output level.
- Item 149 tests whether per-layer JVP through remaining layers captures non-uniform downstream amplification that the global metric misses.
- Cycle 135: curvature-damage correlation |rho|=0.645, curvature peaks early (blocks 2-4), damage peaks late.

# Dependencies
- 146 (cycle 136, REFUTED): provides Frobenius baseline rho=0.797 and per-block PPL deltas. MET.
- Cycle 135 per-block PPL deltas available in `autoresearch/results/cycle_135_curved_inference.json`.

# Key Files
- `experiments/pullback_error_cycle136.py` — template for per-block error measurement + PPL correlation
- `autoresearch/results/cycle_136_pullback_error.json` — Frobenius/G-weighted baselines
- `aleph/pullback_geometry.py` — existing pullback utilities
- `aleph/lm_geometric_compression.py` — teacher loading, probe corpus, swap infra

# Constraints
- JVP through 12-l layers per probe direction per block. 12 blocks x 768 directions = 9216 JVPs. Too expensive.
- Instead: compute ||J_l e_l||_2 for the actual compression error e_l (one JVP per block = 12 JVPs total). This is the quantity that matters.
- CPU runtime: ~12 forward passes through partial networks. Manageable (<5 min on MPS/CPU).
- 3 seeds, 200 eval sequences per the agenda item.

# Handoff
Write `experiments/layer_local_pullback_cycle144.py`. Compute three metrics per block: Frobenius ||e_l||, global ||U e_l||, layer-local ||J_l e_l||. Correlate each with single-block swap PPL delta. Report Spearman rho for all three. Confirmed if layer-local rho exceeds global by >=0.15.
