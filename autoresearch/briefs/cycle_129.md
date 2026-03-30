# Cycle 129: Item 100

## Item
**Is FFN isotropy architectural or learned? — GPT-2 FFN**

## Context
Follow-up to item 99 (cycle 086). Cycle 086 found moderate isotropy ratios across blocks (0.0043–0.0157) — all within 0.01–0.1 range, so criteria not met. Item 100 tests whether isotropy is inherent to architecture or emerges from training.

## Dependencies
- Item 99 (cycle 086): Measured FFN output manifold isotropy on trained GPT-2
  - Block 1 isotropy_ratio = 0.0043 (most isotropic)
  - Blocks 4,5,7 = 0.015–0.016 (moderately isotropic)
  - Spectrum steeper than Marchenko–Pastur (53–72% eigenvalues outside support)

## Tests
1. **Random-init GPT-2** (no training): measure same isotropy metric
   - Expected if architectural: same ratio; if learned: near 1.0 (anisotropic)
2. **Single-block ReLU swap**: replace GELU with ReLU in one block
   - Expected if learned via smoothness: isotropy breaks; if architectural: persists
3. **Training checkpoints** (if available): isotropy vs training step curve

## Constraints
- Isotropy measurement from cycle 086: eigenvalue ratio (λ_max/λ_min) on post-GELU activations
- Model: GPT-2 (124M params, 12 blocks, 3072-dim FFN)
- Confirmed if: random-init NOT isotropic AND trained IS
- Refuted if: random-init already isotropic
- Inconclusive if: both moderately isotropic

## Key Files
- Artifacts (cycle 086): `experiments/ffn_manifold_geometry_gpt2.py`, `autoresearch/results/cycle_086_manifold_geometry.json`
- Related: `autoresearch/surveys/gelu_relu_inference.md` (GELU–ReLU analysis)

## Handoff
Check whether isotropy is structural (ReLU geometry, weight initialization) or learned (training dynamics). Test 1 is critical: if random init is already isotropic, item is REFUTED immediately.
