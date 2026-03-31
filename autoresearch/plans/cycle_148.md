# Plan — Cycle 148 (Item 158)

## Goal
Determine whether the AE encoder's ReLU nonlinearity is necessary for its 3% advantage over topk, or whether end-to-end training alone accounts for the gain.

## Method
Fork `experiments/ae_param_match_k512_cycle137.py` into `experiments/linear_ae_ablation_cycle148.py`.

### Changes from cycle 137
1. Add `use_relu` flag to `BottleneckFFN.__init__`. When False, forward does `z = self.encoder(h)` (no ReLU).
2. Three methods: `relu_ae` (d_bn=64, ReLU), `linear_ae` (d_bn=64, no ReLU), `topk` (lstsq baseline).
3. Drop full AE (d_bn=512) — not needed for this ablation.
4. Pre-polish only (500 AE training steps, no polish phase). Cycle 137 already has post-polish data for ReLU AE and topk; adding polish here would conflate training duration with architecture.
5. Blocks [4, 7], seeds [42, 137, 7], 200 eval sequences.

### Param counts
- ReLU AE d_bn=64: 3072×64 + 64 + 64×768 + 768 = 246,592
- Linear AE d_bn=64: identical (same weight shapes, just no ReLU activation)
- topk k=512: 512×768 + 768 + 768×512 = 787,712

### Verdict criteria (from agenda)
- **Confirmed** if ReLU AE beats linear AE by >=2% relative on >=2/2 blocks (nonlinearity matters).
- **Refuted** if linear AE matches ReLU AE within 1% (nonlinearity is cosmetic; training objective is what matters).
- **Inconclusive** if difference is 1-2% or block-dependent.

## Output
- `autoresearch/results/cycle_148_linear_ae_ablation.json` — per-seed per-block PPL for all 3 methods
- stdout summary table with mean±std across seeds

## Verification
- Float PPL should match cycle 137 (~57.65) — same corpus, same teacher.
- ReLU AE pre-polish numbers should be within noise of cycle 137 pre-polish (B4~54.2, B7~55.3).
- If linear AE diverges or produces NaN, that itself is informative (ReLU provides regularization).

## Runtime
~20 min. 2 blocks × 3 methods × 3 seeds × 500 training steps = 9000 training steps total. Each step is a single forward+backward on 4 sequences of length 128.
