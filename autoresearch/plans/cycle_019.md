# Cycle 019 Plan — Activation Pattern Entropy Cascade (Toy ReLU)

## Pre-flight checks
- P1: `.venv/bin/python -c "from aleph.datasets import make_spirals; print('ok')"` — must print ok
- P2: `.venv/bin/python -c "import torch; print(torch.__version__)"` — must succeed
- P3: `ls autoresearch/results/` — no stale cycle_019 files
- P4: confirm cycle_010 single-affine residuals available in `autoresearch/results/cycle_010_*.json`

## Architecture
- 2→32→32→32→32→1, ReLU after each hidden layer
- Train: Adam lr=1e-3, 3000 epochs, BCE loss, spirals n=2000 3 turns 2× scale
- Evaluate: test set N=500 separate points

## Activation pattern extraction
For each hidden layer Li (i=1..4):
- Hook output of linear layer (pre-ReLU) to get z_i
- Binary mask: q_i = (z_i > 0).float()  → shape (N, 32)

## Entropy computation
Marginal H(Li):
  - Convert each 32-bit row to an integer index (or tuple)
  - Count pattern frequencies → probability distribution
  - H(Li) = -Σ p log2(p)

Cascade (joint entropy via concatenation):
  - H(L1) = marginal of layer 1
  - H(L1,L2): concatenate q1 and q2 (N×64), hash rows → count → entropy
  - H(L1,L2,L3): concatenate q1,q2,q3 (N×96)
  - H(L1,L2,L3,L4): all four (N×128)

Conditional entropies:
  - cH(L2|L1) = H(L1,L2) − H(L1)
  - cH(L3|L1,L2) = H(L1,L2,L3) − H(L1,L2)
  - cH(L4|L1,L2,L3) = H(L1,L2,L3,L4) − H(L1,L2,L3)

## Single-affine residual comparison
Load cycle_010 results JSON for layer-wise r values. If unavailable, compute inline:
For each layer Li, fit a single linear model (affine map) to (input_to_Li, output_after_ReLU_Li)
using least squares. Residual = 1 − R² or mean squared error normalized by output variance.

## Output artifacts
- `autoresearch/results/cycle_019_entropy_cascade.json` — all entropy values
- `autoresearch/results/cycle_019_entropy_cascade.png` — bar chart with two y-axes: marginal H + conditional H cascade
- `autoresearch/results/cycle_019_summary.md`

## Confirmation logic
- Conditional entropy drop ratio = cH(Li|…) / H(L1) for each i
- Confirmed if min(drop_ratio) ≤ 0.5 AND that layer has r ≤ median(r_all)
- Refuted if all drop_ratios > 0.8
- Inconclusive otherwise
