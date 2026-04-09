# Cycle 376 Brief — Per-Token CKA and Trajectory Divergence (Item 384)

## Context
- **c371**: Second-order geometry (eff_rank, participation ratio, mean cosine) cannot distinguish 3-block (PPL ~47) from 6-block (PPL ~65). All four metrics share the same information source (covariance eigenspectrum).
- **c371 audit recommendation**: Try CKA (kernel-normalized representational similarity) and per-token L2 divergence as higher-order diagnostics.
- **Working recipe**: topk k=512 + lstsq + MLP adapter h=64. 3-block [b2,b1,b10] PPL ~47, 6-block [b2,b1,10,b4,b3,b11] PPL ~65.
- **No saved weights**: must train both configs from scratch (same as c371/c374/c375).
- **Existing CKA code**: `experiments/cka_gpt2_layers.py` (c024) has a tested `linear_cka()` implementation.

## Hypothesis
CKA measures kernel-normalized alignment between float and compressed hidden representations. Unlike second-order covariance statistics, CKA captures global distributional similarity and is invariant to orthogonal transform and isotropic scaling. If the composition barrier manifests as representational misalignment, CKA should identify specific layers where 6-block diverges from float while 3-block stays aligned.

## Task
1. Train 3-block and 6-block compressed models (standard recipe, 500 steps, seed 42).
2. Run a shared float model and each compressed model on calibration data, hooking all 12 block output activations.
3. Compute layer-wise linear CKA(float_hidden_l, compressed_hidden_l) for each of 12 layers.
4. Compute per-token L2 divergence at each layer: mean and 95th percentile of ||h_float - h_compressed||.
5. Plot CKA profiles and L2 divergence for both configs.

## Success Criteria
- **Confirmed**: CKA gap > 0.1 at any layer between 6-block and 3-block (6-block diverges where 3-block does not)
- **Refuted**: max CKA gap < 0.05 (profiles indistinguishable)
- **Inconclusive**: max gap in [0.05, 0.1]

## Key References
- Float baseline: 53.89
- 3-block target PPL: ~47
- 6-block target PPL: ~65
- CKA implementation: `experiments/cka_gpt2_layers.py` line 105
- Training infrastructure: `experiments/gpt2_adaptive_k_cycle374.py`
