# Cycle 199 Brief: K-sweep for near-rank-1-input blocks (Item 209)

## Item

209 — K-sweep for near-rank-1-input blocks — does rank-1 residual enable much smaller K?

## Context

- Cycle 198 found blocks 3-11 have input_rank_90=1 (near-rank-1 residual stream).
- Current recipe: BLOCKS=[1,4,7,11], K=512 → PPL 30.089 (cycle 196, measured on calibration data).
- Hypothesis: if FFN input lives on 1D manifold, K=8-32 may be sufficient (rank-1 → scalar input).

## Experiment Design

- Test K={8,16,32,64,128,256,512} for BLOCKS=[1,4,7,11] using topk+lstsq.
- Calibration: 200 sequences, SEQ_LEN=128 from wikitext-2 validation.
- Eval: 100 separate sequences (offset 200-300) from same validation split.
- Script: `experiments/k_sweep_rank1_cycle199.py`
- Output: `autoresearch/results/cycle_199_k_sweep.json`

## Result

REFUTED — see `autoresearch/reviews/cycle_199.md` for full analysis.
