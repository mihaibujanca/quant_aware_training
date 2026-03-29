# Cycle 090 — Attention Block Compression Feasibility Scan — GPT-2

**Item 105** (P1): Pivot from FFN to attention blocks. 89 cycles have compressed FFN only; program.md §36 mandates FFN-only first, leaving attention untouched. This is the minimal gate scan to decide if attention compression is a viable research direction.

## Experiment Design

For all 12 attention blocks in GPT-2:
1. **Per-head output rank** — SVD of head output on 4096 tokens (nt_q @ W_o reduction)
2. **Single-head ablation damage** — Zero one head, measure PPL delta (importance ranking)
3. **Single-block random+lstsq at 6×** — Project V-projection 768→128 per head, lstsq reconstruct

## Success Criteria

- **Confirmed** (≥3/12 blocks): Single-block swap ≤+5% PPL at 6×
- **Refuted** (all blocks): All blocks >+15% PPL
- **Inconclusive** (1-2 blocks): Mixed results

## Context

Latest cycle 089 (item 104): FFN error compounds multiplicatively, ~41% PPL growth per block. Hard ceiling: k≥4 blocks at 6× is unviable. This motivates the pivot: can attention blocks compress better than FFN?

Cycle 085 audit explicitly recommended attention as the next frontier.

## Setup Notes

- Program constraint (line 36): FFN-only compression first. Attention is first departure — treat as feasibility scan, not a full compression.
- No existing attention extraction code in aleph/ (transformer_analysis.py has model hooks but not per-head analysis).
- Likely need: hook-based attention output capture, per-head rank SVD, head ablation via masking.

## Expected Duration

~2–3 hours (data capture + analysis, no SGD).
