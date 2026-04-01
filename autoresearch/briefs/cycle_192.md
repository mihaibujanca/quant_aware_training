# Cycle 192 Brief: Exact Dead-Channel Pruning Baseline (Item 202)

## Item

202 — Exact dead-channel pruning baseline — GPT-2 FFN (Serra/Berzins required baseline)

## Context

The research program (R2) mandates exact white-box pruning (Kumar/Serra/Berzins) as the
foundational baseline. This has not yet been done despite 191 cycles. The core idea:
- Dead neurons: channels where the activation is effectively 0 for all calibration tokens
- Always-active neurons: channels where the activation is always positive (linear behavior)
- These can be removed/folded exactly without any retraining

For GPT-2 with GELU: GELU(z) is never exactly 0 (no hard threshold), but for z << 0,
GELU(z) ≈ 0. A channel is "near-dead" if max_{tokens} GELU(z_c) < epsilon.

## Experiment Design

1. Load float GPT-2, tokenizer, OWT calibration data (N=500 seqs, SEQ_LEN=128)
2. Forward hooks on each block's GELU input to collect z_c per channel across all tokens
3. For each channel c in each block b:
   - Compute max_activation[b][c] = max over all calibration tokens of |GELU(z)|
   - Channel is "near-dead" if max_activation[b][c] < epsilon (try epsilon=0.01, 0.1, 1.0)
4. Count near-dead channels at each threshold
5. Prune near-dead channels: zero out their rows in c_fc.weight, c_proj.weight
6. Measure PPL at each threshold

## Verdict

- Confirmed: near-dead (epsilon=0.01) pruning achieves >5% channel reduction at PPL <25 in ≥8/12 blocks
- Refuted: fewer than 1% of channels near-dead at epsilon=0.01
- Inconclusive: small counts, PPL impact unclear
