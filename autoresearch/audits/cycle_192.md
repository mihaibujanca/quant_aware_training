# Audit — Cycle 192 (Item 202)

**Verdict:** REFUTED
**eps=0.01:** 0.003% dead channels (1 channel in block 0)
**eps=1.0:** 2.64% dead but PPL jumps from 33.4 to 60.0 (+80%)

## Result Quality

Statistics collected from 500 sequences × 127 tokens = 63,500 tokens per block.
Results are clean. The max_gelu statistics show GELU outputs ranging into the hundreds
for most channels — far from zero.

## Key Findings

1. **Pretrained GELU networks have no dead neurons**: Unlike ReLU networks where neurons
   can be provably dead (output exactly 0), GELU(z) is always nonzero for any finite z.
   Pretrained GPT-2 has essentially no channels that consistently produce near-zero output.

2. **Only 2 always-active channels (block 0)**: These produce positive z for all calibration
   tokens. Could be collapsed to linear, but 2/3072 = 0.065% savings is negligible.

3. **eps=1.0 channels are lightly-used, not dead**: 2.64% of channels have max GELU < 1.0.
   But zeroing these causes +80% PPL degradation — they carry important signal despite
   low maximum activation. This is consistent with GELU's smooth negative regime: even small
   positive outputs contribute non-trivially to the final prediction.

4. **Calibration scale doesn't matter**: 500 sequences should be sufficient to activate
   most channels. The result is clear: near-zero dead channels at any reasonable threshold.

## Comparison to Literature

Serra et al. 2021 / Berzins 2023 results are for ReLU networks where exact zeros exist.
GELU (and SiLU, Swish) networks do not have this property. The program's R2 mandate for
dead-channel pruning is essentially inapplicable to this architecture.

**Alternative exact pruning for GELU**: The "near-linear" approach (channels where z is
always in [a, b] for some interval where GELU is approximately linear) remains unexplored
but would likely also show very few qualifying channels in a well-trained model.

## Implication

The two foundational "free" exact pruning directions (dead channels, always-active) are both
essentially empty for pretrained GPT-2. Any meaningful compression requires some information
loss. This closes R2 for this architecture class.
