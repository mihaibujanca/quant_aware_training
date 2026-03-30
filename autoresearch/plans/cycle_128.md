# Cycle 128 Plan: Dead/Always-Active Neuron Census (Item 134) — Rev 1

## Pre-flight
- **P1 (duplicate)**: Items 47 (cycle 032), 94 (cycle 083) both found <1% strict-dead. Item 134 adds: (a) soft census (per-channel token fractions), (b) always-active category, (c) 3-seed consistency, (d) 65536 tokens. REFUTED risk remains high for strict counts but soft census may reveal approximate pruning targets.
- **P2–P4**: No deps, no program.md changes.
- **P5 (null)**: N(0,1) z-values: P(channel dead for ALL 65536) ~ 0. For soft census null: ~0.13% of tokens below z=-3 per channel. We beat null if learned biases shift per-channel soft fractions well above 0.13%.
- **P6 (GELU)**: Dead: z < -3 (GELU(-3) = -0.004). Always-active: z > 4 (GELU(4) ≈ 3.9998, linear). Near-linear: max|GELU(z)-z| < 0.01.
- **P7 (power)**: 65536 tokens × 3 seeds = 196608 token observations per channel.

## Hypothesis
GPT-2 FFN has channels with learned biases pushing pre-GELU activations into GELU tails, enabling exact (strict) or approximate (soft) pruning.

## Steps
1. Load GPT-2. For each seed in [42, 123, 7]: tokenize 256 OWT sequences (seq_len=256).
2. Forward pass with hooks on `block.mlp.c_fc` (pre-GELU). Per channel per block, collect:
   - min(z), max(z) across all 65536 tokens (for strict census).
   - **frac_dead**: fraction of tokens with z < -3 (soft dead).
   - **frac_active**: fraction of tokens with z > 4 (soft always-active).
   - **frac_nearlinear**: fraction of tokens where |GELU(z)-z| < 0.01 (soft near-linear).
3. **Strict census** (per seed): dead = max(z) < -3; always-active = min(z) > 4; near-linear = max|GELU(z)-z| < 0.01 over all tokens.
4. **Soft census** (per seed): for each channel, report frac_dead, frac_active, frac_nearlinear. Summarize distribution: histogram of frac_dead across channels (bins: 0-1%, 1-50%, 50-95%, 95-99%, 99-100%). Same for frac_active. Report count of channels with >95% and >99% token fraction in any removal zone.
5. Report per-block z-distribution summary: channel-wise mean/std of (mean_z, min_z, max_z).
6. Cross-seed consistency: for strict counts report min/max/mean across 3 seeds. For soft fractions report per-channel std across seeds (median and 95th percentile).
7. If total strict-removable >= 1% (any seed): construct pruned model, evaluate single-block and k=4 stacked PPL on blocks [7,5,4,1].
8. If strict-removable < 1%: skip pruning. Soft census results still reported as main finding.

## Validation
- Sanity: verify GELU(-3) = -0.004, GELU(4) ≈ 3.9998 at script start.
- If any block reports >20% strict-dead, suspect hook bug.
- Unpruned model PPL must match float baseline (51.66 ± 0.5).

## Criteria
- **Confirmed**: >=5% strict-removable/block AND exact pruning <0.1% PPL change single-block.
- **Refuted**: <1% strict-removable AND no channels with >95% soft dead/active token fraction (across all 3 seeds). Interpretation: GELU networks have no free compression from stable channels.
- **Inconclusive**: strict <1% but some channels >95% soft fraction (approximate pruning potential exists but untested).

## Artifacts
- `experiments/dead_neuron_census_gpt2_cycle128.py`
- `autoresearch/results/cycle_128_census.json` (strict + soft counts, per-seed)
- `autoresearch/results/cycle_128_census.png` (per-block bar chart + soft fraction histogram)
