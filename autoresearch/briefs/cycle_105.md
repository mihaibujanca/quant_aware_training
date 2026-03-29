# Cycle 105 Brief — Item 116

**Task**: GELU negative-leak aware compression (GPT-2 FFN)

## Context
- **Cycle 103**: 84-88% of GELU-ReLU disagreement energy originates from z<0 (negative leak)
- **Cycle 104**: Cascade is attention-driven, not FFN nonlinearity — GELU attenuates error in early layers, pivots at T6–T7
- **Current recipe**: topk+lstsq+500 Adam at k≤4 gives +5% PPL (k=1, 6× compression)

## Hypothesis
Standard topk+lstsq treats all hidden channels uniformly. Exploit sign asymmetry: partition channels into "positive-dominated" (z>0 ≥50% tokens) and "negative-dominated" (z<0 ≥50% tokens). Negative-dominated channels contribute mainly via GELU's small negative leak — a bias correction, not primary signal. Strategy:
1. Compute empirical z<0 fraction per channel on probe data
2. Approximate aggregate leak as b_leak = W_out @ mean(GELU(z_neg))
3. Drop negative-dominated channels from topk selection
4. Run topk+lstsq on positive-dominated channels only, add b_leak as offset

## Scope
- **Blocks**: [1, 4, 7, 11] (6× compression, 3 seeds)
- **Success**: leak-aware ≥3% PPL gain vs standard topk+lstsq at matched k (≥2/4 blocks) → CONFIRMED
- **Fail**: ≤1% difference (sign statistics don't help) → REFUTED
- **Inconclusive**: 1-3% or block-dependent

## Integration
Cascade already attributed to attention (cycle 104). This tests whether channel-level sign statistics improve FFN compression orthogonally to existing topk recipe.
