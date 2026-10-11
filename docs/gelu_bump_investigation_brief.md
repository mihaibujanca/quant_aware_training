# GELU Bump Investigation Brief

**Date:** 2026-04-01
**For:** New investigation agent
**Status:** Open question, contradictory findings, no clear answer

---

## The question

GPT-2 uses GELU activation in its FFN blocks. GELU differs from ReLU only near z=0 (the "bump" — a smooth transition instead of a hard kink). For z >> 0, GELU(z) ≈ z (same as ReLU). For z << 0, GELU(z) ≈ 0 (same as ReLU). The bump region is roughly |z| < 2.

**What does the bump compute? Why does the model need it?** We cannot currently answer this. 150+ cycles of experiments have produced contradictory-seeming findings that don't resolve into a coherent picture.

---

## The contradiction

Three lines of evidence appear to conflict:

### Line 1: The negative regime is massive but "just a bias"

- **79-91% of channel×token pairs have z < 0** (pre-activations are systematically negative, z_mean ≈ -0.76 to -1.18 across blocks)
- Each negative channel produces max |GELU(z)| = 0.17 — tiny individually
- But ~2800 channels aggregate through W_out into a vector ||y_neg|| that is **1-2.4× the total output norm** in blocks 2-7
- This aggregate is **87-98% token-independent** (cycle 109). It's the same vector regardless of input.
- The token-dependent part of the negative signal is **orthogonal to prediction-sensitive directions** (cycle 174/item 185)

Interpretation: the negative regime computes a fixed bias vector. Not a function of the input.

### Line 2: Removing the negative regime is catastrophic

- Full-model GELU→ReLU swap (zeroing all negative contributions): **+8320% PPL** (cycle 102). Model produces garbage.
- Per-block ReLU swap: ranges from +6% (late blocks) to +2664% (block 0) (cycle 102)

Interpretation: the negative regime is critical for the model to function.

### Line 3: Replacing negative with a bias works locally but fails globally

- Per-block: replacing negative regime with its mean μ_neg (768-dim constant per block) recovers within **+2% PPL for all 12 blocks** (cycle 173/item 184). The bias is a near-perfect replacement.
- Full-model (all 12 blocks simultaneously, precomputed biases): **+88% PPL** (cycle 181/item 191). Catastrophic.
- Full-model with iteratively recomputed biases (each block's bias computed on partially-swapped model): **+63% PPL** (cycle 182/item 192). Better, still broken.

Interpretation: the bias theory is locally correct but the composition across 12 blocks destroys it.

### Line 4: Positive channels carry "99.9% of computation"

- Killing positive channels (cycle 164/item 174): PPL 713,227 (model completely dead)
- Killing negative channels (cycle 102): PPL 1,874 (also dead, but 380× less dead)
- Positive regime ||y_pos|| >> ||y_neg|| in L2 norm

Interpretation: the positive regime does the "real" computation. But this is measured in L2 norm, which isn't functional importance.

---

## What we actually know (hard facts, not interpretations)

### Pre-activation statistics (cycles 108, 165)
- z = W_up @ x + b_up. The bias b_up contributes only -0.08 on average. The negativity comes from W_up @ x.
- The network LEARNED to anti-align W_up rows with the data distribution (cycle 165): 84-99% of rows have negative cosine with E[x]. Random init has z_mean ≈ 0; trained model has z_mean ≈ -1. This is deliberate.
- Block 1 is anomalous: tightest pre-activations (σ=0.52), 44% "bump-dominated" channels.

### Per-block ablation table (cycle 173/item 184)
All measured per-block in isolation (one block modified, others float), PPL metric:

| Condition | Description | Result (all 12 blocks) |
|-----------|-------------|----------------------|
| C0 | Float baseline | PPL = 22.7 |
| C1 | Kill negative (ReLU this block) | Within +6% for most blocks |
| C2 | Kill positive | Catastrophic (PPL 1000+) |
| C3 | Replace negative with mean | **Within +2% for ALL 12 blocks** |
| C4 | Replace positive with mean | Catastrophic |
| C5 | Both replaced with means | Catastrophic |

C3 is the key result: per-block, the negative regime is perfectly replaceable by a constant.

### Full-model composition (cycles 181-182)
- ReLU + precomputed bias (all 12 simultaneously): PPL 42.6 (+88%)
- ReLU + iteratively recomputed bias: PPL 37.0 (+63%)
- The progressive curve shows each additional block adds ~1-2 PPL of damage

### The negative regime is NOT uniformly negative per channel (cycle 178)
- 94-99% of channels are positive for SOME tokens. There aren't "permanently negative channels" — channels move between regimes depending on input.
- The "two populations" framing (bump-dominated vs linear) was per-token statistics averaged across tokens, not a fixed partition.

### Training dynamics (cycle 168/item 178)
- Random-init GPT-2: z_mean ≈ 0 (as expected)
- Trained GPT-2: z_mean ≈ -1
- The negativity emerges during training. The network deliberately shifts pre-activations into the GELU suppression zone.

---

## What's been tried and failed to explain

1. **"The negative regime is just a bias"** — True per-block, false globally. A constant per-block correction works in isolation but compounds catastrophically across 12 blocks.

2. **"The token-dependent negative signal matters"** — It's orthogonal to predictions (cycle 174). Replacing it with a constant loses nothing per-block.

3. **"Two fixed populations of channels"** — Not true. 94-99% of channels are sometimes positive. The partition is input-dependent, not structural.

4. **"The positive regime does all the computation"** — True in L2 norm, meaningless in functional importance. Both regimes are essential (removing either kills the model).

5. **"GELU→ReLU + bias should work"** — Works per-block, fails at composition. The remaining +63% after iterative bias correction is NOT a calibration problem — the experiment (cycle 182) was verified to correctly recompute biases on the actual inputs.

---

## The open question, precisely stated

The GELU bump (the smooth transition near z=0) has these properties:
1. Its aggregate contribution is mostly a constant bias (87-98%)
2. This constant bias is perfectly absorbable per-block (+2% PPL)
3. But the bias cannot be absorbed across 12 blocks simultaneously (+63% PPL)
4. The network deliberately trains W_up to place inputs in the bump zone

**Why does the per-block bias approximation fail when composed across blocks?**

Possible hypotheses (untested):
- The GELU derivative near z=0 (smooth 0→1) vs ReLU (discontinuous 0→1) changes how small perturbations propagate. Even with correct bias, the local Jacobian is wrong.
- The residual stream trajectory under GELU stays on a manifold that ReLU+bias doesn't maintain. The bias is correct at the start but the trajectory diverges.
- The 2% per-block error, while small, is STRUCTURED (not random) and compounds multiplicatively through attention, similar to the 41%/block cascade in FFN compression.

---

## Relevant experiment scripts and data

| Cycle | Item | Script | Result JSON |
|-------|------|--------|-------------|
| 102 | 107 | `experiments/gelu_relu_swap_gpt2.py` | `autoresearch/results/cycle_102_gelu_relu_swap.json` |
| 103 | 108 | `experiments/gelu_relu_anatomy_gpt2.py` | `autoresearch/results/cycle_103_gelu_relu_anatomy.json` |
| 109 | 120 | varies | `autoresearch/results/cycle_109_*.json` |
| 164 | 174 | varies | `autoresearch/results/cycle_164_*.json` |
| 165 | 175 | varies | `autoresearch/results/cycle_165_*.json` |
| 173 | 184 | varies | `autoresearch/results/cycle_173_*.json` |
| 174 | 185 | varies | `autoresearch/results/cycle_174_*.json` |
| 175 | 186 | varies | `autoresearch/results/cycle_175_*.json` |
| 178 | 188 | varies | cycle 178 result |
| 181 | 191 | `experiments/full_relu_bias_swap_cycle181.py` | `autoresearch/results/cycle_181_*.json` |
| 182 | 192 | `experiments/iterative_relu_bias_swap_cycle182.py` | `autoresearch/results/cycle_182_iterative_relu_swap.json` |

## Existing analysis documents
- `docs/gelu_nonlinear_zone_analysis.md` — structural analysis writeup (partially outdated by later findings)
- `autoresearch/surveys/gelu_relu_inference.md` — literature survey on GELU→ReLU at inference

## Key model details
- GPT-2 small (124M params, 12 layers, 768 hidden, 3072 FFN intermediate)
- Activation: NewGELUActivation (tanh approximation, max diff from exact GELU: 1.5e-4)
- Pre-activations z = W_up @ x + b_up where x is post-LayerNorm, post-attention residual stream
- FFN output: y = W_down @ GELU(z) + b_down, added to residual stream
