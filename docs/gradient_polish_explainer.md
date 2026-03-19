# Gradient Polish: Why Minimal SGD Fixes Analytic Compression Stacking

## Reading guide

This document builds up gradient polish from first principles in 6 steps. Each step depends on the previous one. If you get stuck on step N, re-read step N-1 — the gap is there.

---

## Step 1: What a single compressed FFN block does

A transformer FFN block is two linear transforms with GELU in between:

```
y = W_down @ GELU(W_up @ x + b_up) + b_down
```

Where W_up is (3072, 768) — expanding to hidden dim — and W_down is (768, 3072) — projecting back. The 3072-dim hidden layer is where the "work" happens: GELU activates different channels for different inputs.

**Compression via topk + lstsq:**
1. **topk**: measure which of the 3072 hidden channels contribute most across a probe corpus (by activation magnitude × output weight norm). Keep the top-k (e.g., 384 for 8x compression). This gives W_up_small (384, 768).
2. **lstsq**: the old W_down (768, 3072) no longer works because 2688 channels are gone. Solve for a new W_down_new (768, 384) via least-squares: minimize ||W_down_new @ h_small - y_float||² over the probe corpus, where h_small is the compressed hidden activations and y_float is the original block output. Closed-form solution, one matrix pseudo-inverse.

**Result**: a compressed block W_up_small (384, 768) + W_down_new (768, 384). When you swap this into the model in place of the original block, single-block PPL is sub-baseline (actually slightly better, likely a regularization effect).

**Key point**: the lstsq solution is *optimal for this block in isolation* — it minimizes reconstruction error assuming the block receives the exact float activations at its input.

---

## Step 2: Why stacking fails

Now compress 4 blocks (say blocks 7, 5, 4, 1 — the easiest ones). Each block's W_down was fitted assuming float inputs. But in the full model:

- Block 1 receives float input → produces slightly distorted output (the topk approximation isn't perfect)
- Block 4 receives input that has passed through distorted-block-1's output (via residual stream) → its W_down was fitted for clean input, not this
- Block 5 receives input distorted by both blocks 1 and 4
- Block 7 receives input distorted by all three

Each block's lstsq solution was calibrated for a world where all other blocks are perfect. None of them live in that world.

**Measured**: single-block = sub-baseline PPL. Stacked k=4 = +30% PPL.

The error compounds super-additively (cycle 058): each block's small error shifts the input distribution for downstream blocks, which were fitted for the original distribution, so they err more, shifting the next block's inputs even further.

---

## Step 3: Why cascaded re-probing doesn't fix it

The intuitive fix: compress blocks in inference order. After compressing block 1, re-run the probe data through the model with compressed block 1 active. Now block 4 sees the *actual* distorted inputs it will face at inference. Run lstsq for block 4 using these distorted activations. Repeat for blocks 5, 7.

**Result (cycle 060)**: +28.37% PPL vs +30.20% independent. Only 1.83pp improvement.

**Why it fails**: lstsq is still *per-block local*. Block 4's lstsq minimizes block 4's reconstruction error given distorted inputs. But it doesn't know what block 5 needs. Suppose block 4's lstsq introduces a small bias in direction v. If direction v happens to be exactly what block 5 amplifies, the cascaded approach makes things worse for block 5 while making block 4 better. Each block optimizes its own error without considering the chain.

Analogy: you're tuning 4 instruments in a quartet. Cascaded re-probing is like tuning each instrument to sound good with the previous one, sequentially. But instrument 2 doesn't know what instrument 4 needs. The quartet might still sound off.

---

## Step 4: What gradient polish actually does

After lstsq fitting all 4 blocks independently (the standard pipeline), add one more step:

1. **Freeze everything** in GPT-2 except the 4 compressed W_down matrices
2. **Forward pass**: run the full model (all 12 blocks, 4 of which are compressed) on held-out text
3. **Loss**: cross-entropy on next-token prediction — the model's actual training objective
4. **Backward pass**: gradients flow through the ENTIRE stacked model back to each W_down
5. **Adam update**: adjust the 4 W_down matrices (lr=1e-4)
6. **Repeat** for 500 steps on ~4K tokens

Total free parameters: 4 × 384 × 768 ≈ 1.2M (out of GPT-2's ~124M). Everything else is frozen.

**Result (cycle 061)**: +30.2% → +4.9% (independent polish) or +3.8% (joint polish). 500 steps.

---

## Step 5: Why it works so well — three reasons

### 5a: Global loss, not local reconstruction

lstsq minimizes per-block reconstruction: "how well does this block reproduce the float output?" Gradient polish minimizes the *model-level loss*: "how well does the full model predict the next token?" These are different objectives.

When the gradient flows backward through block 7 → block 5 → block 4 → block 1, each W_down receives a gradient that encodes: "here's how changing you affects the *final output* of the model, accounting for everything downstream." Block 4's gradient includes information about what blocks 5 and 7 need.

This is the fundamental advantage: each W_down is optimized in the context of the full chain, not in isolation.

### 5b: Excellent initialization

The lstsq solution is near-optimal per block. Adam doesn't need to find the right channels or discover the right output subspace — that's already done. It only needs to make small corrections to account for inter-block interactions.

This is why 500 steps suffice. If you initialized W_down randomly, you'd need orders of magnitude more steps. The lstsq warm-start means Adam operates in a tiny neighborhood of a good solution.

### 5c: Low-dimensional search space

Only 1.2M of 124M parameters are free (~1%). The optimization landscape is effectively low-dimensional and smooth (linear → GELU → linear, composed through residual connections). There are no difficult saddle points or sharp minima — just a smooth correction on top of a good starting point.

---

## Step 6: What this means and what we don't yet understand

### What it means

The compression pipeline is now:
1. **Analytic** (topk + lstsq): identifies the right channels and fits per-block output projections. Zero iteration, pure linear algebra. Gets you sub-baseline per-block.
2. **Polish** (500 Adam steps on W_down only): fixes inter-block error propagation. Gets stacking from +30% to ~+5%.

The analytic step does the geometric heavy lifting. The polish step handles the compositional dynamics that pure linear algebra can't capture (because they emerge from the nonlinear interaction of multiple blocks through the residual stream).

### What we don't understand yet

**Q1: What is Adam actually correcting?** The lstsq W_down matrices are optimal per-block. After polish, they're slightly different. What changed? Experiments 80 and 83 (Jacobian rank census and stacking error spectral decomposition) will answer this. Hypothesis: the lstsq error is concentrated in a few spectral directions of the Jacobian, and Adam learns to redirect those specific error modes away from directions that downstream blocks amplify.

**Q2: Is there an analytic version of polish?** If the error has low-rank spectral structure, it might be possible to compute the correction without Adam — e.g., by projecting the stacking error onto the Jacobian singular vectors and solving a small linear system for the correction. This would make the entire pipeline zero-iteration.

**Q3: How does polish scale?** 500 steps works for k=4 blocks. Does k=6 or k=8 need more steps? Does it scale linearly or super-linearly with the number of compressed blocks? Does it scale with model size (GPT-2-medium, GPT-2-large)?

**Q4: Why does joint polish outperform independent polish?** Joint polish (all 4 W_down updated simultaneously through the stacked forward pass) gives +3.8% vs independent polish (each W_down polished with only its own block compressed) at +4.9%. The 1.1pp gap confirms that inter-block coordination matters — but why is the gap so small? Is independent polish already capturing most of the inter-block correction indirectly (through the model loss)?

---

## Quick reference: key numbers

| Configuration | PPL degradation |
|---|---|
| Single block, 6x, lstsq | Sub-baseline (~-1%) |
| k=4 stacked, 6x, lstsq only | +30.2% |
| k=4 stacked, 6x, cascaded re-probing | +28.4% (barely helps) |
| k=4 stacked, 6x, 50 Adam steps | +18.4% |
| k=4 stacked, 6x, 500 Adam steps (independent) | +4.9% |
| k=4 stacked, 6x, 500 Adam steps (joint) | +3.8% |

Relevant cycles: 033, 057-063
