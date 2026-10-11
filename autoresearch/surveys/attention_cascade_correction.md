# Attention Cascade & Correction Placement — Literature Survey

**Last updated:** 2026-03-30
**Question:** How do compression errors propagate through attention, and where should corrections be placed?

---

## Methodology: per-layer isolation is mandatory

**Cumulative error measurement is misleading and must not be used for sensitivity analysis.** This was demonstrated in our own parameter-golf experiments (session 9f51a650): measuring RMSE of hidden states at each layer boundary showed layer 12 with 235 RMSE (vs ~23 at layer 11), suggesting catastrophic sensitivity. But when layer 12 was quantized in isolation (everything else float), its actual output impact was +0.0004 bpt — the cheapest layer. The 235 RMSE was entirely propagated error from earlier layers.

**Required methodology for all cascade/sensitivity experiments in this project:**

1. **Per-component isolation**: Quantize/compress exactly ONE component (one layer's attention, one layer's MLP, one specific projection matrix) while keeping everything else at float precision. Measure output metric (PPL, BPB) degradation. This gives the TRUE local sensitivity, free of propagation confounds.

2. **Decomposition must be additive-tested**: After measuring components in isolation, measure them together and compute the interaction term: `interaction = error(A+B) - error(A) - error(B)`. If interaction is large (>20% of combined), the components are coupled and cannot be analyzed independently.

3. **Our cycle 123 (QK vs V decomposition) used this correctly**: held attention patterns fixed while swapping V (and vice versa), measured interaction term explicitly.

### What varies across architectures and training regimes

Our parameter-golf model (31.8M, XSA + LeakyReLU² + partial RoPE) showed:
- Attention uniformly cheap to quantize at int4 (most layers +0.001 bpt or less)
- MLP sensitivity varies 10x across layers (L0 MLP: +0.005, L2 MLP: +0.0002)
- Mixed precision (int4 attn + int5 MLP) + 3 correction layers = 100% float recovery

Our GPT-2 experiments (cycle 123) showed:
- QK and V contribute roughly equally (neither dominates in any layer)
- This is fundamentally different from the parameter-golf result

**These are different architectures with different activations (LeakyReLU² vs GELU), different attention (XSA vs standard), and different training regimes.** The sensitivity profile is NOT a universal property of transformers — it depends on:
- Activation function (ReLU variants vs GELU vs SiLU)
- Attention variant (standard, XSA, GQA, MQA)
- Training data and duration (how well-trained the model is)
- Model scale (outlier patterns change with width/depth)

### What IS and ISN'T validated by our experiments

**IS validated (methodology):**
- Per-component isolation as the correct measurement approach
- The decomposition framework (QK vs V, local vs propagated, metric vs topological)
- That correction placement matters and should be empirically determined per model
- That mixed-precision allocation guided by per-component sensitivity outperforms uniform

**IS NOT validated (cannot generalize):**
- Any specific sensitivity ordering (e.g., "attention is always cheaper than MLP")
- Any specific correction placement (e.g., "correct after FFN, not after attention")
- Any specific rank budget (e.g., "rank-8 suffices")
- That results on GPT-2 transfer to Mixtral, LLaMA, or any other architecture

**The contribution is the methodology, not the specific numbers.** Each new architecture/training regime needs its own per-component sensitivity profile before designing a compression or correction strategy.

---

## The key gap

**Nobody has decomposed attention block error into QK-space (pattern change) vs V-space (value change).** This is the central measurement our experiments (items 140-144) will make. Cycle 123 made this measurement on GPT-2 FFN compression and found neither dominates — but this may differ for other architectures and compression methods.

## What is known

### Structural sensitivity hierarchy
- **Basu 2603.20991** (March 2026): Maps sensitivity across 5 architectures (117M-8B). **Value projections compress nearly for free. Early-layer MLP up-projections are catastrophically sensitive.** Residual connections contract errors by growing hidden state faster than error.
- **RILQ 2412.01129** (AAAI 2025): FFN up-projection is rank-critical for error correction. Q-projection is rank-redundant. Model-level loss implicitly routes rank budget to critical modules.

### Keys more sensitive than values
- **KV-AdaQuant 2502.15075**: Key matrices have higher norms, more outlier-prone than values. Proposes asymmetric bit allocation (more bits for K). This is for KV cache (activations), but sensitivity ordering likely transfers to weights.
- **Dimensional collapse 2508.16929**: Attention outputs confined to ~60% effective rank due to W_O anisotropy. W_V and W_O are consistently lower-dimensional than W_Q and W_K — V-space has natural compression headroom QK-space does not.

### Attention pattern distortion
- **EdgeQAT 2402.10787**: "Performance drop of quantization primarily stems from information distortion in quantized attention maps." Maximizes entropy of quantized Q/K distributions and optimizes cosine similarity between quantized and FP16 attention maps. Closest to explicitly measuring pattern distortion.
- **BinaryAttention 2603.09582** (CVPR 2026): 1-bit QK attention retains only sign of Q and K, adds learnable bias. Matches full-precision on ViT/DiT. Shows QK similarity structure survives extreme quantization with proper correction.
- **Bondarenko 2306.12929** (NeurIPS 2023): Outliers in activations CAUSED by attention heads learning no-ops. To produce zeros in attention matrix, model pushes softmax inputs to extremes → creates outliers elsewhere. The QK mechanism is the SOURCE of the outlier problem.

### Error propagation through residual stream
- **QEP 2504.09629** (NeurIPS 2025): Error grows near-exponentially within quantized blocks. Closed-form weight adjustment compensates accumulated upstream error. Plug-and-play with GPTQ/AWQ/QuIP.
- **Our cycle 089**: 41%/block super-multiplicative compounding, R²=0.933.
- **Our cycle 104**: Attention accounts for ≥70% of cascade in 8/11 block transitions. GELU actually attenuates.

### Attention recovery via knowledge distillation
- **Kim & Lee 2211.11014** (EMNLP 2022): MSE on attention scores is insufficient for recovery. Attention-map loss and attention-output loss are better distillation objectives. Implies QK-space and V-space need separate treatment.

### Correction placement strategies (from literature)
1. **Per-weight-matrix additive** (CALDERA, LoftQ, LQER, INT2.1): W = Q + LR on ALL projections uniformly. No placement analysis.
2. **Pre-quantization rotation** (QuaRot, SpinQuant, ButterflyQuant): Change coordinate system so quantization is easier. Not corrections per se.
3. **Cross-layer propagation** (QEP, GPTQ): Re-compute with actual quantized activations as input to next block.
4. **LoRA for recovery** (original LoRA ablation): W_q+W_v is the sweet spot. W_q alone insufficient.
5. **Our approach** (cycles 115-116): Rank-r on residual stream after each compressed FFN.

### What nobody has tried
- Correction specifically for attention pattern recovery (separate from value recovery)
- U-Net style before+after attention corrections
- Placement ablation for compression error correction (as opposed to fine-tuning)

### Internal results (this project)

| Cycle | Finding | Architecture | Generalizes? |
|-------|---------|-------------|-------------|
| 089 | 41%/block cascade, super-multiplicative | GPT-2 (GELU, 12L) | Unknown for other architectures |
| 104 | Attention drives 70%+ of cascade | GPT-2 (GELU, 12L) | Contradicts parameter-golf (attn cheap) — architecture-dependent |
| 113 | Amplification uniform across heads | GPT-2 (GELU, 12L) | May differ for GQA/MQA |
| 115 | Rank-8 corrections recover 74% at k=4 | GPT-2 (GELU, 12L) | Rank budget likely model-dependent |
| 123 | QK and V contribute equally to cascade | GPT-2 (GELU, 12L) | Different from parameter-golf (attn uniformly cheap) |
| param-golf | Attn int4, MLP int5, 3 corrections = 100% recovery | Custom (XSA, LeakyReLU², 13L) | Architecture-specific |

## References

- Parameter-golf session 9f51a650 — internal, per-layer isolation methodology for mixed-precision quantization
- Basu, "Structural Sensitivity in Compressed Transformers" — arXiv:2603.20991
- RILQ, "Rank-Informed Low-Rank Quantization" — arXiv:2412.01129
- KV-AdaQuant, "More for Keys, Less for Values" — arXiv:2502.15075
- Dimensional collapse in attention outputs — arXiv:2508.16929
- EdgeQAT — arXiv:2402.10787
- BinaryAttention — arXiv:2603.09582
- Bondarenko et al., "Quantizable Transformers" — arXiv:2306.12929
- QEP, "Quantization Error Propagation" — arXiv:2504.09629
- Kim & Lee, "KD for Attention Recovery" — arXiv:2211.11014
- CALDERA — arXiv:2405.18886
- SpinQuant — arXiv:2405.16406
- QuaRot — arXiv:2404.00456
- ButterflyQuant — arXiv:2509.09679
- LQER — arXiv:2402.02446
- Low-Rank Correction for Quantized LLMs — arXiv:2412.07902
- LoRA — arXiv:2106.09685
