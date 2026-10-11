# GELU→ReLU at Inference: Literature Survey

**Last updated:** 2026-03-29
**Question:** Does GELU's smooth transition matter at inference, or is it just a training dynamics feature? If the latter, we can swap to ReLU and recover discrete piecewise-linear structure (MASO theory applies).

---

## Key finding from the literature

**GELU's smoothness is primarily a training feature, not an inference feature.** Multiple independent lines of evidence:

1. **ReLU's Revival** (NeurIPS 2024, arXiv:2410.09637): With LayerNorm, GELU vs ReLU trained from scratch differs by only 2.5% PPL on GPT-2. Without LayerNorm, ReLU actually wins by 8.2%. LayerNorm does the same job as GELU's smooth gradient.

2. **SUGAR** (May 2025, arXiv:2505.22074): Uses GELU-like gradients during backprop but ReLU during forward pass. Matches or beats GELU on Swin Transformer. Strongest evidence: forward-pass smoothness is not what matters.

3. **ReLU Strikes Back** (Apple, ICLR 2024 Oral, arXiv:2310.04564): "Relufication" of LLaMA-7B/13B/70B. With 30B tokens finetuning (~3-5% of original training budget), gap closes to 1-2% accuracy. Also achieves 90%+ activation sparsity → 32% FLOPS reduction.

4. **AERO** (Oct 2024, arXiv:2410.13060): Removes GELU entirely for private inference. In softmax-only architectures, GELU causes training collapse. Shows smooth transition is a training dynamics feature.

## The unmeasured gap

**Nobody has published the raw zero-shot GELU→ReLU swap on a pretrained model without any finetuning.** Every paper either trains from scratch with ReLU or finetunes after swapping. The raw swap damage is genuinely unknown. This is our experiment 107.

## Why this matters for geometric compression

If GELU→ReLU swap is cheap (< +10% PPL):
- **MASO theory applies**: ReLU gives discrete binary activation patterns, not continuous GELU' values
- **Piecewise-linear structure becomes available**: finite partition, affine map per region, mergeable
- **Activation sparsity is free compression**: 90%+ sparsity in some layers (ReLU Strikes Back)
- Cycle 085's "patterns are maximally diverse" finding becomes irrelevant (that was GELU smoothness, not model structure)

If the swap is expensive (> +20% PPL):
- Tells us exactly which tokens/channels use the GELU bump at inference
- Those are the "interesting" neurons — the ones where the smooth transition carries task-relevant information
- Maps the inference-critical nonlinear structure of the FFN

## Available pretrained ReLU models

| Model | Activation | Base | Size | Source |
|-------|-----------|------|------|--------|
| ReluLLaMA-7B | ReLU | LLaMA-2 7B | 7B | `SparseLLM/ReluLLaMA-7B` |
| ReluLLaMA-13B | ReLU | LLaMA-2 13B | 13B | `SparseLLM/ReluLLaMA-13B` |
| ReluLLaMA-70B | ReLU | LLaMA-2 70B | 70B | `SparseLLM/ReluLLaMA-70B` |
| ProSparse-LLaMA-2-7B | ReLU (89% sparsity) | LLaMA-2 7B | 7B | `SparseLLM/prosparse-llama-2-7b` |
| TurboSparse-Mistral | dReLU | Mistral 7B | 7B | `PowerInfer/TurboSparse-Mistral-Instruct` |

No ReLU GPT-2 exists pretrained. For GPT-2 scale: raw-swap or train from scratch (cheap).

No SUGAR models released — SUGAR is a training technique (`torch.autograd.Function` with ReLU forward, smooth surrogate backward), not a model. Implementation is trivial but no pretrained checkpoints.

## Related: piecewise-linear GELU approximations (hardware-oriented)

- **DAPA** (DAC 2026, arXiv:2603.19338): Distribution-aware piecewise approximation, focuses segments on high-probability input regions
- **NPLA** (IEEE 2023): Non-uniform piecewise linear lookup tables for FPGA/ASIC
- **ISPA** (IEEE 2025): Exploits GELU symmetry, 16 segments → MSE 4.29e-9

These are hardware optimizations, not relevant to our software compression work.

## References

- Mirzadeh et al. "ReLU Strikes Back: Exploiting Activation Sparsity in LLMs" (ICLR 2024 Oral) — arXiv:2310.04564
- Bick et al. "ReLU's Revival: On the Entropic Overload in Normalization-Free LLMs" (NeurIPS 2024) — arXiv:2410.09637
- Nrusimha et al. "AERO: Softmax-Only LLMs for Efficient Private Inference" — arXiv:2410.13060
- Song et al. "TurboSparse: Achieving LLM SOTA Performance with Minimal Activated Parameters" — arXiv:2406.05955
- Zhang et al. "ReLU-squared Wins" — arXiv:2402.03804
- Heo et al. "The Resurrection of the ReLU" (SUGAR) — arXiv:2505.22074
- Xu et al. "Faster Inference of Integer SWIN Transformer by Removing GELU" — arXiv:2402.01169
