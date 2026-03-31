# Cycle 170 — Literature Scan: FFN Compression 2025-2026

## Our baseline
- **topk+lstsq** (no corrections): PPL 70.7 on WikiText-2, GPT-2 (124M), 6x FFN compression (k=4 of 24 channel groups, 4 of 12 blocks), calibration-only, ~1.58M params
- **topk+lstsq + corrections**: PPL ~59, +49K correction params (~1.6M total)

## Annotated Method Table

| # | Method | Paper | Year/Venue | Approach | Training? | FFN-only? | Compression | PPL (WikiText-2) | Baseline PPL | PPL degradation | Scale |
|---|--------|-------|-----------|----------|-----------|-----------|-------------|-------------------|-------------|-----------------|-------|
| 1 | **FLAP** | An et al., 2312.11983 | AAAI 2024 | Fluctuation-based importance metric, adaptive per-layer sparsity, bias compensation. Structured neuron pruning. | No (calibration-only) | Yes (prunes FFN neurons + attn heads) | 20% params removed | 14.62 | 12.62 | +16% | LLaMA-7B |
| 2 | **SliceGPT** | Ashkboos et al., 2401.15024 | ICLR 2024 | PCA rotation absorbs into weights, then delete rows/columns to shrink embedding dim. | No (calibration-only) | No (shrinks all weight matrices) | 25% params removed | ~6.3 (est.) | 5.68 | ~11% | LLaMA-2-7B |
| 3 | **SVD-LLM V2** | Wang et al., 2403.07378 / 2503.12340 | ICLR 2025 / NAACL 2025 | Truncation-aware SVD with data whitening. Per-layer rank allocation. | No (calibration-only) | Yes (applied per weight matrix, can target FFN only) | 20% params removed | 7.12 | 5.68 | +25% | LLaMA-7B |
| 4 | **SoLA** | AAAI 2025 | AAAI 2025 | Identifies high-activation FFN components, preserves them full-rank, SVD-compresses the rest. Adaptive rank allocation. | No (calibration-only) | Yes (targets FFN activation sparsity) | 30% params removed | 4.44 | ~3.56 | +25% | LLaMA-2-70B |
| 5 | **2SSP** | Sandri & Cunegatti, 2501.17771 | TMLR 2025 | Two-stage: (1) width pruning of FFN neurons by output-magnitude importance, (2) depth pruning of attention submodules. | No (calibration-only) | Stage 1 is FFN-only; stage 2 prunes attention | 25-50% | Best at 25% across 4 LLM families | varies | outperforms 5 SOTA competitors | LLaMA-2-7B/13B, Mistral, Qwen |
| 6 | **PruneNet** | ICLR 2025 | ICLR 2025 | Policy network learns which FFN dimensions to keep via spectral distribution shift (KS-distance). Calibration-free. | No (calibration-free!) | Yes (only prunes FFN layers) | 20-30% | 80-95% zero-shot retention | - | - | LLaMA-2-7B |
| 7 | **BlockPruner** | ACL Findings 2025 | ACL 2025 | Segments transformer into MHA+MLP submodules, prunes by perplexity-based importance. | No (calibration-only) | Can target MLP submodules specifically | 25% | 12.09 (vs EvoPress 10.37) | 5.47 | +121% | LLaMA-2-7B |
| 8 | **CFSP** | Wang & Ma, 2409.13199 | COLING 2025 | Coarse-to-fine: allocate sparsity budget across blocks by importance, then prune within blocks. Recovery fine-tuning optional. | Optional fine-tuning | Yes (FFN neurons) | 20-50% | outperforms FLAP, LLM-Pruner | varies | - | LLaMA-3-8B/70B |
| 9 | **StructuredFFN** | NeurIPS 2024 | NeurIPS 2024 | Block-diagonal + low-rank parameterization of FFN. Self-guided training from init. | Yes (train from scratch) | Yes (FFN-only) | 32% FFN params (68% removed) | 13.81 | 12.46 | +10.8% | 1.3B Transformer |
| 10 | **Spark Transformer** | NeurIPS 2025 | NeurIPS 2025 | Learned top-k predictor identifies 8% of FFN neurons to activate. Statistical top-k avoids sorting. | Yes (train from scratch) | Yes (FFN activation sparsity) | 92% FFN sparsity (8% active) | competitive with dense | Gemma-2 baseline | ~neutral | Gemma-2 scale |
| 11 | **LLM-BIP** | Dec 2024 | arxiv 2024 | Block-wise forward importance propagation for structured pruning of attn heads + FFN channels. Single forward pass. | No (calibration-only) | Yes (prunes FFN channels + attn heads) | 20% | reduces PPL by 14.09 vs Wanda | - | - | LLaMA-7B |

## Gap Analysis vs Our Recipe

### Methods directly comparable to our approach (calibration-only, FFN-focused, structured):

1. **FLAP** — Most directly comparable. Structured neuron pruning with bias compensation. At 20% removal on LLaMA-7B: +16% PPL. Our recipe at ~75% FFN removal in 4 blocks (effectively ~25% total FFN params removed model-wide) achieves PPL 70.7 (vs baseline ~30 for GPT-2, so +136%). FLAP is much more conservative but also much less aggressive. **NOT TRIED in this repo.**

2. **SoLA** — Activation-aware split: keep high-activation neurons full-rank, SVD the rest. This is conceptually close to our topk selection but uses a split (preserve+compress) rather than prune+reconstruct. The key insight (activation magnitude identifies important neurons) is essentially what our topk does, but SoLA's two-tier approach (full-rank top + low-rank bottom) is different from our topk+lstsq (keep top + fit). **NOT TRIED. Worth testing the split variant.**

3. **2SSP** (width+depth combined) — Stage 1 (FFN neuron pruning by output magnitude) is closest to our approach. The key difference: they combine with depth pruning in stage 2 (dropping entire attention submodules). We haven't tried combining FFN compression with attention layer dropping. **PARTIALLY NEW. The combined width+depth idea is untried.**

4. **PruneNet** — FFN-only, calibration-free. Uses spectral shape preservation (KS-distance on singular value distribution) rather than activation-based importance. This is a fundamentally different selection criterion from our magnitude/activation-based topk. **NOT TRIED. Novel selection criterion.**

5. **SVD-LLM V2** — Low-rank factorization with data whitening. We've explored low-rank in cycle 1 (Exp 1) but on toy models, not on GPT-2 FFN with truncation-aware whitening. The whitening step (rotate weight matrices to align truncation loss with singular values) is the key novelty. **PARTIALLY TRIED (SVD explored but not whitened SVD on GPT-2).**

6. **CFSP** — Coarse-to-fine budget allocation is essentially our per-block importance scoring + adaptive compression ratio. We tried block-adaptive ratio in cycle 160 (refuted as duplicate of cycles 94/50). **ALREADY TRIED (or equivalent).**

7. **StructuredFFN / Spark Transformer** — Training-from-scratch methods. Outside our problem scope (post-training compression). **NOT APPLICABLE.**

### Methods already tried or equivalent:
- Channel selection by importance/magnitude — tried (cycles 141, 143, 161)
- Block-adaptive compression ratio — tried and refuted (cycles 94, 50, 160)
- Low-rank on toy models — tried (Exp 1)
- Depth pruning (removing layers) — not tried but not our scope (FFN-only compression)

### Two untried methods worth testing:

**1. SoLA-style split: preserve top-k full-rank + SVD the rest (instead of dropping)**
- Instead of topk+lstsq (keep k channels, fit), keep k channels unchanged AND low-rank compress the remaining (24-k) channels. This preserves more information than hard pruning.
- Expected benefit: better reconstruction quality at same effective parameter count.
- Quick test: k=4 full-rank + rank-r SVD on remaining 20 groups, compare PPL to topk+lstsq at same total params.

**2. PruneNet-style spectral selection criterion (KS-distance on singular values)**
- Instead of selecting channels by activation magnitude (topk), select by which channels' removal least distorts the singular value distribution of the FFN weight matrix.
- This is a weight-only criterion (no calibration data needed).
- Quick test: compare PruneNet's spectral criterion vs topk for channel selection on GPT-2 FFN.

## Verdict: CONFIRMED

Found >=2 methods not yet tried that are comparable or better:
1. **SoLA split** (preserve top + SVD rest) — different from our prune+reconstruct
2. **PruneNet spectral selection** — different criterion from our topk
Both are calibration-only, FFN-focused, and achievable at GPT-2 scale.

## Sources
- FLAP: https://arxiv.org/abs/2312.11983
- SliceGPT: https://arxiv.org/abs/2401.15024
- SVD-LLM: https://arxiv.org/abs/2403.07378
- SVD-LLM V2: https://arxiv.org/abs/2503.12340
- SoLA: https://ojs.aaai.org/index.php/AAAI/article/view/33923
- 2SSP: https://arxiv.org/abs/2501.17771
- PruneNet: https://arxiv.org/abs/2501.15296
- BlockPruner: https://arxiv.org/abs/2406.10594
- CFSP: https://arxiv.org/abs/2409.13199
- StructuredFFN: https://proceedings.neurips.cc/paper_files/paper/2024/file/0877af85978e9e630b77f6221db47876-Paper-Conference.pdf
- Spark Transformer: https://arxiv.org/abs/2506.06644
- LLM-BIP: https://arxiv.org/abs/2412.06419
