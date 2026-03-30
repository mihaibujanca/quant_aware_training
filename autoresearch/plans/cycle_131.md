# Cycle 131 Plan: Learned Bottleneck AE for FFN Compression — POC (Item 88)

## Task
Can a nonlinear encoder-decoder (E: R^3072→R^k, D: R^k→R^768) beat topk+lstsq at matched k=384 (8× compression) on single-block FFN swap? Blocks 4 and 7.

## Hypothesis
At 8× compression, axis-aligned topk+lstsq discards structural correlations between neurons. A learned bottleneck trained under model loss can exploit these correlations, yielding lower PPL.

## Design: BottleneckFFN Module
```
E(h) = ReLU(W_enc @ h + b_enc)      # W_enc ∈ R^{384×3072}
D(z) = W_dec @ z + b_dec             # W_dec ∈ R^{768×384}
Forward: x → frozen_c_fc(x) → GELU → E → D → + frozen_c_proj_bias
```
Single-layer encoder (no second linear). Frozen: c_fc weights+bias.
Params: encoder = 384×3072 + 384 = 1,180,032. Decoder = 768×384 + 768 = 295,680. Total = 1.48M.
topk+lstsq baseline: ~590K. Gap is 2.5×; if AE wins, follow-up tests param-matched variants.

**E_linear**: W_enc applied without ReLU, i.e. E_linear(h) = W_enc @ h + b_enc.
Nonlinearity metric: ||E(h) − E_linear(h)|| / ||E(h)|| on 1000 held-out samples.

## Seeds & Budget
Seeds: [42, 137, 7]. Per seed: 2 blocks × 2 methods × 1000 steps = 4000 steps.
Total: 3 × 4000 = 12000 steps. batch_size=4, seq_len=128. ~25 min on CPU.

## Steps
1. Build `experiments/bottleneck_ae_poc_cycle131.py`.
2. Load GPT-2 + OWT corpus (reuse `load_nanochat_teacher`, `build_probe_corpus`).
3. Float baseline: evaluate unmodified GPT-2 PPL (200 seqs). Must be ~57.6.
4. For each seed in [42, 137, 7], for each block in [4, 7]:
   a. **topk+lstsq**: `derive_ffn_compression` at k=384, swap, 500 Adam polish, eval PPL.
   b. **Bottleneck AE**: replace MLP with BottleneckFFN, freeze all except E+D, train 1000 steps (lr=1e-4, Adam, CE loss on next token), eval PPL.
   c. **Nonlinearity**: compute ||E(h) − E_linear(h)|| / ||E(h)|| on 1000 samples.
5. Report mean ± std across 3 seeds for each (block, method). Save JSON + table.

## Validation
- Float PPL must be ~57.6. If >60, abort.
- topk+lstsq single-block must degrade <20% over float. If not, use k=512.
- AE PPL must be < float PPL × 2. If not, training failed.

## Criteria
- **Confirmed** if AE outperforms topk+lstsq by ≥5% relative PPL (mean) on ≥1 block, outside 1-std overlap.
- **Refuted** if AE within 2% of topk+lstsq AND encoder nonlinearity < 5%.
- **Inconclusive** otherwise.

## Artifacts
- `experiments/bottleneck_ae_poc_cycle131.py`
- `autoresearch/results/cycle_131_bottleneck_ae.json`
