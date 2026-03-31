# Cycle 169 Plan: Bias Shift Under Compression (Item 179)

## Pre-flight
- P1 (duplicate): no prior cycle tested b_leak stability under compression. Cycle 166 tested b_leak on float model only.
- P2 (deps): none. Uses existing compression pipeline + cycle 166's b_leak methodology.
- P3 (scope): 12 blocks × 2 conditions (float, compressed) + 1 PPL fix eval. ~15 min.
- P4 (program.md): no modifications needed.

## Task
Measure whether topk+lstsq compression shifts the constant bias b_leak = E[y_neg] at each FFN block, and if so, whether fixing b_leak to float values reduces the PPL cascade.

## Hypothesis
Compressed upstream blocks produce different hidden activations → different GELU negative patterns → b_leak shifts. If the shift is large and contributes to cascade, fixing it is a cheap correction (768 params/block).

## Steps
1. Load GPT-2, tokenizer, 200 OWT sequences (seed 42). Verify baseline PPL in [15, 65].
2. Compute b_leak_float per block: hook mlp.act, record y_neg = gelu(z) where z <= 0, average over 4096 tokens. Shape (3072,) per block.
3. Compress k=4 blocks [7,5,4,1] at 6x with topk+lstsq (standard pipeline: probe_ffn_geometry → derive_ffn_compression → swap_compressed_ffns). Verify compressed PPL.
4. Compute b_leak_compressed per block: same hooks on compressed model, same 4096 tokens.
5. Per-block relative shift: ||b_leak_compressed - b_leak_float|| / ||b_leak_float||.
6. If any block has shift > 20%: for each such block, add correction = b_leak_float - b_leak_compressed as a fixed bias after activation. Measure PPL with correction vs without.
7. Save JSON results + shift bar plot.

## Validation
- Baseline PPL must be in [15, 65].
- Compressed PPL should be ~70 (consistent with prior k=4 results).
- b_leak must have shape (3072,) and negative mean (consistent with cycle 166: mean ≈ -0.09).
- For uncompressed blocks, b_leak_compressed should be very close to b_leak_float (sanity check).

## Criteria
- **Confirmed**: shift > 20% AND fixing reduces cascade by >= 10%.
- **Refuted**: shift < 5%.
- **Inconclusive**: shift exists but fixing doesn't help.

## Artifacts
- `experiments/bias_shift_compression_cycle169.py`
- `autoresearch/results/cycle_169_bias_shift.json`
- `autoresearch/results/cycle_169_bias_shift.png`

## Fallbacks
- If compression pipeline OOMs: reduce probe tokens to 2048.
- If b_leak shift is bimodal (some blocks high, some low): report block-by-block.
- Note: compressed blocks have different d_ff (512 vs 3072). b_leak for compressed blocks is measured on the FULL model output (post W_down), not on compressed hidden. The hook on mlp.act sees whatever the compressed MLP produces.

## Key subtlety
For compressed blocks, the MLP module is replaced with CompressedMLP which has a different hidden dim (512 vs 3072). The activation is still GELU but operates on 512-dim hidden. The b_leak hook must handle both original (3072) and compressed (512) hidden dims. However, the b_leak that matters is the contribution to the block output (768-dim), so we measure b_leak AFTER W_down projection, not at the hidden layer.

Wait — actually cycle 166 measured b_leak at the activation output (3072-dim), which then goes through W_down. For this cycle, we should measure the EFFECT on block output: b_leak_effect = W_down @ b_leak_hidden. This makes compressed and uncompressed blocks comparable (both 768-dim). Revised: hook the full MLP module, compute y_neg contribution to MLP output.
