# Task
For each of GPT-2's 12 transformer blocks (b0-b11), zero out the FFN residual contribution and measure PPL. Rank blocks by importance (PPL delta). Identify blocks where |delta| < 0.5 PPL as candidates for skipping or aggressive compression.

# Hypothesis
At least 2 blocks have |PPL delta| < 0.5 when their FFN output is zeroed, shrinking the effective composition problem.
- **CONFIRMED**: >= 2 blocks with |delta| < 0.5 PPL across all 3 eval batches.
- **REFUTED**: <= 1 block meets the threshold.
- **INCONCLUSIVE**: threshold met in some batches but not others (batch variance > 0.5 PPL for borderline blocks).

# Pre-flight Results
- P1 PASS: Cycle 339 ran this on pgolf (not GPT-2). No past experiment zeros individual FFN contributions on GPT-2 and measures PPL delta per block. Cycle 243 restored float MLPs one-at-a-time in a compressed model (different: measures compression damage, not block importance).
- P2 PASS: No dependencies.
- P3 PASS: Inference-only, no training. 12 forward passes per eval batch + 1 baseline.
- P5: Null baseline = float GPT-2 PPL (~53.89). Delta = PPL_zeroed - PPL_float. A block is "prunable" only if |delta| < 0.5.
- P6 PASS: Zeroing FFN output before residual add is a clean architectural intervention (hook on `transformer.h[i].mlp`, set output to zero tensor).
- P7: Forward pass is deterministic. Use 3 disjoint eval batches (100 seqs each) for robustness. Report mean +/- std across batches.

# Steps
1. Create `experiments/gpt2_block_importance_item376.py`. Load GPT-2 + tokenizer, wikitext-2 test split. Prepare 3 disjoint eval batches of 100 sequences each (SEQ_LEN=128), from token offsets [0, 12800, 25600].
2. Measure float baseline PPL on each batch. Sanity check: must be within [52, 56] (ref 53.89).
3. For each block i in 0..11: register a forward hook on `model.transformer.h[i].mlp` that replaces the output with `torch.zeros_like(output)`. Measure PPL on all 3 batches. Remove hook. Store delta = PPL_zeroed - PPL_float per batch.
4. Rank blocks by mean |delta| ascending (most prunable first). Mark blocks with mean |delta| < 0.5 as "prunable". Check that std across batches < 0.5 for borderline blocks.
5. Cross-reference with cycle 366 error profile: blocks with high error growth (b10, b11) should have high importance. If they appear prunable, flag as suspicious.
6. Save results JSON to `autoresearch/results/cycle_367_block_importance.json` (per-block deltas, ranking, prunable list). Save bar chart (delta per block, threshold line at 0.5) to `autoresearch/results/cycle_367_block_importance.png`.
7. Print verdict (CONFIRMED/REFUTED/INCONCLUSIVE) based on criteria above.
8. Update `autoresearch/agenda.md`: move item 376 to PAST with outcome. Write `autoresearch/results/cycle_367_manifest.txt`. Git commit.

# Validation
- Float baseline PPL in [52, 56] (sanity).
- Sum of all deltas should be << PPL of zeroing ALL FFNs (interactions exist, but sum gives upper-bound sanity).
- b10/b11 expected high importance (cycle 366: error explosion). If either appears prunable, investigate.

# Artifacts
- `experiments/gpt2_block_importance_item376.py`
- `autoresearch/results/cycle_367_block_importance.json`
- `autoresearch/results/cycle_367_block_importance.png`
- `autoresearch/results/cycle_367_manifest.txt`

# Fallbacks
- If MPS OOM, reduce to 50 seqs per batch.
- If all blocks have |delta| > 0.5 (none prunable), report ranking anyway — useful for mixed-compression item 377.
- If batch variance is high (>1.0 PPL std), increase to 200 seqs per batch for a second run.
