# Cycle 102 Plan: Zero-shot GELU→ReLU Swap — GPT-2

**Item:** 107
**Script:** `experiments/gelu_relu_swap_gpt2.py` (new)

## Pre-flights

**P1 -- Null hypothesis:** GELU's smooth bump carries task-critical information at inference; replacing it with ReLU destroys >30% relative PPL.

**P2 -- Criteria (verbatim from agenda):**
- Confirmed if: full-model swap PPL degradation <= +10% relative (GELU bump is dispensable at inference).
- Refuted if: full-model swap PPL degradation > +30% relative (GELU bump is critical at inference).
- Inconclusive if: +10-30% (some blocks tolerate swap, others don't).

**P3 -- Novelty vs replication:** Survey `gelu_relu_inference.md` shows 4 papers that train-from-scratch or finetune after swap. Raw zero-shot swap on pretrained GPT-2 is unpublished.

**P4 -- Confounds:** (1) HF GPT-2 uses `NewGELU` (tanh approx), not `nn.GELU()` — must patch the correct class. (2) Stride length and context window must match prior cycles for comparable baselines.

**P5 -- Cost:** 1 baseline eval + 1 full-swap eval + 12 per-block evals × 3 seeds = 42 evals, ~200 seqs each. ~30 min on MPS.

**P6 -- Data:** OWT eval split (`data/openwebtext_sample.txt`) + WikiText-2 (HF `wikitext-2-raw-v1`). Both used in prior cycles.

**P7 -- Abort:** Baseline PPL outside [25, 65] on OWT. HF tokenizer/model loading fails.

## Implementation

1. **Baseline.** Load `gpt2` from HF, eval PPL on OWT (200 seqs) and WikiText-2 (full test), 3 seeds [42, 123, 7]. Verify OWT ~51.66.

2. **Full-model swap.** Find all GELU modules (inspect `model.transformer.h[i].mlp.act` — GPT-2 uses `NewGELU`). Replace each with `nn.ReLU()`. Eval PPL on both datasets, 3 seeds. Report mean ± std and relative delta.

3. **Per-block swap.** For each block i in 0..11: swap ONLY block i's GELU→ReLU, keep all others as GELU. Eval PPL on OWT, 3 seeds. Produces 12-row sensitivity table.

4. **Activation sparsity.** On the full-swap model, hook each block's post-activation output. Compute sparsity = fraction of zeros per block over 200 seqs. Report 12-row table.

5. **Correlation with compression difficulty.** Load `cycle_048_local_complexity_gpt2.json` (P1_frob_std per block). Compute Spearman correlation between per-block swap damage (step 3) and P1_frob_std. Report rho and p-value.

6. **Verdict logic.**
   ```
   delta_full = (ppl_swap - ppl_base) / ppl_base * 100  # on OWT, mean over seeds
   if delta_full <= 10.0:   CONFIRMED
   elif delta_full > 30.0:  REFUTED
   else:                    INCONCLUSIVE
   ```

7. **Artifacts.** Save `autoresearch/results/cycle_102_gelu_relu_swap.json` (all PPLs, sparsities, correlations) and `autoresearch/results/cycle_102_gelu_relu_swap.png` (2-panel: per-block sensitivity bar chart + sparsity bar chart).

8. **Commit & agenda.** Update item 107 in `autoresearch/agenda.md` with outcome, cycle number, artifact paths. Commit with `[autoresearch] cycle 102: GELU→ReLU zero-shot swap — GPT-2`.
