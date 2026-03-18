# Task
Item 21: Single-affine collapse quality on GPT-2 FFN blocks. Fit A, b per block on probe tokens, measure normalized residual r on held-out tokens, then swap in affine approximation for best blocks and measure perplexity.

# Hypothesis
- **Confirmed if**: at least 2 GPT-2 blocks achieve r < 0.15 AND swapping in the single-affine approximation causes <= 5% relative perplexity increase.
- **Refuted if**: all blocks have r > 0.5 or the swap always causes > 20% perplexity increase regardless of r.
- **Inconclusive if**: r is low but the perplexity increase is high, suggesting the affine approximation captures training-corpus behavior but not the function that matters for generation.

# Pre-flight Results
- P1: Not a duplicate. Cycle 009 measured Jacobian variance L (per-token J deviation from mean J), NOT affine-fit residual r. Cycle 010 measured r on toy ReLU, not GPT-2.
- P2: Dependencies satisfied — item 9 (confirmed cycle 004), item 20 (confirmed cycle 010).
- P3: Concrete — one script, clear metrics, clear criteria.
- P4: No program/prompt changes needed.

# Steps
1. Create `experiments/single_affine_collapse_gpt2.py` adapting `experiments/single_affine_collapse_toy.py` and the GPT-2 loading pattern from `experiments/jacobian_variance_atlas.py`.
2. Load GPT-2 and Shakespeare corpus using `load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))` and `build_probe_corpus(ProbeCorpusSpec(text_path="data/shakespeare.txt", sequence_length=128, probe_train_sequences=64, probe_holdout_sequences=32, eval_val_sequences=32))`.
3. For each of 12 FFN blocks, register a forward hook on `handle.mlp` to capture `(h_in, h_out)` pairs where `h_in = inputs[0]` and `h_out = output` (both flattened to `[N_tokens, 768]`). Collect 512 tokens from `probe_train` (fit split) and 512 from `probe_holdout` (held-out split).
4. Fit affine map: augmented lstsq `[h_in | 1] @ [A; b] = h_out` on fit split (512 tokens). Compute `r = ||h_out - A h_in - b||_F / ||h_out - mean(h_out)||_F` on both fit and held-out splits.
5. Report per-block r_fit, r_held. Identify the 2 blocks with lowest r_held.
6. **Perplexity swap test**: For each of the 2 best blocks, monkey-patch `handle.mlp.forward` to return `h_in @ A.T + b` (the fitted affine map). Evaluate full-model perplexity on `eval_val` split using the standard cross-entropy eval from `_evaluate_model` or manual NLL computation. Compute relative ppl increase = (ppl_swapped - ppl_baseline) / ppl_baseline.
7. Also test swapping both best blocks simultaneously to check accumulation.
8. Save JSON to `autoresearch/results/cycle_011_single_affine_gpt2.json` with per-block r values, baseline ppl, swap ppl per block, relative increases.
9. Save bar chart of r per block + annotated swap ppl to `autoresearch/results/cycle_011_single_affine_gpt2.png`.
10. Evaluate verdict per criteria. Write summary to `autoresearch/results/cycle_011_summary.md`.
11. Update `autoresearch/agenda.md`: move item 21 from NEXT to PAST EXPERIMENTS with commit hash and artifact paths.
12. Commit: `[autoresearch] cycle 011: single-affine collapse quality on GPT-2 FFN blocks`

# Validation
- Baseline perplexity should be ~30-40 on Shakespeare (sanity check GPT-2 is loaded correctly).
- r_fit <= r_held for all blocks (no overfitting to fit split; gap should be small).
- Swapping a block with identity (A=I, b=0) should give higher ppl than the fitted affine — sanity check.
- Compare cycle 009 L values with r values: explain any discrepancy (L measures Jacobian variation, r measures overall affine approximation quality including bias — they can diverge).

# Artifacts
- `experiments/single_affine_collapse_gpt2.py`
- `autoresearch/results/cycle_011_single_affine_gpt2.json`
- `autoresearch/results/cycle_011_single_affine_gpt2.png`
- `autoresearch/results/cycle_011_summary.md`

# Fallbacks
- If 512 tokens are insufficient for stable lstsq on 768-dim input (underdetermined), increase to 1024 tokens (8 more sequences) or use ridge regression (`torch.linalg.lstsq` with driver="gelsd" handles rank-deficiency, but add explicit Tikhonov regularization lambda=1e-4 if r_fit >> r_held).
- If all r > 0.5 (likely given cycle 009 refutation of linearity), still run the ppl swap test for the 2 lowest-r blocks to get the full inconclusive/refuted picture.
- If MPS OOM on hook collection, fall back to CPU (GPT-2 is small enough).
