# Task
Item 90: Mixed-corpus lstsq refit with matched sample sizes. Two conditions at n=2048: A'=Shakespeare-only, C=mixed (512 each from 4 corpora). Fix channels from Shakespeare. Eval on WikiText-2. Blocks [7,5,4,1] at 6x, 3 seeds.

# Hypothesis
- **Confirmed if**: mixed-corpus refit (C) outperforms Shakespeare-only (A') by >= 5% relative PPL at matched n=2048.
- **Refuted if**: Shakespeare-only (A') at n=2048 matches mixed within 3% relative PPL.
- **Inconclusive if**: improvement 3-5% or varies across seeds.

# Pre-flight
- P1 (data): `data/{shakespeare.txt,wikitext2_valid.txt,openwebtext_sample.txt,c4_sample.txt}` all exist.
- P2 (deps): `_select_hidden_units`, `_fit_output_projection`, `probe_ffn_geometry`, `swap_compressed_ffns`, `collect_hidden_activations` (from cycle 076 script). All validated in cycle 076.
- P3 (ground truth): Cycle 044 Shakespeare-only at 2048 → PPL ~135 (+94%). Condition A' here should be in the same ballpark. Cycle 076 condition C (mixed 2048) gave PPL ~113 (+61%).
- P4 (compute): 3 seeds x 2 conditions x 4 blocks. Probe geometry ~2min/seed. Total ~10min.
- P5 (MP3 check): At 6x compression, hidden_dim ~128 for block 7 (768/6=128). W_down lstsq has p=129 (128 + bias). With n=2048, ratio = 15.9x. Well above 2x threshold. SAFE.

# Steps
Script: `experiments/mixed_corpus_matched_refit_gpt2.py` (new file, adapted from cycle 076 script).

1. **Load GPT-2 + build corpora.** 3 seeds [42, 123, 7]. Load GPT-2 via `load_nanochat_teacher`. Build token tensors from each corpus file:
   - Shakespeare: tokenize, take first 2048 tokens (for condition A')
   - Per-corpus for mixed: tokenize each of {shakespeare, wikitext2, openwebtext, c4}, take 512 tokens each
   - WikiText-2 eval: 200 sequences of length 128

2. **Channel selection (fixed from Shakespeare).** Per seed: run `probe_ffn_geometry` on Shakespeare with `max_tokens_per_layer=2048`. Extract atlas. Channel indices per block come from `_select_hidden_units` with budget `hidden_fraction=1/6, selection_mode="topk"`. These indices are FIXED across both conditions.

3. **Condition A': Shakespeare-only lstsq refit at n=2048.** Per seed, per block: call `collect_hidden_activations(teacher, shakespeare_tokens, block_id, n_tokens=2048)`. Slice hidden to selected channels. Call `_fit_output_projection(hidden_sel, output, use_bias=True)`. Build `CompressedFfnPack` via `build_compressed_pack_from_refits` (reuse cycle 076 helper).

4. **Condition C: Mixed lstsq refit at n=2048.** Per seed, per block: call `collect_hidden_activations` for each of the 4 corpora with n_tokens=512. Concatenate hidden and output tensors (total 2048 tokens). Slice hidden to selected channels. Same `_fit_output_projection` and pack building.

5. **Evaluate.** For each condition x seed: `swap_compressed_ffns` then `_evaluate_model` on WikiText-2 validation. Record raw PPL, eval_loss, and relative degradation vs uncompressed baseline.

6. **Aggregate and verdict.** Compute mean/std PPL per condition across seeds. Relative improvement = (mean_A' - mean_C) / mean_A' * 100. Apply thresholds: >= 5% → confirmed, < 3% → refuted, 3-5% → inconclusive.

7. **Save.** JSON to `autoresearch/results/cycle_078_matched_refit.json`. Bar plot (2 conditions x 3 seeds) to `autoresearch/results/cycle_078_matched_refit.png`.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/mixed_corpus_matched_refit_gpt2.py`

# Validation
- WikiText-2 baseline PPL ~70 (consistent with cycles 044/070/076).
- Condition A' should give PPL ~135 (+94%), matching cycle 044 Shakespeare-only at 2048 tokens.
- Condition C should give PPL ~113 (+61%), matching cycle 076 condition C.
- If A' gives PPL >> 200 or PPL < 80, something is wrong — stop and investigate.
- Cross-seed std < 20% of mean per condition (tighter than cycle 076's 30% since sample size is now adequate).

# Artifacts
- `experiments/mixed_corpus_matched_refit_gpt2.py`
- `autoresearch/results/cycle_078_matched_refit.json`
- `autoresearch/results/cycle_078_matched_refit.png`

# Fallbacks
- If A' and C are within 3%: corpus diversity does NOT help at matched n. The cycle 076 "improvement" was purely sample-size. Report as refuted.
- If A' gives PPL ~135 but C gives ~113 (matching prior): genuine diversity benefit of ~16% relative. Report confirmed.
- If both conditions degrade equally to ~135: Shakespeare at 2048 is already well-determined, and mixing doesn't help beyond adequate sample size. Key insight for the pipeline.
