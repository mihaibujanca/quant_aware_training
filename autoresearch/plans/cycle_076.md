# Task
Item 66: Fix channel selection from Shakespeare. Run lstsq W_down refit with (A) Shakespeare-only 512 tokens, (B) mixed 512 tokens (128 each from 4 corpora), (C) mixed 2048 tokens (512 each). Evaluate all on WikiText-2 validation PPL.

# Hypothesis
- **Confirmed if**: mixed refit (B or C) reduces WikiText-2 PPL degradation by >= 10pp relative vs Shakespeare-only (A).
- **Refuted if**: mixed refit performs worse than or equal to Shakespeare-only.
- **Inconclusive if**: improvement exists but < 5pp.

# Pre-flight
- P1 (data): `data/{shakespeare.txt,wikitext2_valid.txt,openwebtext_sample.txt,c4_sample.txt}` all exist.
- P2 (deps): `_select_hidden_units`, `_fit_output_projection`, `probe_ffn_geometry`, `swap_compressed_ffns` in `aleph/lm_geometric_compression.py`. All used in cycle 044.
- P3 (ground truth): Cycle 044 condition C uncorr_rel ~94-97% (Shakespeare->WikiText zero-shot). Condition D ~41% (with correction). This experiment measures raw compressed PPL (no correction).
- P4 (compute): 3 seeds x 3 conditions x 4 blocks. Probe geometry ~2min/seed. Lstsq refit trivial. Total ~15min.

# Steps
Script: `experiments/mixed_corpus_lstsq_refit_gpt2.py`

1. **Load GPT-2 + build corpora.** 3 seeds [42, 123, 7]. Build separate ProbeCorpus for each of {shakespeare, wikitext2, openwebtext, c4}. Also build WikiText-2 eval corpus (200 sequences).

2. **Channel selection (fixed from Shakespeare).** Per seed: `probe_ffn_geometry` on Shakespeare (512 tokens via probe_holdout). Extract selected_indices per block via `_select_hidden_units` with budget 1/6. These indices are FIXED for all conditions.

3. **Condition A: Shakespeare-only lstsq refit.** Use Shakespeare probe's hidden_sample/output_sample (512 tokens). For each block: slice hidden_sample to selected channels, call `_fit_output_projection`. Build CompressedFfnPack manually.

4. **Condition B: Mixed 512 tokens.** Per seed: collect 128-token probe samples from each of the 4 corpora (shakespeare, wikitext2, openwebtext, c4). Concatenate hidden_sample and output_sample. Lstsq refit on the mixed 512 tokens with same fixed channels.

5. **Condition C: Mixed 2048 tokens.** Same as B but 512 tokens per corpus (2048 total).

6. **Evaluate.** For each condition: `swap_compressed_ffns` then `_evaluate_model` on WikiText-2 validation. Record raw PPL and relative degradation vs uncompressed WikiText-2 baseline.

7. **Save.** JSON to `autoresearch/results/cycle_076_mixed_refit.json`. Bar plot (3 conditions x 3 seeds) to `autoresearch/results/cycle_076_mixed_refit.png`.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/mixed_corpus_lstsq_refit_gpt2.py`

# Validation
- WikiText-2 baseline PPL ~70 (consistent with cycle 044/070).
- Condition A degradation should be ~90-100% (matches cycle 044 condition C uncorrected).
- Cross-seed std < 30% of mean per condition.
- MP3 check: 512 tokens >> 128 params (W_down is 768 x ~128 at 6x, lstsq has 128 predictors). Safe.

# Artifacts
- `experiments/mixed_corpus_lstsq_refit_gpt2.py`
- `autoresearch/results/cycle_076_mixed_refit.json`
- `autoresearch/results/cycle_076_mixed_refit.png`

# Fallbacks
- If mixed 512 and 2048 show no difference: report as evidence that 512 mixed tokens suffice.
- If all conditions degrade equally: channel instability (Jaccard 0.34-0.58) dominates over refit corpus. Note this confirms item 64's finding.
