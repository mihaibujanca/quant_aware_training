# Cycle 087 — Mixed-corpus stacking evaluation — GPT-2 FFN

Item 96. Type: empirical. Depends on: 90 (confirmed cycle 078).

## Criteria (verbatim)
- **Confirmed if**: mixed-corpus reduces stacked PPL by ≥10% relative vs OWT-only on WikiText-2, AND does not degrade OWT by >3%.
- **Refuted if**: mixed-corpus degrades stacked OWT PPL by >5%.
- **Inconclusive if**: helps WikiText-2 but degrades OWT by 3–5%.

## Design
Two conditions, both stacking blocks [7,5,4,1] at 6× (k=64):
- **(A) OWT-only**: probe + refit from OWT (2048 tokens, standard pipeline)
- **(B) Mixed-corpus**: refit from 512 tokens each of Shakespeare, WikiText-2, OWT, C4 (2048 total)

Channel selection: from OWT for condition A, from OWT for condition B too (match probe source).
Refit: lstsq via `_fit_output_projection`. Then +500 Adam steps (lr=1e-4, W_down only, joint).
Seeds: [42, 123, 7]. Eval: OWT (200 seqs) AND WikiText-2 (200 seqs).

## Implementation
New file: `experiments/mixed_corpus_stacking_gpt2.py`

Structure (merge stacking pipeline from `gradient_polish_scaling_gpt2.py` with mixed-corpus
activation collection from `mixed_corpus_matched_refit_gpt2.py`):

1. Load GPT-2 teacher. Build OWT eval corpus + WikiText-2 eval corpus (200 seqs each, seed=999).
2. Compute baseline PPL on both eval sets (uncompressed teacher).
3. Per seed:
   a. Build OWT probe corpus (PROBE_TRAIN=200, HOLDOUT=200 seqs, seed=seed).
   b. `probe_ffn_geometry` on OWT corpus → `atlas_owt` (channel selection source for BOTH conditions).
   c. **Condition A (OWT-only)**: `derive_ffn_compression(atlas_owt, budget)` → pack → swap → student_A.
   d. **Condition B (mixed-corpus)**: collect 512-token activations per corpus per block via
      `collect_hidden_activations`, concatenate → `refit_data_B`, build pack via
      `build_compressed_pack_from_refits(teacher, atlas_owt, refit_data_B, "mixed_2048")` → student_B.
   e. Evaluate both students (unpolished) on OWT eval + WikiText-2 eval → record PPL.
   f. Polish both students: 500 steps Adam, lr=1e-4, joint (all 4 blocks), using OWT holdout tokens.
   g. Evaluate both students (polished) on OWT eval + WikiText-2 eval → record PPL.
4. Aggregate: mean ± std across 3 seeds for each (condition × polish × eval_set).
5. Print summary table + save JSON to `autoresearch/results/cycle_087_mixed_stacking.json`.
6. Compute relative PPL change: `(B_ppl - A_ppl) / A_ppl * 100` for WikiText-2 and OWT.
7. Apply criteria, print verdict.

## Key imports
From `aleph.lm_geometric_compression`: `TeacherLoadSpec, ProbeCorpusSpec, CompressionBudgetSpec,
build_probe_corpus, load_nanochat_teacher, probe_ffn_geometry, derive_ffn_compression,
swap_compressed_ffns, _evaluate_model, _fit_output_projection, _apply_activation,
_iter_lm_batches, _layer_target_hidden, _select_hidden_units, _standard_weight, _standard_bias`.

Reuse `collect_hidden_activations` and `build_compressed_pack_from_refits` from cycle 078 script
(copy into new file with minimal adaptation — change channel selection source to OWT atlas).

Reuse `_polish_blocks`, `_collect_polish_params`, `_find_compressed_mlp` from cycle 061 script.

## Verification
- Condition A unpolished PPL should roughly match cycle 061 seed-42 0-step result (~+30% degradation).
- Baseline GPT-2 PPL on OWT should be ~29–31 (sanity check).
