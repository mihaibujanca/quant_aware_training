# Cycle 179 — Plan

**Item:** 189 — PruneNet spectral channel selection — GPT-2 FFN k=4

## Pre-flights

1. Load GPT-2, eval baseline PPL on 200 OWT sequences (expect ~22.7). Abort if outside [15, 65].
2. Probe FFN geometry for blocks [1,4,7,11] using existing `probe_ffn_geometry`.
3. Verify W_down shape is (768, 3072) per block via `_standard_weight(handle.c_proj)`.

## Design

### Spectral channel selection (greedy)

For each FFN block b in [1, 4, 7, 11]:
1. Extract W = _standard_weight(c_proj) — shape (768, 3072), the output projection.
2. Compute reference spectrum: σ_ref = SVD(W).singular_values (768-dim).
3. Greedy channel selection — build set S of 512 channels:
   a. For each candidate channel j, compute alignment = ||W[:,j]||₂ projected onto top singular vector of W. Pick j* with max alignment as seed.
   b. For i = 2..512: for each remaining candidate c, compute σ_trial = SVD(W[:, S∪{c}]).singular_values. Pad/truncate to match σ_ref length. Pick c* minimizing KS-distance(σ_trial, σ_ref).
   c. Optimization: since SVD of 768×512 is expensive × 2560 candidates per step, batch by computing ||W[:,c]|| norms and restricting candidates to top-1024 by norm, then run exact KS on those.

### Topk baseline (control)

Standard `_select_hidden_units` with `selection_mode="topk"` from the existing atlas.

### Both conditions

- lstsq fitting: `_fit_output_projection(hidden[:, selected], output, use_bias=True)`
- Polish: 500 steps Adam on c_proj weights, lr=1e-4 (same as cycle 178)
- Eval: `_evaluate_model` on 200 OWT eval sequences

## Measurements

For each of 3 seeds {42, 137, 7}:
1. `ppl_base`: float baseline
2. `ppl_spectral_pre`: spectral + lstsq, no polish
3. `ppl_spectral_post`: spectral + lstsq + 500-step polish
4. `ppl_topk_pre`: topk + lstsq, no polish
5. `ppl_topk_post`: topk + lstsq + 500-step polish
6. Per-block: KS-distance for spectral selection, channel overlap with topk

## Verdict logic

- Mean ppl_spectral_post <= mean ppl_topk_post - 2 --> CONFIRMED
- Mean ppl_spectral_post >= mean ppl_topk_post + 2 --> REFUTED
- Otherwise --> INCONCLUSIVE

## Script

`experiments/spectral_channel_selection_cycle179.py`

## Output

- `autoresearch/results/cycle_179_spectral_selection.json`
- `autoresearch/results/cycle_179_spectral_selection.png`
