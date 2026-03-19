# Cycle 060 Plan — Iterative Cascaded Compression

## Goal
Reduce stacking PPL degradation from +30.2% (independent, cycle 058) to ≤+10% by re-probing through compressed blocks sequentially before fitting the next block.

## Hypothesis
lstsq refit for block N assumes float inputs but receives distorted outputs from compressed blocks 1..N-1. Cascading the refit eliminates this mismatch.

## File
`experiments/cascaded_compression_gpt2.py` (new, based on `stacking_error_budget_gpt2.py`)

## Implementation
1. Load GPT-2 teacher, build OWT probe corpus (4096 tokens, 200 eval seqs, seq_len=128).
2. Probe ALL target blocks [1,4,5,7] on float teacher → `float_atlas`. This gives the independent baseline.
3. **Independent baseline** (control): `derive_ffn_compression(float_atlas, budget)` → stacked pack → eval PPL. Replicates cycle 058 k=4 result.
4. **Cascaded variant**:
   - Sort blocks in layer order: [1, 4, 5, 7].
   - `working_model = deepcopy(teacher.model)`.
   - For each block_id in [1, 4, 5, 7]:
     a. Probe `working_model` for `block_id` only → single-layer atlas (uses `probe_ffn_geometry` with `layer_ids=(block_id,)`). The model already has prior blocks compressed, so activations are distorted.
     b. `_derive_single_block_pack(atlas, budget, block_id)` → single-block pack.
     c. `_replace_ffn_modules(working_model, ...)` in-place with that pack.
   - Collect all 4 `CompressedFfnLayer` objects → final `CompressedFfnPack`.
   - Eval: swap all 4 layers into a fresh teacher copy → `_evaluate_model`.
5. Run both variants for seeds [42, 123, 7]. Report mean ± std PPL and % degradation.

## Key Detail
`probe_ffn_geometry` forwards through `teacher.model` — so passing the partially-compressed `working_model` as the teacher's model naturally produces distorted activations at downstream blocks. Need to temporarily set `teacher.model = working_model` or create a new `TeacherBundle` with the working model.

## Metrics
- `ppl_float`, `ppl_independent`, `ppl_cascaded` (mean ± std over 3 seeds)
- `pct_degrad_independent`, `pct_degrad_cascaded`
- Per-block reconstruction MSE for both variants

## Outputs
- `autoresearch/results/cycle_060_cascaded.json` — all metrics
- `autoresearch/results/cycle_060_cascaded.png` — bar chart: float vs independent vs cascaded PPL

## Verification
- Independent baseline should reproduce +30% ± 3pp from cycle 058.
- Cascaded block 1 should be identical to independent block 1 (no prior distortion).

## Success Criteria
- **Confirmed**: cascaded ≤ +10% degradation.
- **Refuted**: cascaded ≥ +25%.
- **Inconclusive**: between +10% and +25%.
