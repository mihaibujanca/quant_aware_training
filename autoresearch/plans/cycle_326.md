# Cycle 326 — Item 339: Encoder/Decoder Split vs b2→b3 Eff_Rank Drop

## Task
Check whether the b2→b3 eff_rank collapse (86→4.89) coincides with the encoder/decoder boundary in 13L_640d_322.

## Pre-flight
- P1 (duplicate): no prior cycle checked this.
- P2 (dependencies): cycle 318 eff_rank profile done, checkpoint exists.
- P3 (concrete): yes — read config, load model, print boundary + skip norms.
- P4 (modifies program.md): no.

## Key Finding (from config JSON)
`parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_summary.json` shows `"num_encoder_layers": 0` (auto = `num_layers // 2 = 6`). Encoder = blocks 0-5, decoder = blocks 6-12. Boundary is between block 5 and 6. Block 3 is deep inside the encoder. Expected verdict: **REFUTED-no-correlation**.

## Steps
1. `cat autoresearch/cross_project/parameter_golf.md` (required Step 0).
2. Write `parameter-golf/local_scripts/pgolf_encoder_decoder_analysis.py`:
   - Load checkpoint via `analyze_checkpoint.load_model`.
   - Print `model.num_encoder_layers`, `model.num_decoder_layers`, `model.skip_map`.
   - Print per-skip Frobenius norm of `model.skip_weights` (shape `(n_skips,)`; each is a scalar, print as-is).
   - Print boundary block index = `model.num_encoder_layers` (first decoder block).
   - JSON to stdout.
3. Write `experiments/pgolf_enc_dec_split_cycle326.py` (qat-side wrapper): subprocess-invoke Step 2 script, parse JSON, cross-reference to cycle 318 profile (b0=188, b1=160, b2=86, b3=4.89), write summary.
4. Run: `/Users/mihai/Projects/parameter-golf/.venv/bin/python parameter-golf/local_scripts/pgolf_encoder_decoder_analysis.py --checkpoint /Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz`
5. Write `autoresearch/results/cycle_326_summary.md` with boundary index, skip norms table, and cross-reference to cycle 318.
6. Verdict: **REFUTED-no-correlation** if boundary is block 6 (3 blocks away from b3). CONFIRMED only if boundary is block 3±1.
7. Commit: `[autoresearch] cycle 326: item 339 enc/dec boundary at b6, not b3 — REFUTED-no-correlation`.

## Validation
- `model.num_encoder_layers` printed by pgolf script matches config JSON value (6).
- Skip_weights norms are printed regardless (needed for INCONCLUSIVE check).

## Artifacts
- `parameter-golf/local_scripts/pgolf_encoder_decoder_analysis.py`
- `experiments/pgolf_enc_dec_split_cycle326.py`
- `autoresearch/results/cycle_326_summary.md`
- `autoresearch/results/cycle_326_exec_result.txt`
- `autoresearch/results/cycle_326_manifest.txt`
