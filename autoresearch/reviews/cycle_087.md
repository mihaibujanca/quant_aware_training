verdict: APPROVE
round: 1
issues: 0
notes: 3

## R1: Premise validity — PASS

Item 96 asks whether cycle 078's mixed-corpus refit advantage (+19.1% single-block) transfers to 4-block stacking. This is the natural next step — single-block gains can vanish or amplify when stacking, so the test is necessary. Criteria are well specified: 10% relative WikiText-2 improvement with <3% OWT degradation. Thresholds are reasonable relative to cycle 078's single-block effect size.

## R2: Measurement soundness — PASS

Channel selection is correctly fixed to OWT for both conditions, isolating the refit data source as the only variable. Token counts are matched at 2048 for both conditions (512 x 4 corpora vs 2048 OWT). lstsq fitting is well-determined: 2048 samples for ~513 parameters per regression (MP3 ratio ~4:1). The `_fit_output_projection` lstsq is the same code path for both conditions. Adam polish parameters (W_down only, lr=1e-4, 500 steps, joint) match the cycle 061 configuration that produced the known +3.79% degradation baseline.

## R3: Statistical adequacy — PASS

Three seeds [42, 123, 7], mean +/- std aggregation. 200 eval sequences per eval set. Same protocol as cycles 061 and 078. Adequate for the 10% relative effect size being tested.

## R4: Method-architecture alignment — PASS

`collect_hidden_activations` hooks `mlp.c_fc` (pre-GELU hidden layer), collects post-activation hidden and MLP output. Same hook structure as cycle 078. `_polish_blocks` optimizes `c_proj.weight` (W_down) with Adam, freezes everything else. `_evaluate_model` uses `corpus.eval_val` partition. All correct for GPT-2 FFN compression pipeline.

## R5: Known pitfall scan — PASS

- MP1 (threshold above trivial): N/A — PPL comparison, not threshold detection.
- MP3 (underdetermined fitting): 2048 tokens >> 513 params. Ratio 4:1. Checked.
- MP5 (insufficient capacity): Both conditions use identical architecture (same blocks, same k), only refit data differs. If one fails, both fail symmetrically. Checked.
- MP7 (post-LayerNorm): Hooks are on mlp.c_fc, not post-LayerNorm. Checked.

No pitfalls apply.

## Note 1 (non-blocking): Baseline PPL sanity check is wrong

The plan states "Baseline GPT-2 PPL on OWT should be ~29-31." Cycle 061 measured GPT-2 baseline on OWT at 57.653. The 29-31 value is GPT-2's canonical Wikitext-103 test PPL, not the OWT eval PPL from this pipeline's text_path-based corpus construction (which uses a shuffled 20% val split of the OWT sample file). The implementer should use ~55-60 as the OWT sanity check range, or simply compare against cycle 061's recorded value.

## Note 2 (non-blocking): WikiText-2 train/eval partition overlap

Condition B collects 512 WikiText-2 tokens for refit via `build_probe_corpus(seed=seed)`, while the WikiText-2 eval set uses `build_probe_corpus(seed=999)`. Since `_read_text_chunks` shuffles docs before the 80/20 train/val split, different seeds produce different partitions. A small number of refit docs (~1 of 4 sequences) could land in the eval partition under a different seed. At ~0.5% of eval tokens, this is negligible for PPL measurement but worth noting.

## Note 3 (non-blocking): Polish data distribution asymmetry

Both conditions are polished on OWT holdout tokens. This is a valid controlled design, but it subtly favors condition A (OWT-only refit) because the polish gradient signal is distribution-matched to condition A's refit data but not to condition B's 75% non-OWT refit data. If the experiment shows mixed-corpus helps WikiText-2 pre-polish but the advantage shrinks post-polish, this asymmetry is one explanation. Consider adding a sensitivity check: polish condition B on mixed-corpus holdout tokens as a third condition, if time permits.
