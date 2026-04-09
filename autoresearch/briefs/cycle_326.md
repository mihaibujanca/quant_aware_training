# Item 339: Encoder/decoder split analysis on parameter-golf 13L_640d

## Context
Cycle 318 measured eff_rank profile on parameter-golf 13L_640d_322: blocks 0–2 have HIGH eff_rank (188, 160, 86), block 3 collapses to 4.89 (largest drop). Hypothesis: the block-3 collapse coincides with the encoder/decoder boundary in the model's skip-connection architecture. Parameter-golf's GPT class splits layers into `num_encoder_layers + num_decoder_layers = num_layers` and wires encoder outputs back to decoder inputs via learned `skip_weights` and `skip_map` (U-Net pattern). If the encoder is 3 layers (blocks 0–2), the collapse at block 3 (first decoder) would be geometrically explained.

## Dependencies
- Cycle 318 eff_rank profile (blocks 0–3)
- Parameter-golf checkpoint: `13L_640d_322_7k_wd70_batch20k_best.npz`
- Model class source: `parameter-golf/train_gpt_mlx.py` lines 685–794 (GPT.__init__, _build_skip_map)

## Key Files
- `autoresearch/cross_project/parameter_golf.md`
- `parameter-golf/train_gpt_mlx.py` (GPT class, skip_map builder)
- `parameter-golf/local_scripts/analyze_checkpoint.py` (load_model)

## Constraints
- Mostly analysis (no forward pass required). Run cap ~5 min.
- REQUIRED: read parameter_golf.md section 0 before coding.
- Step 1: load checkpoint, extract num_encoder_layers, num_decoder_layers, skip_map, skip_weights.shape.
- Step 2: print per-skip-weight Frobenius norm, boundary block index, cross-reference to cycle 318 eff_rank.

## Handoff
Verdict thresholds: (a) CONFIRMED-boundary-explains-drop if encoder/decoder boundary is block 3 ± 1 AND skip_weights are non-trivial (norm ≥ 0.1) AND rank drop coincides; (b) REFUTED-no-correlation if boundary elsewhere; (c) INCONCLUSIVE if boundary at b3 but skip_weights are zero.
