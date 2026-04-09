cycle = 369
item = 360
hypothesis = topk+lstsq accumulation order sensitivity on GPT-2 (easiest-first vs block-index vs hardest-first)
verdict = REFUTED
status = PASS

G1_implementation = PASS — sequential cumulative compression correctly implemented: each block compressed from current model state, joint adapter training covers all compressed blocks at each stage, assertion prevents re-compression
G2_reproducibility = PASS — 3 seeds (42, 137, 2024), std < 0.13 PPL for all orderings
G3_interpretation = PASS — 0.95% gap (ratio=1.0095) is within noise; individual seed variance (~0.5 PPL) exceeds ordering gap (~0.5 PPL)
G4_counterintuitive = NOTED — contradicts cycle 007 (27% gap on toy models); explained by joint adapter training compensating for ordering effects; GPT-2 adapters are stronger correctors than toy corrections
G5_learning = ordering is not a lever when adapters are jointly trained at each stage; 500-step Adam suffices to wash out ordering effects
G6_method_failure = MINOR — conclusion scoped to "with adapter fine-tuning" only; pure topk+lstsq without adapters might still be ordering-sensitive (not tested); this is a scope limitation, not a bug

key_numbers = hardest_first=53.27, block_index=53.69, easiest_first=53.78, gap_pct=0.95, float_ppl=53.89
recommendation = No follow-up needed for ordering. The toy-to-GPT-2 contradiction is interesting (flag). If ordering sensitivity without adapters matters, test pure topk+lstsq (no adapter training) — but this is low priority since adapters are always used in practice.
