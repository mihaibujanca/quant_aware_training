cycle = 378
item = 386
outcome = CONFIRMED
status = PASS

G1_no_leakage = PASS — training on wikitext-2 validation (200 seqs), checkpoint selection on separate val eval (50 seqs), test split (200 seqs) touched exactly once per config after checkpoint selection. Config metadata in JSON confirms split roles.
G2_seeds = PASS — 3 seeds (42, 137, 2024). Float+adapter std=0.048, compressed+adapter std=0.080. Tight.
G3_val_test_consistency = PASS — val gap ~8.5 (c377), test gap 7.49. Same direction, <1.1 PPL difference. Slight narrowing on test is normal.
G4_float_baseline_shift = PASS — float val 53.89, float test 56.52. Shift +2.63, normal for wikitext-2.
G5_project_implications = Compression actively hurts when adapters are available. The "beat float" anomaly is fully explained by adapter capacity, not compression. Closes the question cleanly; next step is Pareto sweep (item 387).

interesting = no — result matches val finding, no surprises.
