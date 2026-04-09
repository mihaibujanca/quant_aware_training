cycle = 380
item = 388
verdict = REFUTED
status = PASS

G1_all_blocks_adapted = PASS — BLOCKS = list(range(12)), loop at L227-239 replaces all 12 MLPs. Param assertion confirms 1,189,632 = 12 * 99,136. JSON matches.
G2_three_seeds_test_eval = PASS — seeds [42, 137, 2024], test PPL 39.71/39.37/39.47 (std=0.14). Test split = wikitext-2-test (200 seqs), separate from train (val calib 200 seqs) and checkpoint selection (val 50 seqs). No leakage.
G3_gain_vs_noise = INCONCLUSIVE — 0.21 PPL gain (39.73 -> 39.52) is ~1.5x the 12-block std (0.14) but the 3-block ref (39.73) is single-seed. Best 12-block seed 39.37 beats ref by 0.36; worst 39.71 matches it within 0.02. Without multi-seed 3-block variance, 0.21 is indistinguishable from noise.
G4_checkpoint_selection = PASS — all 3 seeds peak at step 800-900 then drift upward (overfitting on 50-seq val). best_state saved at argmin(val_ppl) and restored before test eval (L186, L268). Mechanism correct; overfitting is harmless.
G5_saturation_meaning = The adapter-on-float ceiling is ~39.5 PPL regardless of block count (3 vs 12). Marginal gain per additional adapter block is 0.21/9 = 0.023 PPL. This locks in c377's 3-block float+adapter (39.73) as a valid near-optimal control for all compression comparisons. No need to revisit the float baseline at higher block counts.

additional_notes = The 4x param increase (297K -> 1.19M) for 0.21 PPL is a clear diminishing-returns signal. Combined with c379 finding that compressed adapters saturate at h=32, the adapter architecture is fully characterized: float h=64 on 3 blocks and compressed h=32 are the cost-effective configs. Future work should focus on closing the float-vs-compressed gap (~7.5 PPL from c378) rather than expanding adapter coverage.
