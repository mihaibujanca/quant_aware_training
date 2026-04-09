cycle = 381
item = 389
question = Does 6-block composition penalty scale sublinearly (<15) or superlinearly (>25) vs 3-block ref (7.49)?
verdict = INCONCLUSIVE
penalty = 24.57
threshold_confirmed_lt = 15.0
threshold_refuted_gt = 25.0

G1_same_config = PASS — both configs use blocks [2,1,10,4,3,11], same seeds/LR/steps. Float h=64, compressed h=128. Consistent.
G2_seeds_and_eval = PASS — 3 seeds, test-only eval. Float std=0.13, comp std=0.22. Low variance.
G3_float_saturation = PASS — float+adapter: 3-block 39.73, 6-block 39.65, 12-block 39.52. Nearly flat. Compressed: 3-block 47.22, 6-block 64.22 (+17 PPL for +3 blocks). Penalty growth is entirely from compressed side.
G4_superlinear_character = FLAG — penalty ratio 3.28x for 2x blocks is superlinear. Missed refutation threshold by 0.43 PPL. The distinction between INCONCLUSIVE and REFUTED here is a threshold boundary artifact, not a clean separation. The character of the result is superlinear.
G5_param_confound = PASS — compressed side has 5.3x more trainable params per block yet performs far worse. Confound works AGAINST the finding (equal params would make compression penalty even larger). Not a threat.

interesting = yes
interesting_reason = Superlinear composition penalty (3.28x for 2x blocks) with param confound favoring compressed side rules out capacity-limited explanation. Compression error compounds structurally through the model. Float side saturated at 3 blocks (0.08 PPL gain from 3 more) while compressed side degraded by 17 PPL. This asymmetry is the clearest evidence yet that the composition barrier is intrinsic to topk+lstsq compression, not an adapter training issue.
recommendation = Consider whether REFUTED is the more honest verdict given 0.43 margin. If keeping INCONCLUSIVE, note the superlinear character. Next: diagnose WHERE the 17 PPL comes from — is it early blocks corrupting later blocks, or uniform degradation? Per-block ablation (add blocks one at a time in sequence order) would isolate the compounding structure.
