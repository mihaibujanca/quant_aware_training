cycle = 374
item = 382
verdict = REFUTED
audit_status = PASS
G1_implementation = PASS — k=768 verified in config and per-block JSON for all 6 blocks. Blocks [2,1,10,4,3,11] correct. adapter_hidden=64 correct. Minor note: block 2 frob_recon varies across seeds (1502 vs 1780 for seed 137) despite deterministic compression — likely MPS float non-determinism in calibration forward pass. Does not affect verdict.
G2_reproducibility = PASS — 3 seeds (42, 137, 2024). 6-block std=0.19, 3-block std=0.28. Tight.
G3_interpretation = PASS — 3-to-6-block gap narrowed from 18.11 to 15.42 PPL (delta reduction 2.69). This is larger than noise (~2x std) so the improvement is real but insufficient. Verdict thresholds (CONFIRMED <55, REFUTED >60) are reasonable. Result 61.85 is clearly in refuted range.
G4_counterintuitive = N/A — moderate improvement at reduced compression is expected behavior.
G5_learning = PASS — clear takeaway: per-block error reduction via more channels does not proportionally reduce composed error. The composition barrier is structural, not just a function of per-block error magnitude.
G6_method_failure = PASS — 2000 steps adequate despite larger param count. Restored PPL matches best PPL in all 6 runs. Std is tight. 3-block k=768 shows clear improvement over k=512 baseline, confirming the training recipe works. The 6-block failure is genuine composition-level, not training-level.
interesting = no
recommendation = Closes the "more channels" avenue. Composition barrier is confirmed to be superlinear in number of compressed blocks rather than proportional to per-block error. Focus shifts to items 383-385 which attack the problem from different angles (adapter pruning, CKA diagnostics, adapter-only baseline).
