cycle = 379
item = 387
verdict = CONFIRMED
status = PASS_WITH_CAVEATS

G1_param_counts = PASS — code formula ffn_params_per_block(512, 128) = 3*(2*768*512 + 512 + 768 + 512*128 + 128 + 768*128 + 768) = 2,857,344. Matches JSON. Inline assertion in script verifies k=512 h=64 case. Note: the audit prompt formula had a typo (512*3072 instead of 512*768) but the actual code is correct.
G2_eval_split = PASS — train on wikitext-2-val (first 200 seqs as calib), checkpoint selection on val (seqs 200-250), final eval on wikitext-2-test (200 seqs). No test leakage into training or selection.
G3_single_seed = NOTED — single seed 42 per plan. No 3-seed validation of Pareto-frontier points yet. Manifest acknowledges this.
G4_confirmation_criterion = WEAK_PASS — criterion as coded: any comp with params <= 80% of ANY float config AND PPL <= that float PPL + 2. Triggered by comp_k512_h128 (2.86M, 47.24) vs float_k3072_h0 (14.17M, 56.52). This is trivially true: compressed+adapter beats unaugmented float by 9 PPL at 20% params. The more meaningful comparison (comp vs float+adapter at matched capacity) shows a 7 PPL gap at 5-6x fewer params. The criterion tests "compression + adapter > raw float" which is a low bar. The manifest's own narrative correctly notes the 7 PPL gap.
G5_block_set = NOT_TESTED — only blocks [2, 1, 10]. No evidence the Pareto split generalizes to other block sets or block counts. Known limitation.

additional_notes = Float h=0 correctly marked non-Pareto (dominated by comp_k512_h32). Compressed adapter saturates at h=32 (47.49 vs 47.24 for h=128 — 0.25 PPL for 15% more params). This is the most actionable finding: h=32 is the cost-effective adapter size for compressed blocks. Adapter input dimension is k (not d_model), so float adapters see 6x more signal — may partly explain the PPL floor difference.
