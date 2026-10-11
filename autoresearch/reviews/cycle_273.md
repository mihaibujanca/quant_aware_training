# Cycle 273 Review — Item 283

**Decision: APPROVE**

## Summary
Trivial extension of cycle 247's hetero K runner to 4 blocks with two configs. Hetero {1:512, 4:512, 7:512, 11:22} (1.20M trainable) vs uniform K=22 control (70,656 trainable). 4-block joint Adam W_out+b_out, LR=1e-4, 4000 steps, 3 seeds, best-state restore. 6 fits total. Confirms iff hetero ≤ 44.5 AND control ≥ 60; refuted iff hetero ≥ 47.

## Checks
- **Brief alignment**: BLOCKS=[1,4,7,11], K_PER_BLOCK exactly per item 283, control uniform K=22, 3 seeds, 6 fits, joint W_out+b_out, LR=1e-4, 4000 steps, disjoint splits, best-state restore, verdict thresholds verbatim. PASS.
- **Trainable count (hetero)**: 3 × (512 × 768 + 768) + 1 × (22 × 768 + 768) = 3×394,176 + 17,664 = 1,182,528 + 17,664 = **1,200,192**. Brief says ~1.2M. Plan recomputes EXPECTED_TRAINABLE per config and asserts. PASS.
- **Trainable count (uniform_k22)**: 4 × (22 × 768 + 768) = 4 × 17,664 = **70,656**. PASS.
- **Reuse of cycle 247 lstsq init**: `fit_block_lstsq` uses score = `act_all.std(0) * W_proj.norm(dim=1)` then top-K (cycle 218 importance pruning). Same as cycle 247/255/266. K=22 path is well-supported (cycle 270 used same code with K=22). PASS.
- **Disjoint splits**: cycle 247 base loads 200 calib + 100 eval from wikitext-2 val (texts ≥50 chars, SEQ_LEN=128, no overlap). PASS.
- **Best-state restore**: cycle 247 snapshot/restore loop assertions `abs(final_ppl_restored - best_ppl) < 1e-3` preserved. PASS.
- **CompressedMLP at K=22**: same module — fc_weight (768, 22), b_fc (22,), W_out (768, 22), b_out (768,). Forward `F.gelu(x @ fc_weight + fc_bias) @ W_out.T + b_out` is shape-correct. PASS.
- **Per-config seeding**: each (cfg, seed) reloads GPT-2 fresh and reseeds — no state leakage between configs. PASS.
- **Output namespacing**: `cycle_273_hetero_eff_rank.json`, `cycle_273_summary.md`, `cycle_273_run.log`. No collision. PASS.
- **Run cap**: cycle 247 6-block ran ~36 min on MPS for 3 seeds × 4000 steps. 4-block × 2 cfgs × 3 seeds = 6 fits, hetero ~25 min, control (small params, fast Adam) ~20 min → ~45 min total. Within session budget. PASS.
- **Verdict logic**: hetero gate AND control gate both checked before CONFIRMED; REFUTED only on hetero ≥ 47; otherwise INCONCLUSIVE. PASS.

## Risks
- **Hetero verdict gate is tight**: cycle 266 [1,4] K=512 hit 43.852, cycle 270 single-block-11 K=22 hit 54.469. The hypothesis is that adding K=22 block-11 patch on top of [1,4,7] K=512 (cycle 210=46.56) breaks below 44.5. Plausible but not guaranteed; INCONCLUSIVE band 44.5–47 is realistic.
- **Control K=22 gate (≥60)**: cycle 270 single-block-11 K=22 gave 54.469. Joint 4-block uniform K=22 (4×17K params) is much weaker than 1-block; unclear it hits 60 — if it lands at e.g. 55-58 the control gate fails and the verdict becomes INCONCLUSIVE even if hetero is good. Acceptable per item 283 spec; the hetero comparison is the primary signal.
- **lstsq stability at K=22**: act_sub is (200×128, 22) = (25600, 22), well-conditioned. lstsq is fast and stable. PASS.
- **MPS memory**: 4-block × 1.2M params is comfortably below cycle 221's 6-block × 2.36M which ran fine. PASS.
- **Per-config best_state dict keyed by block**: must reset best_state per fit (snapshot inside `run_seed`). cycle 247 already does this. PASS.

## Decision
APPROVE — execute as planned. Plan correctly extends cycle 247's K_PER_BLOCK pattern to a CONFIGS loop, preserves all critical hyperparameters, and applies the item 283 verdict criteria verbatim. The hetero trainable count (1,200,192) and uniform trainable count (70,656) are derivable from the brief and will be asserted at runtime.
