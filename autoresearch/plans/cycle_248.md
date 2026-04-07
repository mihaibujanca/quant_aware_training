# Plan — Cycle 248 / Item 258

## Item 258 (verbatim)
> **K=128 + W_out + W_fc trainable at 6 blocks — does unfreezing W_fc help where bias alone failed?** BLOCKS=[1,2,4,5,7,8] K=128, trainable={W_out, b_out, W_fc, b_fc} per block. Joint Adam LR=5e-5, 4000 steps, 3 seeds. Confirmed ≤61.0, Refuted ≥66.0. Baselines: cycle 231 W_out-only 67.562; cycle 221 K=512 59.515; float 53.895.

## Agenda arithmetic discrepancy (resolved)
Agenda reports "14.77M trainable" from `6 × (768×128 + 128 + 3072×768 + 3072)`. This inflates `W_fc` to the **full (768, 3072)** c_fc, which contradicts K=128 compression — CompressedMLP's forward path uses only a sliced (768, 128) c_fc (topk channels). Training the full 3072-wide c_fc while only 128 outputs feed W_out is a degenerate setup: the 2944 unused channels receive no gradient through W_out. Plan follows cycle 213's **sliced-trainable** semantics: W_fc is the (768, 128) topk slice, matching the K=128 compressed surface. Per-block trainable = K×768 + 768 + 768×K + K = 197,504 → 6 blocks = **1,185,024 (~1.19M)**. Documented in the script header; final report flags the agenda typo.

## Target script
`experiments/k128_wout_wfc_6block_cycle248.py` (new).

## Design
Fork `experiments/adam_6block_k128_cycle231.py` and swap `CompressedMLP` → `FullCompressedMLP` from cycle 213 (W_fc and b_fc promoted from buffers to trainable Parameters, sliced to K=128 at init by topk+lstsq). Extend the per-seed `snapshot/restore` to include W_fc and b_fc so best-state restore is complete over all four trainable tensors. Keep cycle 231's seed loop, 3-seed aggregation, disjoint calib/eval, summary md writer.

## Config
- `BLOCKS=[1,2,4,5,7,8]`, `K=128`
- `LR=5e-5` (lower than cycle 231's 1e-4 — larger trainable surface)
- `N_STEPS=4000`, `EVAL_EVERY=100`, `SEEDS=[42,137,2024]`
- `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, disjoint split
- Freeze LM head, attention, and all blocks outside BLOCKS. Unfreeze {W_fc, b_fc, W_out, b_out} in BLOCKS.
- Init: topk+lstsq on W_out; W_fc/b_fc start at topk-sliced float c_fc (identical to cycle 213 init path).
- Verdict: CONFIRMED ≤61.0, REFUTED ≥66.0, else INCONCLUSIVE.

## Reporting
Per-seed: init_ppl, best_ppl, best_step, final@4000, W_fc drift `||ΔW_fc||_F / ||W_fc_init||_F` computed across BLOCKS (average) to confirm W_fc moved off float. Aggregate mean/std, trainable param count (log 1,185,024), wall-clock, verdict vs 67.562 (W_out-only) and 59.515 (K=512 ceiling). Artifacts: `autoresearch/results/cycle_248_k128_wout_wfc.json`, `autoresearch/results/cycle_248_summary.md`.

## Overfit caution
Trainable/calib ratio: 1.19M params / (200 × 128 = 25.6K tokens) ≈ 46× overparam. EVAL_EVERY=100 + best-state restore mandatory; expect best_step well before 4000. If seeds diverge (final >> best), note as overfitting signature, not noise.

## Verification
- Dry-run: Python import check, confirm trainable count prints 1,185,024.
- Smoke run on single seed / 200 steps locally if fast enough, else hand to runner.
- Let runner execute final 3-seed sweep; do not manually invoke.
