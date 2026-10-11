# Task

Cycle 229, item 239 (P2): apply the cycle 215 sequential per-block Adam recipe to BLOCKS=[1,2,4,5,7,8] K=512, 3 seeds, and compare against the cycle 221 joint 6-block ceiling (best_mean=59.515).

# Hypothesis

Item 239 verbatim: "Cycle 215 showed sequential per-block Adam ≈ joint Adam at 3 blocks (47.253 vs 46.556, joint wins by 0.7 PPL). Cycle 221 showed 6-block joint Adam ceilings at best_mean=59.515 (1.5 PPL above the 58.0 confirmed threshold) with severe overfitting. Cycle 215's audit explicitly recommended retesting sequential at 6 blocks. Apply the cycle 215 recipe to BLOCKS=[1,2,4,5,7,8] K=512: sequentially correct each block (Adam W_out+b_out, 2000 steps, best-state restore per block), 3 seeds (42/137/2024). This is one cycle (~25 min: 6 blocks × 2000 steps × 3 seeds). Confirmed if: sequential 6-block best_mean ≤ 58.0 PPL. Refuted if: sequential best_mean ≥ 60.0 PPL. Inconclusive if: 58.0 < best_mean < 60.0."

# Pre-flight Results

P1-P4 clean. Cycle 215 covered 3 sequential blocks; cycle 221 covered 6 joint blocks. 6-block sequential is NEW.

# Steps

1. Create `experiments/adam_sequential_6block_cycle229.py` based on `experiments/adam_sequential_cycle215.py` (same CompressedMLP / collect_mlp_io / fit_block_lstsq / eval_ppl helpers; same Adam-loop structure with best-state restore per block). Override config.
2. Config: `BLOCKS=[1,2,4,5,7,8]`, `K=512`, `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `EVAL_EVERY=100`, `N_STEPS_PER_BLOCK=2000`, `SEEDS=[42,137,2024]`, `OUTPUT_PATH="autoresearch/results/cycle_229_sequential_6block.json"`, `SUMMARY_PATH="autoresearch/results/cycle_229_summary.md"`. Constants: `JOINT_6BLOCK_BASELINE=59.515` (cycle 221), `SEQUENTIAL_3BLOCK_BASELINE=47.253` (cycle 215), `JOINT_3BLOCK_BASELINE=46.556`.
3. Per seed:
   a. Build the 6-block topk+lstsq compressed model (collect_mlp_io across all 6 blocks once; fit_block_lstsq each block).
   b. Eval `init_ppl` (expect ~86.7 to match cycle 221).
   c. For `block_idx` in `[1, 2, 4, 5, 7, 8]` (ascending order):
      - Freeze ALL parameters; unfreeze only `model.transformer.h[block_idx].mlp.W_out` and `.b_out`. Assert `n_trainable == 768*K + 768 == 393984`.
      - Snapshot `pre_state` (W_out, b_out) for collapse rollback.
      - Fresh `torch.optim.Adam(trainable, lr=LR)` (no shared state across blocks — matches cycle 215).
      - Run 2000 steps with CE loss (`out = model(ids, labels=ids)`), reshuffled calibration order, eval every 100 steps, snapshot best (W_out, b_out) state.
      - On exit: collapse check (`best_ppl > prev_ppl` ⇒ restore pre_state, log `collapsed:true`); else restore best state.
      - Record `pre_block_ppl`, `best_ppl`, `best_step`, `final_ppl_pre_restore`, `restored_ppl`, `collapsed`, `history`.
   d. After all 6 blocks: eval `final_ppl` (== restored_ppl of last block).
4. Aggregate per seed: `init_mean±std`, `after_block_{1,2,4,5,7,8}_mean±std`, `final_ppl_mean±std`.
5. Verdict per item 239: CONFIRMED if `final_mean ≤ 58.0`; REFUTED if `final_mean ≥ 60.0`; INCONCLUSIVE otherwise.
6. Save JSON with full per-seed records, aggregate, baselines, and `verdict`.
7. Write summary markdown: per-seed table (init / after_b1 / after_b2 / after_b4 / after_b5 / after_b7 / after_b8 / final), trajectory description per seed, and comparison block (vs joint 6-block 59.515; vs sequential 3-block 47.253; vs joint 3-block 46.556).

# Validation

- `init_ppl ≈ 86.7` (must match cycle 221's init).
- After block 1 alone: substantial improvement (cycle 215 saw block-1 drop from 61.7 → ~50).
- Trajectory after each block should be monotone non-increasing if no collapse.
- `n_trainable` per block exactly 393984.
- `restored_ppl` matches recorded `best_ppl` within 1e-3.

# Artifacts

- `experiments/adam_sequential_6block_cycle229.py`
- `autoresearch/results/cycle_229_sequential_6block.json`
- `autoresearch/results/cycle_229_summary.md`
- `autoresearch/results/cycle_229_manifest.txt` (written by execute agent)

# Fallbacks

- Wall time over budget (>30 min): reduce `N_STEPS_PER_BLOCK` to 1500 (cycle 215 best step ≈ 2000, cycle 221 best step ≈ 1367 — 1500 keeps both regimes covered).
- If a block's `best_ppl` ≥ `pre_block_ppl` (no improvement), restore `pre_state` and set `collapsed:true` in that block's record (logic already in cycle 215 template).
- If `init_ppl` deviates by >1.0 from cycle 221's 86.7, abort and re-check tokenization / data slicing before proceeding.

# Update bookkeeping

- Move item 239 from NEXT/P2 to PAST/Cycle 229 in `autoresearch/agenda.md`.
- Append TSV row to `autoresearch/results/results.tsv`.
- Commit with cycle 229 message.
- Update `autoresearch/STATUS.md`.
