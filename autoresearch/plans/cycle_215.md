# Task
225. **Iterative block-by-block Adam correction — sequential refinement**
   - Cycles 206-212 corrected all compressed blocks simultaneously. Alternative: correct blocks sequentially (block 1 first, then fix block 4 given block 1 corrected, then block 7). Each block's W_out is corrected in isolation using the already-corrected downstream context. May reduce overfitting by keeping each optimization smaller. BLOCKS=[1,4,7] K=512, 2000 Adam steps per block.

# Hypothesis
- Confirmed if: sequential correction PPL ≤46.0 (at or below joint Adam best 46.556) without overfitting
- Refuted if: sequential correction PPL ≥60 (worse than joint Adam)
- Inconclusive if: PPL 46-60

# Pre-flight Results
- P1 (duplicate): NO. Item 217 = joint Adam (cycle 206), item 220 = step curve (cycle 210), item 208 = sequential topk+lstsq progressive fitting (cycle 197). Sequential Adam W_out fitting is new.
- P2 (dependency): NO (none per agenda).
- P3 (≤30 min): OK. Cycle 206 ran 1000 steps in ~few min; 3 seeds × 3 blocks × 2000 steps = 18000 total step-block units but each block trains only ~390K params (vs 1.18M joint) so backward is comparable. Estimate ~2-4 min per block-seed → 18-36 min. Budget-tight; fallback at 1500 steps ready.
- P4 (program/prompts): NO edits needed.
- Trivial baselines: float PPL=53.895; topk+lstsq init PPL=61.746; joint Adam 1000 steps=48.111 (cycle 206); joint Adam best=46.556 (cycle 210, step 4000); cycle 205 single-block K=512 block 1 alone = 54.689 PPL.

# Steps
1. Create `experiments/adam_sequential_cycle215.py` by copying `experiments/adam_wout_correction_cycle206.py` and modifying the training loop.
2. Config (match cycle 206 exactly): BLOCKS=[1,4,7], K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, EVAL_EVERY=100, N_STEPS_PER_BLOCK=2000, SEEDS=[42,137,2024], optimizer=Adam (defaults).
3. Data loading, float PPL eval, CompressedMLP, collect_mlp_io, fit_block_lstsq, eval_ppl: copy verbatim from cycle 206.
4. For each seed in SEEDS:
   a. `torch.manual_seed(seed)`; `random.seed(seed)`.
   b. Reload float GPT-2; build topk+lstsq compressed model over BLOCKS (call collect_mlp_io + fit_block_lstsq per block, install into `model.transformer.h[b].mlp`).
   c. Eval init PPL (expect ~61.7).
   d. For block_idx in [1, 4, 7] (sequential order):
      - Freeze ALL model parameters (`for p in model.parameters(): p.requires_grad_(False)`).
      - Unfreeze ONLY `model.transformer.h[block_idx].mlp.W_out` and `.b_out`.
      - Build fresh Adam optimizer on those 2 params (lr=1e-4).
      - `random.shuffle(calib_ids)`; run 2000 steps (wrap inner `for ids in calib_ids` with step counter as in cycle 206).
      - Every 100 steps, eval full-model PPL on eval_ids. Track `best_ppl_block` and snapshot `best_state = {"W_out": W_out.detach().clone(), "b_out": b_out.detach().clone()}` when improved.
      - After 2000 steps, restore `best_state` into the block's parameters (early-stop-within-block policy).
      - Record `after_block_{idx}_best_ppl`, `after_block_{idx}_final_ppl` (before restore), `history` (list of {step, eval_ppl, calib_loss}).
   e. After all three blocks done, eval `final_ppl` (with best states restored).
5. Aggregate across seeds: mean ± std for `init_ppl`, `after_block_1_best`, `after_block_4_best`, `after_block_7_best`, `final_ppl`. Also record per-seed values.
6. Verdict: compare `mean(final_ppl)` against thresholds: ≤46.0 CONFIRMED, ≥60.0 REFUTED, else INCONCLUSIVE. Also flag overfitting if any block's last eval > its best by >2 PPL (recorded, not verdict-blocking).
7. Write `autoresearch/results/cycle_215_sequential_adam.json` with:
   ```
   {"cycle":215,"item":225,"verdict":...,"float_ppl":...,"joint_adam_baseline":46.556,
    "topk_lstsq_init_baseline":61.746,
    "seeds":[{"seed":42,"init_ppl":...,"blocks":[{"block":1,"best_ppl":...,"best_step":...,"final_ppl":...,"history":[...]},...],"final_ppl":...},...],
    "aggregate":{"init_ppl_mean":...,"init_ppl_std":...,"after_block_1_mean":...,"after_block_4_mean":...,"after_block_7_mean":...,"final_ppl_mean":...,"final_ppl_std":...},
    "config":{...}}
   ```
8. Print summary table: seed × {init, after_b1, after_b4, after_b7, final}.

# Validation
- Init PPL ≈ 61.7 (match cycle 206).
- After block 1 alone: expect <61 (cycle 205 single-block K=512 block 1 → 54.689).
- After all 3 blocks: expect near joint Adam range (46-50) if hypothesis holds.
- Sanity: per-block trainable params = 768×512 + 768 = 393,984 + 768 = 394,752 (~395K). Assert this in code.
- Seeds must produce 3 distinct init_ppl values (topk indices deterministic from data, but Adam trajectory seed-dependent — init PPL identical is fine; final PPLs must differ).

# Artifacts
- `experiments/adam_sequential_cycle215.py`
- `autoresearch/results/cycle_215_sequential_adam.json`
- `autoresearch/results/cycle_215_manifest.txt` (written by execute stage: item_number, outcome, key_metric, commit, paths, learned, next)

# Fallbacks
- If wall-time >28 min projected after seed 1, reduce N_STEPS_PER_BLOCK to 1500 for remaining seeds and note in JSON under `config.reduced_steps`.
- If a block's training pushes PPL WORSE than pre-block state (best_ppl_block > pre_block_ppl), restore the pre-block state for that block and log `{"block":i,"collapsed":true}`. Report this as a structural finding (not silent).
- If sequential final_ppl >> joint (≥60): record as REFUTED with explicit failure mode (which block degraded, by how much).
- If best-state snapshotting fails (OOM on MPS), fall back to LAST state and set `best_state_policy:"last"` in JSON with a warning print.

# Update bookkeeping
- Move item 225 from NEXT/P0 to PAST EXPERIMENTS in `autoresearch/agenda.md` (delete from NEXT, add dated entry under `### Cycle 215`).
- Append row to `autoresearch/logs/experiments.tsv`.
- Commit: `[autoresearch] cycle 215: sequential block-by-block Adam correction`.
- Refresh `autoresearch/STATUS.md`.
