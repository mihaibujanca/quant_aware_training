# Task

Item 250 (P0) verbatim: **K=1 Adam correction at 6 blocks — does the K=1 floor hold at cascade depth?** Re-run cycle 230 machinery with BLOCKS=[1,2,4,5,7,8] (cycle 221 best 6-block subset) at K=1 only, 3 seeds {42, 137, 2024}, W_out+b_out Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore. Trainable: 6 × (1×768 + 768) = 9216. Compare against (a) K=512 6-block 59.515 ± 0.094 (cycle 221), (b) K=128 6-block 67.562 ± 0.061 (cycle 231), (c) K=1 3-block 59.245 ± 0.029 (cycle 230).

# Hypothesis

K=1 6-block lands in [58.0, 62.0] — the ~59.5 PPL floor is a structural cascade-depth barrier and channel capacity is irrelevant even at 6 blocks. Alternative: K=1 collapses like K=128 did (≥70), meaning the K-vs-PPL curve at 6 blocks penalizes both extremes and only K=512 sits at the floor.

# Pre-flight Results

Pre-flight clean. Trivial K-swap on cycle 221 template; no dependencies. Cycle 230 already validated K=1 lstsq+Adam recipe at 3 blocks (59.245 ± 0.029). Cycle 221 scaffolding (`experiments/adam_6block_earlystop_cycle221.py`) is the direct template — only K constant changes.

# Steps

1. Create `experiments/adam_6block_k1_cycle240.py` by copying `experiments/adam_6block_earlystop_cycle221.py`.
2. Set constants: `BLOCKS=[1,2,4,5,7,8]`, `K=1`, `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `N_STEPS=4000`, `EVAL_EVERY=100`, `SEEDS=[42,137,2024]`. Update `OUTPUT_PATH="autoresearch/results/cycle_240_k1_6block.json"`, `SUMMARY_PATH="autoresearch/results/cycle_240_summary.md"`. Update header docstring + cycle/item ids in result dict (cycle=240, item=250).
3. Per seed: load GPT-2, collect MLP IO via hooks, fit `topk(K=1) + lstsq` per block (CompressedMLP), eval init PPL, freeze all params, unfreeze W_out+b_out for 6 blocks, Adam 4000 steps with EVAL_EVERY=100, snapshot best state on improvement, restore best at end, assert restore matches recorded best.
4. Aggregate mean±std across 3 seeds for init/best/best_step/final@4000. Verdict per item 250: CONFIRMED if best_mean ∈ [58.0, 62.0]; REFUTED if best_mean ≥ 70.0; INCONCLUSIVE if 62.0 < best_mean < 70.0.
5. Write JSON (full per-seed history + aggregate) and summary markdown with comparison table against cycle 221 (K=512), cycle 231 (K=128), cycle 230 (K=1, 3-block).

# Validation

- Trainable param count = 9216 (assert).
- Best-state restore PPL matches recorded best within 1e-3 (assert already in template).
- 3 seeds, all complete, no NaN.
- Init PPL recorded so we can attribute Adam delta vs lstsq init.

# Artifacts

- `experiments/adam_6block_k1_cycle240.py`
- `autoresearch/results/cycle_240_k1_6block.json`
- `autoresearch/results/cycle_240_summary.md`

# Fallbacks

- If init PPL is catastrophic (>200) at K=1 6-block (lstsq init unable to fit single channel across 6 stacked blocks), still run Adam — that's the experiment. Report regardless.
- If runtime per seed exceeds ~20 min on MPS, no mitigation; accept and let runner finish (trainable count is 256× smaller than cycle 221, should be faster, not slower).
- If MPS OOM (unlikely at K=1), fall back to CPU by setting `DEVICE="cpu"`.

# Update bookkeeping

- Append cycle 240 to `autoresearch/program.md` PAST list with verdict.
- Move item 250 from NEXT to PAST in `autoresearch/agenda.md`.
- Add `autoresearch/briefs/cycle_240.md`, `plans/cycle_240.md`, `results/cycle_240_*` to manifest.
