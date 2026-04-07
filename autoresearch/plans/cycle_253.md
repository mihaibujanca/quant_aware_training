# Item
263 (P2): gelu_noneg + μ targeted to compressed blocks only.
Patch ONLY blocks [1,4,7] with relu(gelu(z))+μ_l, leave the other 9 blocks on raw GELU. Then K=512 topk+lstsq+Adam on those 3 blocks against the patched targets.

# Goal
Test whether targeted μ-patching on the compressed blocks alone beats the raw-GELU K=512 [1,4,7] baseline of 46.556 (cycle 210). Sidesteps cycle 246's failure mode (all-12 patch hurt uncompressed blocks).

# Deliverables
- `experiments/gelu_noneg_targeted_cycle253.py` (new)
- `autoresearch/results/cycle_253_gelu_noneg_targeted.json`
- `autoresearch/results/cycle_253_summary.md`

# Config
BLOCKS=[1,4,7], K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, SEEDS=[42,137,2024], LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, DEVICE=mps.
Disjoint calib/eval: `calib=all_ids[:200]`, `eval=all_ids[200:300]` (cycle 246 tokenize_to_seqs pattern, SEQ_LEN fallback to 64).

# Procedure
1. Load GPT-2 small, float baseline PPL on eval split.
2. Phase 1 — μ precomputation on CALIB for ONLY blocks {1,4,7}:
   - Register c_fc input hooks on blocks 1,4,7 (not all 12).
   - For each l in {1,4,7}: `pre = x_l @ W_fc + b_fc`; `gelu = F.gelu(pre, tanh)`; `noneg = relu(gelu)`; `mu_l = (gelu - noneg).mean(0)`.
   - Assert μ ≤ 0 elementwise (max ≤ 1e-6). Record per-block stats.
3. Phase 2 — Patch ONLY blocks 1,4,7 mlp.act with `GeluNonegCorrected(mu_dict[l])`. Leave other 9 blocks untouched (raw GELU). Assert `model.transformer.h[l].mlp.act` is the new module for l in {1,4,7} and unchanged for l not in BLOCKS. Log post_mu_calib_ppl, post_mu_eval_ppl.
4. Phase 3 — Collect (ln_out, mlp_out) via `collect_mlp_io` (cycle 246) on the patched model for BLOCKS. Fit K=512 topk+lstsq against the patched activation target via `fit_block_lstsq_gelu_noneg` (cycle 246) with mu_dict_cpu[l]. Replace mlp with `CompressedMLP` (W_fc_sub, b_fc_sub, mu_sub, W_out, b_out) — NO LoRA (trimmed from cycle 246 version, no A/B). Forward: `act = relu(gelu(x @ fc_weight + fc_bias)) + mu_sub; return act @ W_out.T + b_out`. Log post_compress_eval_ppl.
5. Phase 4 — Freeze all; unfreeze only W_out, b_out of the 3 compressed blocks. Expected trainable = 3 × (768×512 + 768) = 1,182,336.
6. Phase 5 — For each seed in SEEDS: reset W_out/b_out to snapshot_init (lstsq init); shuffle calib order; Adam(lr=1e-4) for N_STEPS=4000; eval every EVAL_EVERY=100; snapshot best state when `eval_ppl < best_ppl`; restore best state at end; record final_eval_ppl (must match best_ppl within 1e-3).
7. Aggregate best_mean, best_std, final_mean, final_std across 3 seeds.

# Verdict
- CONFIRMED if best_mean ≤ 46.0
- REFUTED if best_mean ≥ 47.0
- INCONCLUSIVE otherwise (46.0–47.0)
Baselines to log: cycle 210 raw K=512 [1,4,7] = 46.556; cycle 220 float ref = 45.064.

# Dependencies
- c246 `experiments/gelu_noneg_lora_k128_cycle246.py`: `GeluNonegCorrected`, `collect_mlp_io`, `fit_block_lstsq_gelu_noneg`, `tokenize_to_seqs`, μ precomputation, snapshot/restore helpers.
- c206 `experiments/adam_wout_correction_cycle206.py`: K=512 Adam W_out+b_out template, `CompressedMLP` — extend with `mu_sub` buffer + `relu(gelu(...)) + mu_sub` forward for Phase 3.

# Sanity checks
- μ computed ONLY on blocks {1,4,7} (3 entries in mu_dict, not 12); `mu_l.max() ≤ 1e-6` per block; topk+lstsq uses patched activation `relu(gelu(pre)) + mu_l_cpu`.
- Uncompressed blocks still raw GELU: `type(model.transformer.h[l].mlp.act) is not GeluNonegCorrected` for l ∉ BLOCKS after Phase 2 (pre-Phase 3 mlp replacement).
- `init_eval_ppl == post_compress_eval_ppl` per seed (within 1e-3); `|final_eval_ppl - best_ppl| < 1e-3` per seed.
- Disjoint calib/eval by object id and tensor equality; len checks.
