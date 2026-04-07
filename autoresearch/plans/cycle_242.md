# Cycle 242 Plan — Item 252

## Item (verbatim)
252. **Early-only vs late-only 3-block compression — disentangle cascade depth from block count**
   - Cycle 221 (6 blocks [1,2,4,5,7,8]) → 59.515. Cycle 210 (3 blocks [1,4,7]) → 46.556. The ~13 PPL gap could be driven by (a) more blocks compressed, (b) compressing through the full depth of the model, or (c) compressing consecutive-ish blocks that interact. Test (a) vs (b) directly: run joint Adam W_out+b_out K=512, 3 seeds {42, 137, 2024}, disjoint calib/eval (`all_ids[:200]`/`[200:300]`, SEQ_LEN=128, seed 42), LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore, on TWO configurations with identical block COUNT but different DEPTH coverage: (i) EARLY-only BLOCKS=[1,2,3] — 3 consecutive early blocks, (ii) LATE-only BLOCKS=[9,10,11] — but block 11 is rank-expansion territory (see `notes about block 0/11 in cycles 216/218`), use BLOCKS=[8,9,10] instead to stay away from the LM head. Compare both against cycle 210 BLOCKS=[1,4,7] baseline 46.556 ± 0.028. Expected: if depth coverage dominates, [1,4,7] >> [1,2,3] and [8,9,10] (spread cheaper than contiguous); if block count dominates, all three land near 46-48; if late blocks are structurally harder, [8,9,10] ≫ [1,2,3].
   - Confirmed: one of the two new 3-block configs is ≥ 2.0 PPL worse than 46.556.
   - Refuted: both new configs land within ±0.8 PPL of 46.556.
   - Inconclusive otherwise.

## Plan
1. Copy `experiments/adam_wout_correction_cycle206.py` → `experiments/early_vs_late_3block_cycle242.py`. Replace single `BLOCKS` with `CONFIGS = [[1,2,3], [8,9,10]]`, set `SEEDS=[42,137,2024]`, `K=512`, `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `N_STEPS=4000`, `EVAL_EVERY=100`. Data loader seeded at 42 once (disjoint `all_ids[:200]`/`[200:300]`).
2. Outer loop over `CONFIGS`; inner loop over `SEEDS`. For each (cfg, seed): `torch.manual_seed(seed)`, fresh `GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)`, `collect_mlp_io` for cfg blocks, `fit_block_lstsq` each → install CompressedMLP, `ppl_init = eval_ppl(...)`.
3. Freeze all params; unfreeze `mlp.W_out`/`mlp.b_out` on cfg blocks only. Adam `LR=1e-4`, 4000 steps over shuffled calib_ids (random shuffled with seed). At each `EVAL_EVERY=100`: record `eval_ppl`; track `best_ppl`/`best_state = {id(p): p.detach().clone() for p in trainable}`. After loop, restore best state. Record `ppl_final = eval_ppl` (== best).
4. Aggregate per config: `mean_ppl`, `std_ppl`, `per_seed_ppl`, `per_seed_init_ppl`, `per_seed_history`. Delta vs 46.556 baseline.
5. Verdict logic (applied to `mean_ppl` of each config vs 46.556):
   - If `max(mean_A, mean_B) - 46.556 ≥ 2.0` → `CONFIRMED` (placement matters).
   - Elif `abs(mean_A - 46.556) ≤ 0.8 and abs(mean_B - 46.556) ≤ 0.8` → `REFUTED` (count dominates).
   - Else → `INCONCLUSIVE`.
6. Write `autoresearch/results/cycle_242_early_vs_late.json` with schema: `{cycle:242, item:252, verdict, baseline_1_4_7:46.556, configs:{"1_2_3":{mean,std,per_seed,per_seed_init,per_seed_history,delta_vs_baseline},"8_9_10":{...}}, config:{K,N_CALIB,N_EVAL,SEQ_LEN,LR,N_STEPS,EVAL_EVERY,SEEDS}}`. Write `autoresearch/results/cycle_242_summary.md` comparing both means + stds to the 46.556 baseline with per-seed table and verdict rationale.

## Verification
- Pre-run: `python -c "import ast; ast.parse(open('experiments/early_vs_late_3block_cycle242.py').read())"`.
- Sanity: `ppl_init` of [1,2,3] ≈ cycle-206-era topk+lstsq range (50-70); final ≤ `ppl_init`; seeds within a few PPL of each other (else flag instability).
- Cross-check: [1,4,7] baseline not re-run; rely on cycle 210's 46.556 ± 0.028.
