# Task
Multi-checkpoint eff_rank census on 3 parameter-golf polished artifacts — is the b3 collapse run-stable?

# Hypothesis
The lowest-eff_rank block position is stable across checkpoints with different layer counts and training configs.

**Confirmed/Refuted/Inconclusive criteria (verbatim from agenda item 335):**
- **CONFIRMED-stable** if the lowest-eff_rank block is the same in all 3 checkpoints (or within ±1 layer): b3 is the canonical LASER target across the parameter-golf model family. Items 340-342 are well-targeted.
- **REFUTED-checkpoint-specific** if the lowest-eff_rank block varies by ≥3 layers across checkpoints: the LASER target depends on the specific run, items 340-342 need to be re-targeted per checkpoint.
- **INCONCLUSIVE** otherwise (some shift but small).

# Pre-flight Results
- P1-P5: unchanged from round 0 (all PASS).
- P6 REVISED: `load_model_v2` does not pass `num_encoder_layers` to GPT. The 10L multiskip checkpoint was trained with `num_encoder_layers=6` (skip_weights shape (6,640)), but default `10//2=5` gives 8 skip connections (multi-skip path), crashing on shape mismatch. Fix: infer `num_encoder_layers` from `skip_weights.shape[0]` inside `load_model_v2`, or add `--num-encoder-layers` CLI arg to `pgolf_eff_rank_census.py`.

# Steps
0. **Read cross-project doc**: `cat autoresearch/cross_project/parameter_golf.md`.
1. **Fix `load_model_v2`** in `parameter-golf/local_scripts/pgolf_eff_rank_census.py`: after determining `hp.num_layers`, read `skip_weights` shape from the checkpoint. Brute-force `num_encoder_layers` from 1 to `num_layers` until `_build_skip_map` total skip connections equals `skip_weights.shape[0]`. Pass the inferred value to the GPT constructor via `num_encoder_layers=inferred`. If no `skip_weights` key exists in checkpoint, leave default (0 = auto).
2. **Copy wrapper**: `cp experiments/pgolf_eff_rank_census_cycle318.py experiments/pgolf_eff_rank_multi_cycle335.py`.
3. **Modify wrapper** to loop over 3 checkpoints:
   - `13L_640d_322_7k_wd70_batch20k_best.npz` (cycle 318 baseline)
   - `wd70_11L_5x_polished.npz` (11L)
   - `10L_640d_multiskip_22k_wd70_polished.npz` (10L multiskip)
   Each invocation: `subprocess.run([PGOLF_PYTHON, PGOLF_SCRIPT, "--checkpoint", ckpt, "--n-tokens", "25600"])`.
   Collect all 3 JSON results. Write `cycle_335_pgolf_eff_rank_multi.json` (array of 3 results).
4. **Build summary**: `cycle_335_summary.md` with side-by-side table (rows=blocks, columns=checkpoints). For non-matching depths, use "—" for missing blocks. Highlight min-eff_rank block per checkpoint. Compute relative position (block_index / n_blocks) for cross-depth comparison.
5. **Verdict logic**: find min-eff_rank block index per checkpoint. For cross-depth comparison, use fractional position (idx/n_blocks). CONFIRMED-stable if all min-positions within ±1 block (for same depth) or ±0.1 fractional position (cross-depth). REFUTED if ≥3-layer shift (same depth) or ≥0.3 fractional shift.
6. **Run**: `python experiments/pgolf_eff_rank_multi_cycle335.py`.
7. **Commit**: `git add parameter-golf/local_scripts/pgolf_eff_rank_census.py experiments/pgolf_eff_rank_multi_cycle335.py autoresearch/results/cycle_335_*` then commit.
8. **Update agenda.md**: move item 335 from NEXT to PAST EXPERIMENTS with commit hash and artifact paths.

# Validation
- Each subprocess must exit 0 and print valid JSON to stdout.
- Each result JSON must have `n_blocks` matching checkpoint (13, 11, 10 respectively).
- Cycle 318 baseline should reproduce: b3 eff_rank near 4.89 (within ±0.5 for same tokens).
- 10L checkpoint must load without crash (verifies Step 1 fix).

# Artifacts
- `parameter-golf/local_scripts/pgolf_eff_rank_census.py` — patched `load_model_v2` (num_encoder_layers inference)
- `experiments/pgolf_eff_rank_multi_cycle335.py` — qat-side wrapper
- `autoresearch/results/cycle_335_pgolf_eff_rank_multi.json` — combined JSON
- `autoresearch/results/cycle_335_summary.md` — side-by-side table + verdict

# Fallbacks
- If brute-force `num_encoder_layers` inference fails (no value produces matching skip count): fall back to `--num-encoder-layers` CLI arg, hardcode `{10L: 6}` in wrapper.
- If subprocess times out: increase timeout to 1800s; if still fails, reduce n_tokens to 12800.
- If 11L checkpoint fails: substitute `13L_640d_322_7k_wd70_best.npz` (non-batch20k 13L variant) for a same-architecture different-training-run comparison.
