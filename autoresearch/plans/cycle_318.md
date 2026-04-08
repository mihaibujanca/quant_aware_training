# Task
Item 333 (P0): Output-covariance eff_rank census on parameter-golf 13L_640d ReLU² checkpoint.

# Hypothesis
Verbatim from item 333: parameter-golf has at least one block with `eff_rank ≤ 20` AND no block has `eff_rank ≥ 600` (the rank-collapse phenomenon happens in some block but not all; same U-shape pattern as GPT-2 with shifted endpoints). The cycle 280 LASER recipe should transfer to parameter-golf.

# Pre-flight Results
- P1: not duplicate. Cycle 278 measured GPT-2; cycle 303 was PLAN INVALID without doc.
- P2: cross-project doc exists at `autoresearch/cross_project/parameter_golf.md`. Parameter-golf venv exists. Checkpoint exists.
- P3 ≤30 min: ~30s pgolf forward + Python overhead.
- P4: no program/prompt mods.

# Steps
1. **Step 0** (REQUIRED): `cat autoresearch/cross_project/parameter_golf.md` — DONE in brief stage.
2. **Step 1**: write `/Users/mihai/Projects/parameter-golf/local_scripts/pgolf_eff_rank_census.py` per cross-project doc section 5 verbatim. Reuse `analyze_checkpoint.load_model` and `load_val_tokens`. Monkey-patch `MLP.__call__` to capture pre-`mlp_scale` outputs (per doc section 4 + gotcha 4). Print result JSON to stdout. Args: `--checkpoint`, `--n-tokens`, `--seq-len`.
3. **Step 2**: write `experiments/pgolf_eff_rank_census_cycle318.py` (qat side) per doc section 3 template:
   - subprocess.run([PGOLF_PYTHON, PGOLF_SCRIPT, "--checkpoint", "/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz", "--n-tokens", "25600", "--seq-len", "1024"], capture_output=True, text=True, timeout=600).
   - Parse stdout JSON.
   - Save to `autoresearch/results/cycle_318_pgolf_eff_rank.json` and `autoresearch/results/cycle_318_summary.md`.
4. **Step 3**: run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/pgolf_eff_rank_census_cycle318.py`. Verify JSON contains 13-block profile.
5. **Step 4**: build summary with per-block table (eff_rank, rank_95, rank_99, top-5 eigenvalues), verdict (CONFIRMED-collapse / CONFIRMED-no-slack / CONFIRMED-different-shape / INCONCLUSIVE per criteria), explicit comparison row to cycle 278.

# Validation
- pgolf script JSON has 13 blocks.
- Cross-checks: trainable params = 0.

# Artifacts
- experiments/pgolf_eff_rank_census_cycle318.py
- parameter-golf/local_scripts/pgolf_eff_rank_census.py
- autoresearch/results/cycle_318_pgolf_eff_rank.json
- autoresearch/results/cycle_318_summary.md
- autoresearch/results/cycle_318_manifest.txt

# Fallbacks
- If pgolf forward pass too slow: reduce n-tokens to 12800.
- If brotli load fails: use the .npz file in logs/ (per doc gotcha 7).
- If MLP.__call__ signature differs: inspect train_gpt_mlx.py before patching.

# Update bookkeeping
- Move 333 from NEXT/P0 to PAST/Cycle 318.
- TSV row, commit, STATUS.
