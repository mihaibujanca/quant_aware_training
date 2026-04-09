# Cycle 325 — Item 338: LeakyReLU²-Trained Checkpoint Census

## Task
Verify whether cycles 318/319 ran on a ReLU²- or LeakyReLU²-trained checkpoint. If already lrelu2-trained, document the mislabeling, correct cycle 318/319 summaries, and close item 338 as MOOT. If truly relu2-trained, train an lrelu2 checkpoint and re-run census.

## Hypothesis
The canonical checkpoint `13L_640d_322_7k_wd70_batch20k_best.npz` is ALREADY lrelu2-trained (its summary.json shows `"mlp_act": "lrelu2"`). If confirmed, cycles 318/319 were already measuring a true lrelu2-trained model, and item 338's premise (that they measured the "wrong" activation distribution) is invalidated. The localization at b4/b5/b12 is real, not a predictive-perturbation artifact.

## Pre-flight Results
- P1 (duplicate): No prior cycle checked the checkpoint's actual mlp_act config.
- P2 (dependencies): cycle 318/319 infrastructure done, checkpoint exists.
- P3 (concrete): Yes — verify config, correct records, close item.
- P4 (modifies program.md): No.

## Steps
1. `cat autoresearch/cross_project/parameter_golf.md` (required Step 0).
2. Load `parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_summary.json` and extract `config.mlp_act`. Print the value.
3. If `mlp_act == "lrelu2"` (expected): item 338 is MOOT. Cycles 318/319 already measured an lrelu2-trained model, mislabeled as "ReLU²". Write `autoresearch/results/cycle_325_summary.md`:
   - State the finding: canonical checkpoint was always lrelu2-trained.
   - Reproduce the cycle 318 eff_rank profile and cycle 319 curvature residual baselines verbatim (these ARE the lrelu2-trained census).
   - Verdict: **MOOT-already-lrelu2-trained** — "the LeakyReLU²-trained checkpoint census requested by item 338 was already performed in cycles 318 and 319, which mislabeled the checkpoint as ReLU²-trained."
   - Errata: cycles 318 and 319 summaries say "ReLU²" but the checkpoint is lrelu2.
4. If `mlp_act != "lrelu2"` (fallback): train lrelu2 checkpoint per brief Step 2-3, then run census (see Fallbacks).
5. Write `autoresearch/results/cycle_325_exec_result.txt` and `cycle_325_manifest.txt`.
6. `git add autoresearch/results/cycle_325_* && git commit -m "[autoresearch] cycle 325: item 338 — MOOT, canonical checkpoint was already lrelu2-trained (cycles 318/319 mislabeled)"`.

## Validation
- `python3 -c "import json; print(json.load(open('.../summary.json'))['config']['mlp_act'])"` prints `lrelu2`.
- Cycle 325 summary cites the exact JSON field and value as evidence.
- No training run needed (zero compute).

## Artifacts
- `autoresearch/results/cycle_325_summary.md`
- `autoresearch/results/cycle_325_exec_result.txt`
- `autoresearch/results/cycle_325_manifest.txt`

## Fallbacks
- If `mlp_act == "relu2"` (premise holds after all): train with `MLP_ACT=lrelu2 NUM_LAYERS=13 MODEL_DIM=640 ITERATIONS=7000 TRAIN_BATCH_TOKENS=20480 GRAD_ACCUM_STEPS=1 WARMDOWN_ITERS=4900 MAX_WALLCLOCK_SECONDS=0 MLP_MULT_PER_LAYER=3,3,3,3,3,3,3,2,2,3,2,2,0.5 /Users/mihai/Projects/parameter-golf/.venv/bin/python /Users/mihai/Projects/parameter-golf/train_gpt_mlx.py`. Save to `parameter-golf/logs/13L_640d_lrelu2_for_cycle338.npz`. Then subprocess-invoke `pgolf_eff_rank_census.py` and `pgolf_curvature_residual.py` on it. Apply item 338 verdict thresholds: "CONFIRMED-localization-stable if both eff_rank and (lrelu², relu²) rank_95 profiles match within ±30% relative at every block", "CONFIRMED-shifted-localization if the (lrelu², relu²) rank_95 ≤ 50 set differs by ≥1 block", "REFUTED-no-localization if the lrelu²-trained model has rank_95 > 50 at all blocks".
