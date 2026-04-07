# Cycle 295 Audit — Item 305

**Audit status: PASS**

## Scope
Verify cycle 295 execution against item 305's verbatim Hypothesis and verdict thresholds; verify arithmetic and cross-check against the raw JSON via independent eff_rank recomputation; verify that the substantive INCONCLUSIVE verdict reflects the literal verbatim verdict logic and not auditor interpretation.

## Verdict verification

Item 305 thresholds (verbatim from cycle 295 brief / agenda):
- CONFIRMED iff post-training eff_rank ≤ 3 in ≥ 8 of 12 layers AND init eff_rank ≥ 20 in ≥ 8 of 12 layers
- REFUTED iff post-training eff_rank ≥ 10 in ≥ 6 layers
- INCONCLUSIVE otherwise

Measured (from JSON `aggregates`, independently recomputed from `records`):

| dataset | post ≤ 3 | init ≥ 20 | post ≥ 10 | verdict     |
|---------|---------:|----------:|----------:|-------------|
| spirals |    11/12 |      0/12 |      0/12 | INCONCLUSIVE |
| moons   |    12/12 |      0/12 |      0/12 | INCONCLUSIVE |

Threshold evaluation:
- Spirals CONFIRMED requires (post ≤ 3 in ≥ 8) AND (init ≥ 20 in ≥ 8). Part A: 11 ≥ 8 ✓. Part B: 0 < 8 ✗ → CONFIRMED clause does NOT trigger ✓.
- Spirals REFUTED requires (post ≥ 10 in ≥ 6). Measured 0 < 6 → REFUTED clause does NOT trigger ✓.
- Spirals → INCONCLUSIVE ✓.
- Moons same shape → INCONCLUSIVE ✓.
- Overall (CONFIRMED iff both CONFIRMED; REFUTED iff either REFUTED; else INCONCLUSIVE): both INCONCLUSIVE → overall **INCONCLUSIVE** ✓ (matches manifest, exec_result, JSON).

## Per-layer eff_rank — independent recomputation

For spirals seed 42 layer 0 init, the auditor independently rebuilt the model with `torch.manual_seed(42); aleph.models.MLP(2, 32, 1, depth=12, ReLU)`, ran the spirals (×2.0 input scale per MEMORY.md recipe) through the first Linear+ReLU, formed C = Y^T Y / 2000, computed `torch.linalg.eigvalsh`, normalized, and computed `exp(-Σ p_i log p_i)`.

- Auditor reproduction: **2.756040417**
- JSON `records[0].init_eff_rank[0]`: **2.756041403**
- |Δ| = 9.86e-07 (float64-vs-float32 noise floor) ✓

The eff_rank computation is implemented correctly.

## Per-layer means — independent aggregation

Auditor independently aggregated `records[*].init_eff_rank` and `records[*].post_eff_rank` per dataset by `numpy.mean(axis=0)`:

spirals init_mean = [2.994, 2.705, 1.874, 1.187, 1.058, 1.010, 1.002, 1.001, 1.000, 1.000, 1.000, 1.000]  ✓ matches JSON `aggregates.spirals.init_mean`
spirals post_mean = [3.143, 2.859, 2.106, 2.088, 2.151, 1.937, 1.986, 2.072, 2.049, 2.032, 1.756, 1.736]  ✓
moons   init_mean = [2.179, 2.181, 1.645, 1.154, 1.052, 1.009, 1.001, 1.000, 1.000, 1.000, 1.000, 1.000]  ✓
moons   post_mean = [2.153, 1.463, 1.234, 1.178, 1.204, 1.300, 1.406, 1.520, 1.611, 1.503, 1.358, 1.405]  ✓

Threshold-counting independent recomputation:
- spirals post layers ≤ 3.0: layer 0 = 3.143 fails; layers 1–11 all ≤ 3.0 → 11/12 ✓
- moons post layers ≤ 3.0: all 12 ≤ 2.18 → 12/12 ✓
- spirals init layers ≥ 20: max init = 2.994, none meet → 0/12 ✓
- moons init layers ≥ 20: max init = 2.181, none meet → 0/12 ✓
- spirals post layers ≥ 10: max post = 3.143, none meet → 0/12 ✓
- moons post layers ≥ 10: max post = 2.153, none meet → 0/12 ✓

All threshold counts match the JSON-stored verdicts exactly.

## Convergence check (CLAUDE.md "verify task performance before analysis")

Per-seed final train_acc (from JSON `records`):
- spirals seed 42:   epochs=11325, train_acc=0.9520, converged ✓
- spirals seed 137:  epochs= 9625, train_acc=0.9505, converged ✓
- spirals seed 2024: epochs= 6075, train_acc=0.9510, converged ✓
- moons seed 42:     epochs=   75, train_acc=0.9510, converged ✓
- moons seed 137:    epochs=  100, train_acc=0.9990, converged ✓
- moons seed 2024:   epochs=  100, train_acc=0.9985, converged ✓

All 6 nets cleared the 95% target. No undertrained measurements. The cycle 295 spirals runs required ~6k–11k epochs at lr=1e-3 with the 2× input scale; the brief did not specify input scale but the project's documented recipe (MEMORY.md) requires it for depth-12 spirals trainability — script applies it and the script header documents the deviation. The first run (without 2× scale) failed to converge in 5000 epochs (peak 92.75% spirals seed 42); the rerun with the 2× scale passes for all 3 spirals seeds inside the 20k cap. Honest documentation of the deviation in `cycle_295_exec_result.txt`, `cycle_295_manifest.txt`, and the cycle 295 PAST entry.

## Trainable param count

JSON `records[0].n_params = 11745`. Independent recompute via `sum(p.numel() for p in MLP(2, 32, 1, depth=12).parameters() if p.requires_grad)` returned **11745** (12 hidden Linears + 1 output Linear + biases; the QuantStub/DeQuantStub identity wrappers contribute zero params). Plan stated 11,745 (the 13K figure in the brief was approximate). ✓

## Config / plan adherence

Plan (cycle_295.md) vs execution (verified from JSON `config` and `experiments/spirals_moons_eff_rank_cycle295.py`):
- HIDDEN=32 ✓, DEPTH=12 ✓, N_SAMPLES=2000 ✓
- LR=1e-3 ✓, MAX_EPOCHS=20000 (plan said 5000; bumped after first-run undertraining), TRAIN_ACC_TARGET=0.95 ✓
- SEEDS=[42, 137, 2024] ✓
- spirals: noise=0.5 ✓, n_turns=3 ✓, input_scale=2.0 (added per MEMORY.md, post-plan) ✓
- moons: noise=0.1 ✓
- Adam, BCEWithLogits, full-batch, CPU ✓
- aleph.models.MLP(2, 32, 1, depth=12, ReLU) ✓ (brief reference to `aleph.make_mlp` is wrong; that symbol does not exist in the repo — auditor verified `Grep make_mlp aleph` returns no matches; substitution to `aleph.models.MLP` documented in plan + manifest)
- Eff-rank formula: C = Y^T Y / N, eig via `torch.linalg.eigvalsh`, clamp to 0, normalize, exp(-Σ p log p) — matches the plan ✓ (auditor independently reproduced layer 0 spirals seed 42 init to 9.86e-07 tolerance)
- Output paths match plan ✓

## Substantive interpretation — audit

The verdict is INCONCLUSIVE per the verbatim thresholds, but the substantive content of the result is striking and was documented appropriately in the manifest, exec_result, and PAST entry: **the post-training collapse is real (11/12 spirals, 12/12 moons layers ≤ 3.0) but the collapse cannot be attributed to training** because init eff_rank is already ≤ 3 in 12/12 layers for both datasets — and reaches 1.0 by layer 7. With a 2D input, no random Gaussian + ReLU stack can create new effective dimensions; eff_rank is inherited from the input and only contracted by ReLU. Hanin & Rolnick's N^d activation-pattern bound (lit map line 594) is the right reference: with d=2 a depth-12 32-wide ReLU stack cannot exceed eff_rank ~2 anywhere.

This corresponds to item 307's "CONFIRMED-init-rank" outcome category, applied to its precondition. The exec_result and manifest correctly call this out and recommend a higher-dim input lift (e.g. 100D embedding from MEMORY.md) as the principled item 305 follow-up. The auditor concurs:
- Item 305 cannot answer the cycle 278 question ("training-induced or generic?") on a 2D-input testbed because there is no opportunity to collapse — the testbed starts collapsed. The honest verdict is INCONCLUSIVE per the verbatim thresholds AND the substantive reading (the experiment failed as a control, not as an empirical measurement).
- Item 306 (LASER on spirals at the smallest-eff_rank layer) is well-defined with these results: layers 7–11 sit at eff_rank ≈ 1.7–2.1 post-training (spirals), so SVD truncation to rank 2 should match float on a provably-2D task — and the substantive predictions of item 306 are unchanged.
- Item 307 (eff_rank trajectory through training) will hit the same 2D input bottleneck. The trajectory will start at eff_rank ≈ 1.8 in early layers (drops to 1.0 by layer 7) and post-training the early layers move to ≈ 2.1. The "phase transition vs gradual" question on this testbed collapses to "did the small bumps in layers 0–3 grow gradually or in a step?" with very little dynamic range (∆ ≈ 0.3 in layer 0 from 2.99 to 3.14). Worth flagging that item 307 should be either re-scoped to a higher-dim input lift, or run on a different testbed where init eff_rank can meaningfully drop.

## Artifact existence

- `experiments/spirals_moons_eff_rank_cycle295.py` — exists, ~280 lines
- `autoresearch/plans/cycle_295.md` — exists, 8 H2 headings (Dependencies, Item 305 (verbatim Hypothesis), Approach, Files, Config, Verification, Verdict logic (item 305 verbatim), Baselines)
- `autoresearch/briefs/cycle_295.md` — exists (from brief stage)
- `autoresearch/results/cycle_295_spirals_moons_eff_rank.json` — exists, valid JSON, verdict="INCONCLUSIVE", per-dataset verdicts populated
- `autoresearch/results/cycle_295_summary.md` — exists, per-dataset tables
- `autoresearch/results/cycle_295_manifest.txt` — exists, commit=0cc1359
- `autoresearch/results/cycle_295_exec_result.txt` — exists, commit not embedded but cycle clearly identified
- `autoresearch/logs/cycle_295_run.log` — exists, captures per-seed eff_rank lists and per-dataset verdicts
- Primary commit `0cc1359` exists in git log; bookkeeping commit `934db3c` created with manifest + tsv hash patch
- `experiments.tsv` row 295 appended with 8 columns and hash patched from `pending` to `0cc1359` ✓
- `agenda.md`: item 305 deleted from NEXT P0, added to PAST as Cycle 295 ✓ (auditor verified item 305 no longer appears in NEXT and item 306 is now the first P0 entry)

## Corrections

None on the verdict. One process note: the plan stated max_epochs=5000 and the first run found that spirals does not converge inside that budget at lr=1e-3 without the 2× input scale documented in MEMORY.md. The script was edited to add `SPIRALS_INPUT_SCALE = 2.0` and `MAX_EPOCHS = 20000`, all 6 fits then converged ≥95% acc on the second run, and the deviation is documented in the script header, plan, manifest, and exec_result. Per CLAUDE.md "verify task performance before analysis", the eff_rank measurements were taken on properly-trained networks. No experimental data from the first run is used.

## Final status

**PASS** — keep provisional outcome INCONCLUSIVE. The verdict logic is correctly applied to the verbatim thresholds; the per-layer means and threshold counts match independent recomputation; the eff_rank formula is verified to ~1e-6 against an independent reproduction; all 6 nets converged ≥95% train acc on properly-trained networks; the substantive finding (collapse is input-dimension-induced, not training-induced) is documented appropriately and the implications for items 306 and 307 are flagged for the next planner. The 2× input scale deviation from the brief's literal recipe is the project's documented recipe for depth-12 spirals trainability and is honestly documented in all artifacts.

Auditor note: the result is a clean methodological negative — item 305 was the wrong control for cycle 278's question, because the 2D input precludes the very high-rank init that the CONFIRMED clause was checking for. The experiment correctly executed; the experimental design did not anticipate the input-dim bottleneck. A follow-up at higher input dim (e.g. 100D embedding) would actually answer the question. Items 306 and 307, both of which depend on item 305, should be re-scoped or augmented with the higher-dim input lift before being run; this is the most consequential takeaway for the next planner.
