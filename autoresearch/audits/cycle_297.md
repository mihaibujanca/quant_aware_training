# Cycle 297 Audit — Item 307

**Audit status: PASS**

## Scope
Verify cycle 297 execution against item 307's verbatim verdict thresholds; verify the per-layer eff_rank trajectory recorded matches an independent recomputation; verify per-seed verdict logic; verify the substantive interpretation matches the data.

## Verdict verification

Item 307 thresholds (verbatim from agenda / brief):
- **CONFIRMED gradual** iff per-layer eff_rank curves are monotone-decreasing across the entire training run AND no layer drops by ≥50% within any 100-step window.
- **CONFIRMED phase-transition** iff any layer's eff_rank drops by ≥50% within a 100-step window.
- **CONFIRMED init-rank** iff initial eff_rank is already ≤5 for ≥6 layers.
- **REFUTED** iff eff_rank doesn't collapse at all.

Auditor independently recomputed all four threshold counts from `records[*].snapshots[*].eff_rank` for each seed:

| seed | n_init_le5 | max_rel_drop | monotone non-increasing | refuted-no-collapse | flags |
|-----:|----------:|-------------:|------------------------:|--------------------:|-------|
| 42   | 12/12     | 0.3246       | False                   | False               | CONFIRMED-init-rank |
| 137  | 12/12     | 0.2021       | False                   | False               | CONFIRMED-init-rank |
| 2024 | 12/12     | 0.3063       | False                   | False               | CONFIRMED-init-rank |

Threshold evaluation per seed:
- CONFIRMED-init-rank requires init eff_rank ≤ 5 in ≥ 6 layers. 12/12 ≥ 6 for every seed. ✓ fires
- CONFIRMED-phase requires any 100-step window with relative drop ≥ 0.50. Global max across all 3 seeds × 12 layers = **0.3246** (seed 42 layer 5), well below 0.50. ✗ does not fire
- CONFIRMED-gradual requires monotone non-increasing AND no ≥50% drop. None of the 3 seeds is monotone non-increasing (final > init in essentially every layer). ✗ does not fire on either condition
- REFUTED requires `|init - final| ≤ 0.1` for all layers. Differences are clearly larger (e.g. seed 42 layer 1: 3.026 → 3.583, |Δ|=0.557; seed 2024 layer 3: 1.129 → 2.746, |Δ|=1.617). ✗ does not fire

Per-seed flag = `['CONFIRMED-init-rank']` (single clause). Overall (≥2 of 3 seeds agreeing): **CONFIRMED-init-rank** (3/3). ✓ matches manifest, exec_result, and JSON `overall_verdict`.

## Per-snapshot eff_rank — independent reproduction (init, seed 42, layer 0)

Auditor independently rebuilt the model with `torch.manual_seed(42); aleph.models.MLP(2, 32, 1, depth=12, ReLU)`, applied `make_spirals(2000, 0.5, 3, random_state=42) * 2.0`, generated the deterministic 1000/1000 train/held split with `torch.Generator().manual_seed(42)`, ran the first Linear+ReLU on the 1000-pt train slice, formed C = Y^T Y / 1000, computed `torch.linalg.eigvalsh`, normalized, and computed `exp(-Σ p_i log p_i)`:

- Auditor reproduction: **2.777405331**
- JSON `records[0].snapshots[0].eff_rank[0]`: **2.777404999**
- |Δ| = 3.31e-07 (float32 vs float64 noise floor) ✓

The eff_rank computation is implemented correctly and the snapshot pipeline records the right values.

## Trainable param count

Auditor independent recompute via `sum(p.numel() for p in MLP(2, 32, 1, depth=12).parameters() if p.requires_grad)` = **11,745**, matches `records[0].n_params=11745`. Script asserts this at runtime. ✓ (Same value verified in cycle 295 audit.)

## Convergence check (CLAUDE.md "verify task performance before analysis")

Per-seed final train_acc (from JSON `records`):
- seed 42:   12000 epochs, train_acc=0.9800 → ≥ 0.95 ✓
- seed 137:  12000 epochs, train_acc=0.9700 → ≥ 0.95 ✓
- seed 2024: 12000 epochs, train_acc=0.9870 → ≥ 0.95 ✓

All 3 nets cleared the 95% target. None hit `EARLY_STOP_ACC=0.99` so all ran to MAX_EPOCHS=12000 by design — this is intentional and lets us see whether anything happens AFTER convergence. The eff_rank measurements are taken on properly-trained networks. (Cycle 295 spirals required 6075–11325 epochs to clear 0.95; cycle 297 uses MAX_EPOCHS=12000 which covers that range and runs 685–5925 additional post-convergence epochs to capture trajectory continuation.)

## Plan adherence

Plan (cycle_297.md) vs execution (verified from JSON `config` and `experiments/eff_rank_trajectory_cycle297.py`):
- HIDDEN=32 ✓, DEPTH=12 ✓, N_SAMPLES=2000 ✓ (1000 train / 1000 held per seed via deterministic torch.Generator split)
- LR=1e-3 ✓, MAX_EPOCHS=12000 ✓, SNAPSHOT_EVERY=100 ✓
- INPUT_SCALE=2.0 ✓ (MEMORY.md depth-12 spirals trainability recipe)
- spirals: noise=0.5 ✓, n_turns=3 ✓
- SEEDS=[42, 137, 2024] ✓
- Adam, BCEWithLogits, full-batch, CPU ✓
- aleph.models.MLP(2, 32, 1, depth=12, ReLU) ✓
- Eff-rank formula identical to cycle 295 ✓
- Snapshot at epoch 0 + every 100 epochs through MAX_EPOCHS = 121 snapshots/seed ✓
- Output paths match plan ✓

## Substantive interpretation — audit

The verdict is **CONFIRMED-init-rank** per the verbatim thresholds, but the substantive content of the result is sharper than the verdict alone conveys, and was documented appropriately in the manifest, exec_result, and PAST entry: **training does NOT collapse eff_rank on this 2D-input testbed; it LIFTS it**. In every (seed, layer) cell, the trajectory is upward (final > init or essentially equal). Layers START at eff_rank ≈ 1.0 (already collapsed at random init by ReLU contraction on a 2D input) and training mildly twists the activation manifold UP to eff_rank ≈ 1.2-2.8 over 12000 epochs.

Auditor verifies this directly:
- Layers 7-11 at init across all 3 seeds: eff_rank = [1.000, 1.000, 1.000, 1.000, 1.000] (uniformly 1.000)
- Same layers at final epoch: range 1.12-2.84 (always strictly above the init value)
- The "epoch_at_1.5x_final" metric is therefore undefined for ALL 36 (3 seeds × 12 layers) cells, because init < 1.5*final everywhere — the trajectory never crosses the threshold from above. The script correctly returns None for all 36 cells and the manifest/exec_result correctly explain why.

The "phase transition vs gradual" question on this testbed reduces to "did anything go DOWN at all?" The largest single-window relative drop across all 3 seeds × 12 layers × ~120 windows = **0.3246** (seed 42, layer 5). This is a real downward fluctuation but well below the 0.50 phase-transition threshold. The other large drops (seed 42 layers 5/6/7, around 0.30-0.33) are also fluctuations within an overall upward trajectory, not a sustained collapse.

This is the same conclusion as cycle 295's init census, now confirmed by the trajectory itself rather than just the endpoint comparison. The 2D-input spirals testbed cannot answer cycle 278's training-dynamics question because there is no rank to lose. The honest substantive reading is: the spirals testbed is the wrong control for the cycle 278 question; the principled follow-up is either (A) a higher-dim input lift (100D embedding from MEMORY.md) so init eff_rank can plausibly be > 5 and actually drop, or (B) instrumenting GPT-2 directly during cycle 280's 4000-step Adam retrain to see whether eff_rank moves at all during the LASER fine-tune that produced the cycle 280 result. Both follow-ups are flagged in the exec_result and manifest. The auditor concurs with both recommendations and notes that follow-up B is more directly aligned with the actual research question.

## Cross-check vs cycle 295

Cycle 295 spirals init_mean per layer (3 seeds, full 2000 sample): [2.994, 2.705, 1.874, 1.187, 1.058, 1.010, 1.002, 1.001, 1.000, 1.000, 1.000, 1.000].
Cycle 297 init values (1000 train subset per seed):
- mean across 3 seeds: layer 0 = (2.78+2.86+3.30)/3 = 2.97; layer 1 = (3.03+2.32+2.70)/3 = 2.68; layer 7+ = 1.00 (all)
- The cycle 297 init means are within the same range (small differences from the 2000 → 1000 random subsample). Same qualitative shape: layer 0 ≈ 3, layer 7+ pinned at 1.00. ✓

Cycle 295 spirals post_mean per layer: [3.143, 2.859, 2.106, 2.088, 2.151, 1.937, 1.986, 2.072, 2.049, 2.032, 1.756, 1.736].
Cycle 297 final values fall in the same range (cycle 297 ran ~685-5925 additional epochs past 95%, so values may differ slightly per-seed). ✓

## Artifact existence

- `experiments/eff_rank_trajectory_cycle297.py` — exists, ~340 lines
- `autoresearch/plans/cycle_297.md` — exists, 8 H2 headings (Dependencies, Item 307 (verbatim Hypothesis), Approach, Files, Config, Verification, Verdict logic (item 307 verbatim), Baselines)
- `autoresearch/briefs/cycle_297.md` — exists (from brief stage)
- `autoresearch/results/cycle_297_eff_rank_trajectory.json` — exists, valid JSON, overall_verdict="CONFIRMED-init-rank", per_seed_verdicts populated
- `autoresearch/results/cycle_297_summary.md` — exists, per-seed sections + overall
- `autoresearch/results/cycle_297_trajectory.png` — exists, 4-panel figure
- `autoresearch/results/cycle_297_manifest.txt` — exists, commit=1dcb03b
- `autoresearch/results/cycle_297_exec_result.txt` — exists
- `autoresearch/logs/cycle_297_run.log` — exists, captures per-seed convergence + eff_rank lists + verdicts
- Primary commit `1dcb03b` exists in git log
- `experiments.tsv` row 297 appended with 8 columns; hash patched from `pending` to `1dcb03b` ✓
- `agenda.md`: item 307 deleted from NEXT P0; added to PAST as Cycle 297 ✓ (auditor verified item 307 no longer appears in NEXT and item 308 is now the first P1 entry)

## Corrections

None on the verdict. The CONFIRMED-init-rank verdict is the literal verbatim threshold result and the substantive answer is consistent with it. The 2× input scale (MEMORY.md recipe, inherited from cycle 295) is the project's documented recipe for depth-12 spirals trainability and is honestly documented in the script header, plan, manifest, exec_result, and PAST entry.

## Final status

**PASS** — keep provisional outcome CONFIRMED-init-rank. The verdict logic is correctly applied to the verbatim thresholds; the per-seed counts match independent recomputation; the eff_rank formula is verified to ~3e-7 against an independent reproduction at the seed 42 layer 0 init cell; all 3 nets converged ≥95% train acc on properly-trained networks; the substantive finding (training LIFTS rather than collapses eff_rank on the 2D testbed; the "collapse to rank-2" framing has the wrong sign here) is documented appropriately and the implications for cycle 278's trajectory question are flagged for the next planner.

Auditor note: cycle 297 closes the loop opened by cycle 295. Both cycles agree (via init census and via trajectory) that the 2D-input testbed cannot answer cycle 278's training-dynamics question because there is no opportunity to collapse — the testbed starts collapsed. The principled follow-ups are both in scope for the next agenda refresh: (A) higher-dim input lift, or (B) instrument GPT-2 directly during cycle 280's Adam retrain. Follow-up B is more directly aligned with the research question and more likely to produce a clean answer about whether eff_rank moves during the LASER fine-tune. Items 308 (SDF) and 309 (cycle 280 retrain decomposition) are both still in NEXT and partially address this.
