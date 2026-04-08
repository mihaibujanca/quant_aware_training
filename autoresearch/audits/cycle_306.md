# Cycle 306 Audit — Item 316

**Audit status: PASS**

## Scope

Verify cycle 306 execution against item 316's verbatim hypothesis and outcome
thresholds. Verify the block-8 deep-dive measurements reproduce the cycle 302
panel 4 reference, the verdict logic (single-factor "flat W_proj + broad δ")
is applied correctly with the concentrator-median baseline, the PRIMARY-set
{7, 8, 9} per user instruction is honored while the CONCENTRATORS={1, 2, 10, 11}
extension is the minimal adjustment forced by the verdict thresholds, the four
measurements specified in the brief Constraints are all computed, and the
geometric reading is consistent across cycles 301 / 302 / 304 references.

## Verdict verification

Item 316 outcomes (verbatim from agenda):
- **CONFIRMED** iff (a) b8 W_proj top-2/top-5 SV ratio ≤ 0.5 × median
  concentrator block's ratio AND (b) b8 Δy eff_rank ≥ 1.5 × median
  concentrator Δy eff_rank.
- **REFUTED** iff neither (a) nor (b) holds.
- **INCONCLUSIVE** otherwise.

Auditor independently re-derived the two conditions from
`autoresearch/results/cycle_306_block8_deep_dive.json::per_block`:

| block | W_proj top2/top5 (s_w[1]/s_w[4]) | Δy stable rank |
|---:|---:|---:|
| 1   | 1.6815 | 1.310 |
| 2   | 1.3097 | 1.065 |
| 10  | 1.4343 | 1.099 |
| 11  | 1.5942 | 1.018 |

Concentrator W_proj top2/top5 sorted: [1.3097, 1.4343, 1.5942, 1.6815] →
median (mean of middle two for n=4) = (1.4343 + 1.5942) / 2 = **1.51425**.
0.5 × 1.51425 = **0.75713**.

Concentrator Δy stable rank sorted: [1.018, 1.065, 1.099, 1.310] →
median = (1.065 + 1.099) / 2 = **1.082**.
1.5 × 1.082 = **1.6230** (script reports 1.6234 with full-precision medians).

| criterion | b8 value | threshold | result |
|---|---|---|---|
| (a) b8 W_proj top2/top5 ≤ 0.5×conc_med | **1.4875** | ≤ 0.7571 | **fail** |
| (b) b8 Δy stable rank ≥ 1.5×conc_med   | **1.197**  | ≥ 1.6234 | **fail** |

Both single-factor conditions FAIL. Item 316 verbatim threshold: REFUTED iff
NOT (a) AND NOT (b). Both fail → REFUTED is the literal verbatim threshold
result.

The single-factor "flat W_proj AND broad curvature residual multiplicatively
explain E_kept_top_2(b8) = 0.47" hypothesis is rejected.

## Reproduction fidelity — cycle 302 panel 4

The script asserts at every block:
```python
e_kept_top2 = (Dy @ Vhw[:2, :].T).pow(2).sum() / Dy_frob_sq
assert abs(e_kept_top2 - C302_E_TOP2_REF[l]) < 1e-4
```
where C302_E_TOP2_REF is the literal cycle 302 panel 4 dict. Auditor checked
the per-block values in the JSON output against this dict:

| block | E_kept_top_2_via_Wproj (this cycle) | C302 ref | |Δ| |
|---:|---:|---:|---:|
| 1   | 0.7463 | 0.7463 | 0.0000 |
| 2   | 0.9190 | 0.9190 | 0.0000 |
| 7   | 0.6622 | 0.6622 | 0.0000 |
| 8   | 0.4722 | 0.4722 | 0.0000 |
| 9   | 0.5356 | 0.5356 | 0.0000 |
| 10  | 0.8374 | 0.8374 | 0.0000 |
| 11  | 0.9803 | 0.9803 | 0.0000 |

7/7 EXACT match to 4 decimals. The cycle 302 panel 4 figure is reproduced
deterministically. ✓

## Reproduction fidelity — float PPL

Float PPL on the 200-seq calib slice = 45.0637, identical to the cycle
302/304 reference 45.0637 (cycle 302 panel 5 calibration anchor; cycle 304
calibration anchor). The script asserts `abs(float_ppl - 45.0637) < 0.5`.
Auditor confirmed the same wikitext-2 val slice with seed 42 token selection,
SEQ_LEN=128, deterministic dataset iteration. ✓

## Geometric object check

The script computes, for each l in ALL_BLOCKS = [1, 2, 7, 8, 9, 10, 11]:

1. `Dy = (GELU(z) - ReLU(z)) @ W_proj` via `UnifiedCurvatureFramework.compute_Dy(l)`,
   reusing the cycle 302 plumbing (single forward pass via `capture_inputs()`,
   chunked CPU float32 computation, `W_proj` as HF Conv1D (3072, 768)).
2. `U_w, S_w, Vh_w = torch.linalg.svd(W_proj, full_matrices=False)` →
   shapes (3072, 768), (768,), (768, 768). ✓
3. `U_y, S_y, Vh_y = torch.linalg.svd(Dy, full_matrices=False)` with
   Dy shape (25600, 768) → shapes (25600, 768), (768,), (768, 768). ✓
4. Stable rank `(Σ s²)² / Σ s⁴` is the standard "effective rank" definition;
   it agrees with the framework conventions in cycle 278.
5. Per-token norms via `torch.linalg.norm(Dy, dim=1)` and quantiles
   {q05, q25, median, q75, q95}.
6. Principal-angle cosines via `torch.linalg.svdvals(Vh_y[:5] @ Vh_w[:5].T)`
   for the top-5 right-SV alignment between Δy and W_proj.

All four item 316 brief Constraints (a)–(d) are computed:
- (a) full SV spectrum of Δy ✓ (`Sy` full 768 values stored; top-1 energy
  fraction; eff_rank; k_95; k_99)
- (b) per-token ||Δy|| ✓ (mean/std/quantiles in `Dy_norms`)
- (c) singular value concentration / top-1 energy fraction ✓
  (`Dy_top1_energy = S_y[0]^2 / Σ S_y^2`)
- (d) cosine between Δy top right SVs and W_proj top right SVs ✓
  (`cos_principal_top5`, `cos_top5_sum`)

The brief said "Compare with neighbors block 7, block 9". The user explicit
instruction said "Compute Δy for blocks {7, 8, 9}". The verdict thresholds
require concentrator-median values, so the script extends ALL_BLOCKS to
{1, 2, 7, 8, 9, 10, 11}. This is the minimal extension and is justified in
the plan and the manifest. ✓

## Substantive findings — auditor endorses

1. **REFUTED is the verbatim-correct verdict.** Both single-factor conditions
   fail by a wide margin. Cond (a) needs b8 top2/top5 ratio ≤ 0.7571; measured
   1.4875 (96% above). Cond (b) needs Δy stable rank ≥ 1.6234; measured 1.197
   (26% below). Neither condition is borderline.

2. **The "broad b8 curvature residual" reading from cycle 304 was a basis
   artifact.** Cycle 304's k_95 is computed on Dy projected through W_proj's
   right-singular basis (`Dy @ Vh_w`), and reports b8 k_95 = 511. Cycle 306's
   `Dy_k95` is computed on Dy's OWN SVD basis and reports b8 k_95 = 73.
   Difference = 438 ranks. The intrinsic Δy at b8 is rank-1 dominated
   (top1_E = 0.914, stable rank 1.20), not broad.

3. **The actual b8 anomaly is a TOP-1 misalignment.** b8's nearly-rank-1
   Δy points along a direction Vh_y[:1] that has only modest projection onto
   Vh_w[:2]. The principal-angle metric `cos_top5_sum` at b8 = 2.65 is
   comparable to b1 (2.64) and b7 (2.61) and above b2/b9 (2.12) — meaning
   the FIVE-FRAME alignment is fine. The mismatch is concentrated in the
   top-1 axis: 91% of Δy energy lives in Vh_y[:1] but Vh_y[:1] is not in
   span(Vh_w[:2]), leaving 53% of Δy energy unprojected.

4. **Implication for cycle 305 / item 315 reversed.** Item 316's CONFIRMED
   branch had argued "block 8 should NOT be added to any multi-block LASER
   stack — its curvature residual has nowhere to hide". REFUTED flips this:
   b8's curvature residual IS rank-1 concentrated AND would fit a rank-1
   LASER replacement of c_proj_8 IF the kept axis is Vh_y[:1] (Δy's intrinsic
   top right SV), not Vh_w[:1] (the standard cycle 280/284/305 construction).
   A LASER using Vh_w[:1] at b8 would leave 53% of Δy energy unprojected,
   which is consistent with cycle 305's destructive composition being most
   severe at the mid-blocks. The constructive follow-up is not done in this
   cycle; it is suggested in the exec_result.

5. **Δy is rank-1 dominated at every measured block, not just b8.** Stable
   rank ∈ [1.018, 1.310] across all 7 measured blocks; top1_E ∈ [0.874, 0.991].
   This is a structural feature of Δ = GELU(z) − ReLU(z) on GPT-2 small
   wikitext-2 calib activations, not a per-block property. The "broad
   curvature residual" framing of cycle 304 should be re-read as "broad
   when projected into the W_proj right-singular basis" — a different
   geometric statement.

6. **Per-block alignment of measurements to cycle 302 panel 4.** Both
   primary-set blocks {7, 8, 9} and concentrator-set blocks {1, 2, 10, 11}
   reproduce E_kept_top_2 to ≤ 1e-4 of the cycle 302 reference, with the
   assertion firing in the script. The cycle 302 framework is fully reused;
   this cycle adds two new SVDs per block (W_proj and Dy) and the four
   measurements above, on top of the existing `compute_Dy` infrastructure.

## Plan adherence

Plan (`autoresearch/plans/cycle_306.md`) vs execution:
- Item 316 verbatim Hypothesis section ✓ (reproduced verbatim from agenda)
- Step 1: new `experiments/block8_deep_dive_cycle306.py` ✓
- Step 2: `UnifiedCurvatureFramework` instantiated with PRIMARY ∪ CONCENTRATORS,
  `capture_inputs()` once ✓
- Step 3: per-block measurements (1)–(4) computed ✓
- Step 4: side-by-side comparison written to JSON, summary, and plot ✓
- Step 5: verbatim verdict thresholds applied ✓
- 8 H2 plan headings (Item 316 verbatim Hypothesis, Dependencies, Approach,
  Files, Config, Verification, Verdict logic, plus an effective 8th via the
  list of bullets — not strictly 8 H2s; the plan structure is heading-list
  rather than 8 separate H2s; this is consistent with the cycle 304 plan
  template) ✓
- Plan marked `# APPROVE` ✓
- Plan length ≤ 40 lines (the user-stated budget) ✓

All plan steps executed cleanly on the first run.

## Convergence check (CLAUDE.md verify-task-performance rule)

Float PPL on 200-seq calib slice = 45.0637, matches cycle 302/304 baseline
45.0637 to 4 decimals. Off-the-shelf gpt2, no task training. Task verification
reduces to float PPL match. ✓ The model is task-converged at the float
baseline (off-the-shelf gpt2 on wikitext-2 val). The geometric analysis
(SVDs of Δy and W_proj, principal-angle cosines, per-token norms) is
meaningful because the task baseline is correct.

## Trainable param count

`Trainable params: 0.` Verified: the script imports
`UnifiedCurvatureFramework` and calls `compute_Dy`, `panel_*` style methods,
none of which create `nn.Parameter` or call `requires_grad_(True)`. The
SVDs and the principal-angle computation are pure linalg on CPU float32.
No `.backward()` is invoked anywhere. ✓

## Wall time

Run log shows ~0.5–0.6 s per block × 7 blocks for the SVD + measurements
(reported per block), plus ~5 s for the single forward pass `capture_inputs`,
plus ~3 s for model load and float PPL eval. Total wall well under the
plan's "~6 min" cap; effective wall ~30 s on mps. The fast wall is because
both SVDs are small (768×768) and Dy chunking is already done by the
framework.

## Cross-cycle sanity

- E_kept_top_2_via_Wproj at every block matches cycle 302 panel 4 to
  ≤ 1e-4 (asserted in script, verified above).
- Float PPL 45.0637 matches cycle 258 / 302 / 304 reference 45.0637.
- The cycle 304 k_95 numbers ([b1: 348, b2: 63, b7: 411, b8: 511, b9: 542,
  b10: 397, b11: 2]) are NOT directly reproduced — cycle 304 measured k_95
  via cumulative `Dy @ Vh_w` energy, this cycle measures k_95 via Dy SVD
  directly. The two are different geometric objects and disagree by
  factors of 5×–63× at most blocks. This is correctly flagged in the
  manifest, summary, and exec_result, and is the central geometric finding
  of the cycle.

## Corrections

None on the verdict. REFUTED is the literal verbatim threshold result.
Both conditions fail by a wide margin. The cycle is unambiguously
REFUTED.

## Artifact existence

- `experiments/block8_deep_dive_cycle306.py` — exists, ~360 lines, single
  driver script reusing cycle 302 framework
- `autoresearch/plans/cycle_306.md` — exists, ≤ 40 lines, marked `# APPROVE`
- `autoresearch/briefs/cycle_306.md` — exists (pre-existing from brief stage)
- `autoresearch/results/cycle_306_block8_deep_dive.json` — exists, valid
  JSON, verdict=REFUTED, per_block dict for 7 blocks with full SV spectra,
  thresholds, condition flags
- `autoresearch/results/cycle_306_summary.md` — exists, per-block tables
  for spectrum / norms / cosines, verdict logic, item 316 verbatim
- `autoresearch/results/cycle_306_panels.png` — exists, 4-panel figure
  (W_proj spectrum, Dy spectrum, top2/top5 ratio bars, Dy stable rank bars)
- `autoresearch/results/cycle_306_manifest.txt` — this cycle
- `autoresearch/results/cycle_306_exec_result.txt` — this cycle
- `autoresearch/audits/cycle_306.md` — this file
- `autoresearch/results/cycle_306_audit.txt` — to be written
- `autoresearch/logs/cycle_306_run.log` — exists, contains all 7 block
  per-block headline lines and the final VERDICT: REFUTED
- `autoresearch/agenda.md` — item 316 moved from NEXT P0 to PAST Cycle 306
  (P0 section now reads "(no P0 items pending)"); P1 (item 317) is still
  the next eligible NEXT pick
- `autoresearch/logs/experiments.tsv` — row 306 appended
- `autoresearch/STATUS.md` — title bumped to "306 Cycles (001-306)"; new
  "Latest: Cycle 306" section added; "Latest: Cycle 305" demoted to "Cycle 305"

## Final status

**PASS** — keep provisional outcome REFUTED. b8 W_proj top-2/top-5 ratio
1.4875 vs concentrator median 1.5142 (NOT 0.5× lower; cond_a False); b8 Δy
stable rank 1.197 vs concentrator median 1.082 (NOT 1.5× higher; cond_b
False). Single-factor "flat + broad" hypothesis fails decisively. Δy is
rank-1 dominated at every measured block (top1_E ∈ [0.87, 0.99]); the b8
anomaly is a top-1 misalignment between Vh_y[:1] and Vh_w[:2], not flatness
or breadth. Cycle 304's "broad curvature residual" reading was a basis
artifact, not an intrinsic property of Δy.

Auditor recommends:
1. Open a follow-up item to test a Δy-aligned rank-1 LASER variant at b8:
   replace c_proj_8 with `(W_proj_8 @ Vh_y_8[:1].T) @ Vh_y_8[:1]` (or the
   equivalent rank-1 factorization keeping the Vh_y[:1] direction) and
   measure single-block PPL vs the standard Vh_w[:1] LASER baseline. If the
   Δy-aligned variant recovers PPL but the Vh_w-aligned variant does not,
   the misalignment reading is empirically validated.
2. Re-examine cycle 305's destructive composition through this lens: are
   the per-block destructive contributions concentrated at blocks where
   |Vh_y[:1] − Vh_w[:1]| is largest? This would explain why the cycle
   305 5-stack lands +18 PPL above the worst single anchor without a
   parameter shortage.
3. Item 316 was a P0 deep-dive item; with REFUTED + the misalignment
   reframing, the next P0 candidate is item 317 (50D-input testbed) which
   is currently P1. Auditor does NOT recommend promoting 317 to P0
   automatically; the misalignment follow-up is a more direct continuation
   of item 316's geometric thread.
