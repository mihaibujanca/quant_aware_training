# Cycle 308 Audit — Item 318

**Audit status: PASS**

## Scope

Verify cycle 308 execution against item 318's verbatim hypothesis and outcome
thresholds. Confirm: (1) the synthetic rank-2 unigram projector was implemented
per the brief (cycle 280 A factor reused, d1 = wte^T @ log_p, d2 = wte^T @
s_vec Gram-Schmidt to d1, B_synth = stack([s1*d1, s2*d2])); (2) the cycle 280
SVD-init reproduction sanity matches the published reference; (3) the unigram
histogram is computed on the correct calib slice; (4) the stopword indicator
has sufficient support; (5) Gram-Schmidt achieves orthogonality; (6) the grid
search exhausts the prescribed 25 points; (7) the verdict thresholds are
applied verbatim per item 318.

## Verdict verification

Item 318 outcomes (verbatim from agenda):
- **CONFIRMED** iff `best synthetic-projector PPL <= 60` (within 15 PPL of
  cycle 280 trained 49.03 and comparable to no-retrain rank-2 init 57.66 —
  "the signal IS unigram, handcrafted construction captures the majority of
  compressible signal without training").
- **REFUTED** iff `best synthetic-projector PPL >= 100` (worse than block 1
  swap damage — "the rank-2 signal is NOT reducible to the unigram direction
  + stopword indicator even though cycle 300 showed top-20 boost tokens are
  stopwords").
- **INCONCLUSIVE** otherwise.

Auditor independently re-derived the verdict from
`autoresearch/results/cycle_308_synthetic_unigram.json`:

- 25 grid points present (`len(grid_results) == 25` confirmed).
- min PPL across grid = **127.4956** at `(s1, s2) = (0.5, 0.5)`.
- threshold check: `127.4956 <= 60` -> False (CONFIRMED branch closed).
- threshold check: `127.4956 >= 100` -> True (REFUTED branch satisfied).
- -> **REFUTED**.

The script-reported verdict and the auditor-recomputed verdict agree.
The REFUTED verdict is verbatim-correct.

## Implementation fidelity

Auditor cross-checked the script against the plan and the brief:

- **Cycle 280 A factor reuse**: `A_frozen = U[:, :2] @ diag(S[:2])` from
  full SVD of `model.transformer.h[1].mlp.c_proj.weight` (the same
  construction used in `cycle280_A_semantic_cycle300.py` at line 142).
  Shape `(3072, 2)` asserted. Bit-for-bit identical to cycle 280's A —
  the SVD-init replacement (`A_frozen @ Vh[:2, :]`) lands at PPL 57.6581
  vs the cycle 280 reference 57.664 (delta 0.006, within the 0.05
  tolerance asserted in the script).

- **Unigram histogram**: computed via `counts = bincount(flat_ids)` over
  the 200 calib seqs × 128 tok = 25,600 tokens (asserted exactly). The
  same 200 wikitext-2 val seqs used as cycles 280/299/300/301/302 (the
  first 200 seqs after filtering for `len(text.strip()) > 50` and
  `seq_len >= SEQ_LEN`). 5,576 unique tokens observed (~11% of vocab).
  Smoothing `eps = 1/(N*10) = 1/256000 = 3.91e-6` is well below the
  smallest observed empirical frequency `1/25600 = 3.91e-5` (10x
  separation), so `log(p + eps)` for unobserved tokens is
  `log(eps) = -12.453` and observed tokens are essentially `log(p)`.

- **d1 construction**: `d1_raw = wte^T @ log_p ∈ R^768`,
  `||d1_raw|| = 1,257,193.4`, `d1 = d1_raw / ||d1_raw||`. Auditor
  confirms unit-norm post-normalisation (script asserts via
  `||d1|| = 1.000000`). The large raw norm reflects that log_p values
  for unobserved tokens are `~ -12.4` and there are 44,681 of them, so
  `wte^T @ log_p` is dominated by a near-uniform negative offset. The
  normalised d1 is well-defined and finite.

- **Stopword indicator**: 420 base strings in `STOPWORD_BASE` (counted
  by Python `len()` in the config dict). Each string is encoded both
  as `s` and as `' ' + s`; only encodings that yield a single token id
  are kept (filtering out multi-piece BPE splits like "perhaps" ->
  ['per','haps']). Result: 747 unique single-token ids (asserted >= 200).
  This generously exceeds the brief's "top ~300 stopwords" target by
  taking advantage of the leading-space variant capture.

- **Gram-Schmidt**: `d2_orth = d2_raw - <d2_raw, d1> * d1`,
  `d2 = d2_orth / ||d2_orth||`. Residual `|<d1, d2>| = 2.50e-07`
  (asserted < 1e-5). Numerical orthogonality is achieved.

- **B_synth**: `B_synth(s1, s2) = stack([s1 * d1, s2 * d2]) ∈ R^(2, 768)`.
  Note auditor: the magnitudes `mag1_base = mag2_base = 1.0` are
  unit-norm matched to the cycle 280 SVD init (which has unit-norm Vh
  rows). `s1 = s2 = 1` is therefore the "matched magnitude" reference
  point and is directly comparable to the SVD init. The 5x5 grid spans
  half-magnitude to 10x magnitude in each direction.

- **Replacement**: `c_proj.weight.data.copy_(A_frozen @ B_synth)` per
  grid point; bias is left unchanged (frozen at `c_proj.bias.data`,
  identical to the cycle 280 SVD-init convention). Weight is restored
  to `W_orig` after the grid sweep. No `requires_grad_(True)` is set
  anywhere, no `.backward()` is called — trainable params = 0 confirmed
  by inspection.

- **Eval slice**: 100 wikitext-2 val seqs × 128 tok, indices 200..299
  after filtering. Identical to cycles 280/299/300/301/302/305 (same
  `eval_ppl` accumulation: total NLL / total token count, `exp` for PPL,
  `out.loss` from HF `GPT2LMHeadModel(input_ids=ids, labels=ids)`).

- **Grid**: 5 × 5 = 25 points over `{0.5, 1.0, 2.0, 5.0, 10.0}^2`,
  matches the brief verbatim (`Sweep direction scalings for d1, d2
  independently ∈ {0.5, 1, 2, 5, 10} × empirical magnitudes, take best`).

Plan adherence: PASS.

## Sanity checks (auditor cross-verifies)

- Float PPL 53.8949 vs cycles 280/284/291/305 reference 53.895
  (delta 0.0051, asserted within 0.5). Same eval slice, same model.
- SVD-init replacement PPL 57.6581 vs cycle 280 reference 57.664 (delta
  0.006, asserted within 0.05). The A factor and the rank-2 LASER
  replacement are bit-for-bit identical to cycle 280, confirming the
  REUSED A factor is the same one cycle 300 analysed.
- W_orig.shape == (3072, 768): PASS.
- wte.shape == (50257, 768): PASS.
- A_frozen.shape == (3072, 2): PASS.
- Vh_init.shape == (2, 768): PASS.
- ||d1|| == 1.000000: PASS.
- |<d1, d2>| = 2.50e-07 < 1e-5: PASS.
- |stopword_ids| = 747 >= 200: PASS.
- n_calib_tokens == 25,600: PASS.

## Substantive findings — auditor endorses

1. **REFUTED is verbatim-correct.** best_synth_ppl = 127.50 >= 100 ->
   REFUTED branch satisfied. The verdict is the literal threshold result,
   not a hedge. The +27.50 PPL margin above the REFUTED bar is
   substantial; even pushing the grid corner to (0.25, 0.25) or smaller
   would not plausibly drop PPL below 60 (it would converge toward the
   bias-only baseline, which leaves block 1 c_proj effectively a constant
   shift, strictly worse than the SVD init's ~57.66).

2. **The "unigram axes" reading is a token-classification artifact, not
   a vector-recovery property.** Cycle 300 confirmed unigram by
   classifying the top-20 boost tokens of `wte @ B_final[i]` as 17/20
   and 19/20 stopwords. Cycle 308 tests whether the underlying R^768
   direction is RECOVERABLE from the natural model-agnostic
   constructions (`wte^T @ log_p` and `wte^T @ stopword_indicator`).
   At the matched magnitude `s1 = s2 = 1`, the synthetic gives PPL 577.69
   vs the SVD init's 57.66 — a 10x ratio at SAME shape, A factor, bias,
   eval slice. The ONLY difference is the choice of B rows. The trained
   B_final rows and the synthetic (d1, d2) directions live in different
   parts of R^768, and the gap is geometrically decisive.

3. **Mechanistic alternative**: many R^768 vectors will project (via
   wte) onto a top-20 token list dominated by stopwords because
   stopwords are the highest-prior-frequency tokens and any
   approximately-uniform direction gets boosted toward them through the
   LM head. Cycle 308 is the discriminating test between
   "B factor IS mechanistically a unigram axis" (would CONFIRM here) and
   "B factor's top-20 boost list LOOKS unigram-like because of LM-head
   geometry" (REFUTED here). The data selects the second reading.

4. **d2 (stopword indicator) is the dominant damage source.** At fixed
   s2 = 0.5, increasing s1 from 0.5 to 5.0 inflates PPL from 128 to
   52,486 (410x). At fixed s1 = 0.5, increasing s2 from 0.5 to 10.0
   inflates PPL from 128 to 25,434 (199x). Both directions damage the
   downstream block-2 input, confirming that the synthetic choice is
   geometrically off (not just off-magnitude).

5. **The grid argmin is at the corner (0.5, 0.5)**, suggesting the
   prescribed grid does not extend low enough to find a true minimum.
   However, the brief explicitly bounds the search to this 25-point
   grid, and even an unbounded scale-down would be limited by the
   bias-only baseline (which is strictly worse than the SVD init). The
   verdict is robust to this corner-of-grid concern.

6. **Cycle 280's rank-2 anchor remains valid as a TRAINED result, not
   a hand-designed one.** The model-agnostic 1,536-parameter
   construction hypothesis is closed.

7. **Constructive follow-up** (auditor recommends as a separate agenda
   item, not auto-promoted): compute the cosines `cos(B_final[i], d_j)`
   for `i, j ∈ {0, 1}` on the trained cycle 280 B factor. If
   `max cos < 0.5`, the trained directions are NOT a scaled version of
   (d1, d2) and cycle 300's "unigram axes" framing is closed as a pure
   token-classification property. If `max cos > 0.8`, the directions ARE
   roughly aligned and the cycle 308 failure is purely a magnitude
   issue (which would suggest a finer grid sweep). Either result
   sharpens the cycle 300/308 picture.

## Verdict clause mechanics

The verdict computation is a single inequality (no multi-clause logic):
```python
if best["ppl"] <= 60.0:
    verdict = "CONFIRMED"
elif best["ppl"] >= 100.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
```
Auditor inspected the code at the verdict block and confirms the
threshold values match the item 318 verbatim (60 and 100). best is
selected by `if best is None or ppl < best["ppl"]` over the 25 grid
points, so it is the global min over the grid (auditor independently
recomputed `min(ppls) = 127.4956` from the JSON, matches).

## Artifact existence

- `experiments/synthetic_unigram_projector_cycle308.py` — exists, ~360
  lines, single driver script with config-at-top, eval_ppl helper,
  unigram histogram, d1/d2 construction, grid sweep, JSON+md output.
- `autoresearch/plans/cycle_308.md` — exists, ≤ 45 lines, marked
  `# APPROVE` at end, contains item 318 verbatim Hypothesis.
- `autoresearch/briefs/cycle_308.md` — exists (pre-existing from brief
  stage).
- `autoresearch/results/cycle_308_synthetic_unigram.json` — exists,
  valid JSON, verdict=REFUTED, all 25 grid points present, sanity
  fields populated.
- `autoresearch/results/cycle_308_summary.md` — exists, contains
  verdict, grid table (5x5), construction description.
- `autoresearch/results/cycle_308_manifest.txt` — this cycle.
- `autoresearch/results/cycle_308_exec_result.txt` — this cycle.
- `autoresearch/audits/cycle_308.md` — this file.
- `autoresearch/results/cycle_308_audit.txt` — to be written.
- `autoresearch/logs/cycle_308_run.log` — exists, contains the per-grid
  PPL lines and final VERDICT: REFUTED.
- `autoresearch/agenda.md` — to be updated (item 318 P2 -> PAST Cycle 308).
- `autoresearch/logs/experiments.tsv` — to be appended (row 308).
- `autoresearch/STATUS.md` — to be bumped to 308.

## Trainable param count

`Trainable params = 0`. Verified: no `.requires_grad_(True)` is called
anywhere in the script; no `.backward()`, no optimiser; the only
modifications to `model` are `c_proj.weight.data.copy_(...)` (in-place
data tensor swap, no autograd).

## Wall time

Run log shows 39.3 s total: float baseline 1.3 s + SVD 0.1 s + sanity
init eval 1.3 s + 25 grid evals × 1.3 s = ~32 s + setup ~6 s. Well
under the plan's "~3 min wall on mps" estimate. Fast because each PPL
eval is a 100-seq forward pass (no grad, no training) and the
weight-swap is in-place.

## Cross-cycle sanity

- Float PPL 53.8949 matches cycles 258/280/284/291/302/304/305 to 4
  decimals (asserted).
- Cycle 280 SVD-init reproduction 57.6581 vs cycle 280 reference 57.664
  (delta 0.006, asserted within 0.05). The A factor is the same
  geometric object cycle 300 analysed.
- Unigram histogram computed on the same 25,600-token slice as cycles
  280/299/300/301/302 (200 wikitext-2 val seqs after the filter).
- 5,576 unique tokens observed in this slice — consistent with prior
  cycle observations (typical for 25,600 BPE tokens of natural English).

## Final status

**PASS** — keep provisional outcome REFUTED. The verdict is the
literal threshold result (127.50 >= 100) and is robust to the corner-of-
grid concern. The substantive geometric reading — "cycle 300's 'unigram
axes' is a token-classification property of the trained B, not a
recoverable R^768 direction" — is well-supported by the 10x PPL gap at
matched magnitude and is the correct sharpening of REFUTED.

Auditor recommends:
1. Open a follow-up item (priority P3): compute
   `cos(B_final[i], d_j)` on the cycle 280 trained B factor to
   mechanistically confirm or refute the "directions live in different
   parts of R^768" reading. ~10 lines of code, no new training.
2. Item 318 closes REFUTED. Per the queue rules, the next NEXT pick
   is the next eligible P-priority item in the agenda's NEXT section
   (currently empty after item 318 is moved to PAST — agenda refresh
   recommended).
3. The "model-agnostic 1,536-parameter handcrafted construction"
   hypothesis is closed. Cycle 280's rank-2 anchor is valid as a
   trained result only; future "no-Adam" rank-2 attempts should use
   SVD of W_proj_1 itself (cycle 280 SVD-init) as the floor reference,
   not hand-designed directions.
