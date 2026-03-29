# Audit — Cycle 111 (Item 122)

**Item:** Depth gradient of GELU operating regime — GPT-2 FFN [INTERPRETABILITY]
**Verdict:** REFUTED
**Audit status:** PASS — verdict correct, one flag raised

---

## G1: Computation check

Script: `experiments/gelu_depth_gradient_gpt2.py`

- Hooks capture `c_fc` output (pre-GELU activations) per block — correct target.
- neg_frac = `(z < 0).float().mean()` over all (token, channel) entries — correct.
- GELU derivative formula: `cdf + z * pdf` — matches the exact derivative of GELU(z) = z * Phi(z).
- OLS via `scipy.stats.linregress` on block index vs neg_frac — standard.
- Monotonicity violation count: `neg_frac[i+1] < neg_frac[i]` — correct (checks increasing, matching the hypothesis of deeper = more negative).
- Attention effect: compares `ln2_out @ W + b` (actual) vs `LN2(ln1_in) @ W + b` (hypothetical no-attention). This is a reasonable counterfactual but approximate — re-applying LN2 params to a different input shifts the normalization. Acceptable for a first-order estimate.
- Chunked forward pass with BLOCK_SIZE=1024, 4096 tokens total — correct for GPT-2 context window.
- PPL sanity check (17.37) within expected range for GPT-2 on OWT.

**Verdict: computation sound.** No bugs found.

## G2: Determinism

- Forward pass with `torch.no_grad()` on a fixed text file — deterministic.
- No sampling, no dropout (model.eval()), no random seeds affecting the measurement.
- Skewness uses `np.random.default_rng(0)` for subsampling — seeded.

**Verdict: deterministic.**

## G3: Numbers match criteria

| Metric | Value | Threshold | |
|--------|-------|-----------|---|
| Violations | 6/11 | >=4 → REFUTED | REFUTED |
| Slope | +0.304%/block | <0.2% → REFUTED | Above 0.2% but below 0.5% (confirm threshold) |
| R² | 0.083 | >0.7 → CONFIRMED | Far below |
| p-value | 0.364 | — | Not significant |

REFUTED by the violations criterion (6 >= 4). Slope is 0.304% — above the refute threshold of 0.2% but below the confirm threshold of 0.5%. The refutation is driven entirely by the non-monotonicity, which is overwhelming (6/11 transitions violate).

Independently verified: violation list is blocks 0→1 (−14.2%), 3→4 (−2.6%), 4→5 (−0.1%), 5→6 (−1.1%), 9→10 (−0.8%), 10→11 (−1.9%). OLS slope=0.00304, R²=0.0828, intercept=0.824 — all match JSON to 5 significant figures.

**Verdict: REFUTED is correct.**

## G5: What did we learn?

1. **GELU regime is depth-invariant in GPT-2.** Overall neg_frac is stable at 73–87% across all 12 blocks. The model operates deep in the gating regime at all depths — GELU is ~85% gate, ~15% pass-through everywhere.
2. **Block 1 is anomalous** (neg_frac=0.728, E[GELU']=0.238 — most linear block). This might reflect block 1's known role as a "positional/shallow feature" layer. All other blocks cluster 0.83–0.87.
3. **Attention has negligible effect** on the GELU operating regime (delta < 4.3% in all blocks, mostly < 1.5%).
4. **For compression**: depth-uniform gating means compression strategies don't need to be depth-adapted based on GELU regime. A single strategy works for all layers.

## G6: Relationship to cycle 103

**FLAG: Quantity mismatch between cycle 103 observation and cycle 111 test.**

Cycle 103 (item 108, GELU/ReLU anatomy) measured neg_frac **within the bump zone** (|z| < 0.5), finding 54% → 64% drift from block 0 to block 11. Item 122 was motivated by this observation.

Cycle 111 measured **overall** neg_frac (fraction of all pre-activations z < 0), finding no monotonic trend (73–87%, 6/11 violations).

These are different quantities:
- Bump-zone neg_frac: among activations near zero, do more fall slightly negative?
- Overall neg_frac: what fraction of all activations are negative?

The cycle 103 drift (54% → 64% in the bump zone) could still be real while the overall neg_frac shows no depth gradient — if the bump zone is a small fraction of activations and its internal composition shifts while the overall distribution stays stable.

**The refutation is valid for the specific claim tested** (overall neg_frac monotonicity), but it does not directly contradict cycle 103's bump-zone observation. The agenda item text ("negative fraction of bump-zone activations increases from 54% to 64%") was tested with overall neg_frac instead of bump-zone neg_frac. This is a scope expansion that tests a stronger claim than what cycle 103 suggested.

---

## Summary

| Check | Result |
|-------|--------|
| G1 Computation | PASS |
| G2 Determinism | PASS |
| G3 Criteria match | PASS — REFUTED correct |
| G5 Learning | Depth-invariant GELU regime; block 1 anomaly |
| G6 Provenance | FLAG — different quantity than cycle 103's 54%→64% |
