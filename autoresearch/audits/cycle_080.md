status=FAIL
original_outcome=INCONCLUSIVE
audited_outcome=INCONCLUSIVE
guardrail_flags=G1,G3,G4,G6
interesting=true

## Phase 1: Verdict Check

Best analytic delta is +10.6% (1-pass mean). The INCONCLUSIVE threshold is 10–25%. Numerically, +10.6% is barely inside the inconclusive range. **Verdict is technically correct** but misleading — the analytic correction is essentially a no-op (+10.6% vs unpolished +10.9%, delta of 0.3pp which is within std of 0.6pp).

## Phase 2: Quality Checks

### G1: Implementation — FLAG

**The correction formula is mathematically vacuous.** The code does:

1. `e_b = y_comp - y_float` (compression error)
2. `e_retained = e_b @ V_retained @ V_retained.T` (project onto 99%-variance output subspace)
3. `y_corrected = y_float - e_retained` (target for lstsq re-solve)
4. Re-solve: `W_new = lstsq(h_comp, y_corrected)`

With 99% energy threshold, mean_energy_frac_retained = 0.940, meaning ~94% of the error lives in the retained subspace. So `y_corrected ≈ y_float - 0.94 * e_b = y_float - 0.94*(y_comp - y_float) = 1.94*y_float - 0.94*y_comp`. This is asking the re-solved lstsq to produce an output that overcorrects past the float target. But the re-solve uses the SAME `h_comp` as the original lstsq — the hidden activations haven't changed. The only way this changes W_down is by shifting the target, but lstsq already minimizes ||h_comp @ W.T - y_float||. Shifting the target to `y_corrected` just produces a slightly different W that's no longer optimal for matching y_float. The result (+10.6% vs +10.9%) confirms: the correction does essentially nothing, and what it does is noise.

**Pass 2-3 divergence is expected under this analysis.** Each pass re-collects h_comp from the model modified by the previous pass, creating a feedback loop with no convergence guarantee. Passes 2-3 at +32.7% and +25.4% confirm instability.

### G2: Seeds — PASS

3 seeds (42, 123, 7). Results are consistent across seeds (std < 1pp for 1-pass).

### G3: Baseline mismatch with cycle 061 — FLAG

**Cycle 061**: Baseline PPL = 57.653 (OWT), Unpolished = +30.2%, Adam 500-step = +4.9%. Corpus: OpenWebText.
**Cycle 080**: Baseline PPL = 125.1 (Shakespeare), Unpolished = +10.9%, Adam 500-step = −37.3%. Corpus: Shakespeare.

The entire discrepancy is explained by **corpus change**: cycle 080 uses `data/shakespeare.txt` while cycle 061 used `data/openwebtext_sample.txt`. GPT-2's baseline PPL on Shakespeare (~125) is much higher than on OWT (~58) because Shakespeare is out-of-distribution for GPT-2. The unpolished degradation drops from +30.2% to +10.9% because the lstsq refit on Shakespeare acts as domain adaptation (as flagged in cycle 077 audit for block 7).

This is not a bug but it means cycle 080 results are NOT comparable to cycle 061. The plan should have specified OWT corpus for comparability. The +10.9% unpolished baseline is artificially low due to the Shakespeare domain adaptation effect.

### G4: Adam sub-baseline PPL — FLAG (CRITICAL)

Adam 500-step gives PPL 78.1 vs baseline 125.1 = −37.3%. A compressed model with 4 blocks at 6× compression achieves **37% LOWER perplexity than the float model**. This is the same lstsq-as-regularizer artifact flagged in cycle 077:

- The float model's Shakespeare PPL (~125) reflects GPT-2's poor fit to Elizabethan English.
- Adam polish on Shakespeare probe_holdout (4096 tokens) fine-tunes W_down to specialize for Shakespeare.
- eval_val is also Shakespeare (from the same data/shakespeare.txt, just a val split).
- Result: Adam is doing domain adaptation, not compression polishing. The sub-baseline PPL is meaningless as a compression metric.

**This renders the entire Adam comparison invalid.** The experiment compares "analytic projection" vs "500-step domain fine-tuning" — these are not comparable interventions.

### G5: Learning quality — N/A for analytic; FLAG for Adam

The analytic correction has no learning. The Adam component trains on 32 holdout sequences (4096 tokens) for 500 steps — that's ~62 epochs through the data at batch_size=4. For just W_down parameters across 4 blocks, this is heavy overtraining on a tiny corpus. The sub-baseline PPL is consistent with memorization/domain specialization.

### G6: Method failure — FLAG

The analytic projection method doesn't work (0.3pp improvement, within noise). The experiment correctly identifies this in the interpretation. However, the explanation ("lstsq re-solve optimizes for local block outputs but ignores downstream interactions") is wrong. The actual reason is simpler: **the correction target is constructed from the same h_comp that lstsq already fit, so re-solving with a slightly shifted target produces a nearly identical W_down.** The method would only help if h_comp could change (e.g., by modifying upstream blocks first), but within a single block the correction is algebraically near-trivial.

## Critical Audit Answers

**Q1: Why is unpolished baseline +10.9% vs cycle 061's +30.2%?**
Different corpus. Cycle 061 used OWT (GPT-2's training distribution, baseline PPL 57.7). Cycle 080 uses Shakespeare (out-of-distribution, baseline PPL 125.1). The lstsq refit on Shakespeare provides domain adaptation that partially compensates for compression loss, yielding a much lower delta. Results are not cross-comparable.

**Q2: Adam −37.3% — same lstsq regularization artifact?**
Yes. Confirmed. Adam is fine-tuning W_down on 4096 Shakespeare tokens for 500 steps, and eval is also Shakespeare. This is domain adaptation masquerading as compression polishing. Eval and polish data are from the same file (shakespeare.txt), just different splits. The sub-baseline PPL does not indicate the compressed model is better — it indicates 500 Adam steps of in-domain fine-tuning overcome GPT-2's poor Shakespeare fit.

**Q3: Analytic 1-pass +10.6% ≈ unpolished +10.9% — why did the correction fail?**
The projection onto the retained SVD basis captures 94% of the error energy, but re-solving lstsq with a shifted target using the same hidden activations h_comp produces a nearly identical W_down. The method is algebraically vacuous for single-block correction: lstsq(h_comp, y_float - P·e) ≈ lstsq(h_comp, y_float) when h_comp is fixed. The 0.3pp difference is noise.

## Verdict Assessment

INCONCLUSIVE is technically correct (+10.6% is in the 10–25% band). But the experiment has fatal design flaws:
1. Shakespeare corpus makes results incomparable to prior work
2. Adam baseline is invalidated by domain adaptation
3. The analytic method is a no-op by construction

The verdict should note these caveats. The method is effectively REFUTED (does nothing), but the numerical criterion maps to INCONCLUSIVE.

**Audit status: FAIL** — G1 (method is algebraically vacuous), G3 (corpus mismatch unremarked), G4 (Adam sub-baseline is domain adaptation artifact), G6 (method failure misdiagnosed).
