status=CORRECTED
original_outcome=CONFIRMED
audited_outcome=INCONCLUSIVE
guardrail_flags=G1,G3,G4
interesting=true

## Phase 1: Verdict Check

The execution reports CONFIRMED because condition C (mixed 2048 tokens) reduces WikiText-2 PPL degradation from ~5.2M% (condition A) to ~61%. This exceeds the 10pp threshold by many orders of magnitude.

However, **the comparison is fatally confounded by lstsq sample size, not corpus diversity.**

### The lstsq degeneracy problem

At 6x compression, k = 3072/6 = 512 selected channels. With bias, the lstsq design matrix has 513 columns. Condition A uses 512 tokens — producing a [512, 513] design matrix with **more unknowns than equations**. This is an underdetermined system. The lstsq minimum-norm solution has no generalization guarantee; catastrophic PPL (3.7M) is the expected outcome for a degenerate lstsq, not evidence about corpus effects.

Condition C uses 2048 tokens — a well-conditioned [2048, 513] design matrix (4:1 overdetermination ratio). Condition B uses 512 tokens — also degenerate, but slightly less catastrophic because the mixed distribution covers more of the input space.

### Comparison with cycle 044

Cycle 044 tested the closest analogue: Shakespeare channels → WikiText-2 eval (condition C there). It used `max_tokens_per_layer=2048`, giving a well-conditioned [2048, 513] design matrix even for Shakespeare-only refit. Result: PPL ~135 (uncorrected), +94% degradation — **not millions percent**. This proves that Shakespeare-only refit with sufficient samples is not catastrophic; the PPL 3.7M in cycle 076 is a sample-size artifact.

### What cycle 076 actually tests

The experiment conflates two variables:
1. Corpus diversity (Shakespeare vs mixed) — the intended variable
2. Lstsq sample count (512 vs 2048) — the confound

Conditions A and B both have 512 samples (degenerate). Condition C has 2048 samples (well-conditioned). The entire A→C improvement could be explained by sample size alone. To isolate corpus diversity, you'd need:
- A': Shakespeare-only, 2048 tokens
- B': Mixed, 2048 tokens (512 × 4 corpora)
- C: Mixed, 2048 tokens (as is)

Without A', the CONFIRMED verdict is unsupported.

**Audited verdict: INCONCLUSIVE.** The criterion "mixed refit reduces degradation by ≥ 10pp relative vs Shakespeare-only" cannot be evaluated because the Shakespeare-only baseline is broken by lstsq degeneracy, not by corpus mismatch.

## G1: Implementation Correctness — FAIL

- **Channel selection:** Fixed from Shakespeare across all conditions. CORRECT.
- **Lstsq refit data:** Each condition uses its specified corpus. CORRECT.
- **Eval set:** All conditions evaluated on the same WikiText-2 200-sequence eval set. CORRECT.
- **Block swap:** Multi-block (4 blocks: [7,5,4,1]) swapped simultaneously. CORRECT and matches cycle 044.
- **Design matrix degeneracy:** Condition A has n=512 samples for p=513 parameters (512 channels + bias). **This is the critical bug.** The lstsq is underdetermined. The code does not check or warn about this. Condition B has the same issue. Only condition C (n=2048) is well-conditioned.

## G2: Seeds — PASS

Three seeds (42, 123, 7). Results consistent within conditions. Condition C is remarkably stable: 112.2–113.7 PPL (std=0.74). The catastrophic conditions (A, B) show higher variance as expected from degenerate lstsq.

## G3: Results Interpretation — FAIL

The agenda entry states: "The key insight is that refit quality is dominated by activation distribution coverage, not channel selection." This is **not supported by the evidence**. The actual key variable is lstsq sample count relative to parameter count.

Cycle 044 proved that 2048 Shakespeare tokens → WikiText-2 eval gives PPL ~135 (+94% degradation). If cycle 076 had included a 2048-token Shakespeare-only condition, it would likely have shown similar moderate degradation (~100-150% rather than 5M%), revealing that the A→C jump is mostly sample size, with only a modest corpus-diversity component.

The PPL 3.7M for condition A is not "catastrophic corpus mismatch" — it is lstsq numerical failure from an underdetermined system.

## G4: Counterintuitive Results — FLAG

PPL of 3.7M is extraordinary and demands explanation beyond "corpus mismatch." The audit identifies lstsq degeneracy as the root cause. The execution did not investigate this, violating the counterintuitive-results protocol.

## G5: Learning Quality

N/A — no training involved, lstsq-only.

## G6: Method Failure

The lstsq degeneracy in conditions A and B constitutes a method failure. The experiment should have ensured n >> p for all conditions to isolate the intended variable (corpus diversity).

## Recommendations

1. **Downgrade to INCONCLUSIVE.** The confound makes the verdict unsupported.
2. **Rerun with matched sample sizes.** Add condition A': Shakespeare 2048 tokens (16 holdout sequences). This isolates corpus diversity from sample size.
3. **Add an n>>p guard** to `_fit_output_projection` or the experiment template: warn if n < 2*p.
4. **Update item 66** to note the confound and mark for rerun.
