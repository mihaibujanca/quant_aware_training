# Cycle 044 Design Review — Domain Robustness

## R1: Hypothesis clarity
PASS. Clear question: does Shakespeare-fit channel selection + correction transfer to WikiText-2?

## R2: Measurement soundness
PASS. All comparisons are relative PPL (% increase over corpus-specific baseline). Shakespeare and WikiText-2 have different absolute PPL, so relative comparison is correct. The plan explicitly notes this.

## R3: Capacity / baseline adequacy
PASS. Same GPT-2 model, same compression config (k=4@6x) as cycle 039. Baseline replication (Condition A) serves as sanity check.

## R4: Pitfall check
- **MP1 (threshold above baseline)**: N/A — no thresholding.
- **MP2 (activation mismatch)**: N/A — same model.
- **MP3 (underdetermined fitting)**: PASS. lstsq system is ~25K tokens x 769 -> 768, well-determined (~32:1).
- **MP4 (domain mismatch)**: This IS the experiment — testing domain transfer. The plan correctly includes both corpus-specific and zero-shot conditions to distinguish "method broken on new domain" from "method doesn't transfer."
- **MP5 (insufficient capacity)**: N/A — same model.
- **MP6 (verdict misclassification)**: PASS. Criteria distinguish confirmed/refuted/inconclusive with concrete thresholds from the agenda.

## R5: Confounds
- WikiText-2 and Shakespeare have different token distributions. GPT-2 baseline PPL may differ substantially. The plan handles this via relative comparison. PASS.
- Condition D (re-fit correction only) is a good diagnostic: isolates channel selection vs correction as the transfer bottleneck. PASS.

## R6: Data availability
MINOR CONCERN. WikiText-2 download path needs verification. The plan mentions both URL download and pyarrow extraction without committing to one. The script must handle download failure gracefully (e.g., fallback or clear error). This is an implementation detail, not a design flaw.

## R7: Agenda criteria match
PASS. Plan tests "zero-shot transfer PPL cost within 3pp" (confirmed) and ">20% increase" (refuted), matching item 62 exactly. OpenWebText dropped — agenda says "2 of 3 corpora" suffices for non-inconclusive verdict.

## Verdict: APPROVED

No blocking issues. One implementation note: pin the WikiText-2 download to a specific URL and verify it produces parseable text before running compression.
