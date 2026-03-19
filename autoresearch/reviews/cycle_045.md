# Cycle 045 Design Review — Practical Compression Recipe

## R1: Hypothesis clarity
PASS. Clear deliverable: write recipe, replicate numbers on Shakespeare, reality-check on WikiText-2.

## R2: Measurement soundness
PASS. Replication targets are relative PPL (% increase), consistent with cycles 039/044. 3 seeds, 200 eval seqs -- same protocol as prior cycles.

## R3: Capacity / baseline adequacy
PASS. Same GPT-2 model, same compression pipeline as cycles 037-044. No new methods introduced.

## R4: Pitfall check
- **MP1-MP3**: N/A -- no new thresholds, no new fitting, no new detection.
- **MP4 (domain mismatch)**: ACKNOWLEDGED. The plan explicitly includes WikiText-2 as Condition B and documents the corpus-specificity caveat prominently. This is not a pitfall -- it IS the finding.
- **MP5**: N/A.
- **MP6 (verdict misclassification)**: RISK. The inconclusive-if criterion ("requires corpus-specific tuning not captured in the documentation") is likely to trigger. The plan correctly anticipates this and states INCONCLUSIVE is the expected outcome. The recipe document MUST include the corpus-specificity caveat as a first-class limitation, not a footnote. If the executor writes a recipe without the caveat, the cycle would be CONFIRMED on a technicality while hiding a fundamental limitation. **Mitigation**: plan Step 1 lists the caveat as a required prominent section.

## R5: Confounds
- WikiText-2 numbers from cycle 044 used corpus-specific probe+eval. The plan replicates this exactly. PASS.
- Shakespeare numbers from cycle 039 used 6x. Plan uses 6x. PASS.

## R6: Data availability
PASS. Shakespeare at `data/shakespeare.txt`, WikiText-2 cached from cycle 044.

## R7: Agenda criteria match
PASS. Plan's verdict criteria match item 63 exactly: 1pp tolerance for confirmed, 3pp for refuted, corpus-specific caveat for inconclusive.

## Verdict: APPROVED

One emphasis: the recipe document must lead with the corpus-specificity limitation, not bury it. The method works for narrow-distribution text; it does not work for general text. This is the honest finding from 44 cycles of work.
