# Known Measurement Pitfalls

Each entry describes a class of experiment design error. The design review
agent checks whether any apply to the current plan. Not all are relevant
to every experiment — use judgment about applicability.

## MP1: Threshold above trivial baseline
When a detection threshold exceeds what a uniform/null distribution produces,
nearly everything gets flagged regardless of signal.
Diagnostic: "If all units contributed equally, would they exceed this threshold?"

## MP2: Detection method mismatched to activation regime
Methods calibrated for one activation output range may be meaningless under
a different activation that never reaches that range.
Diagnostic: "Can the activation function actually produce values in the
detection region?"

## MP3: Underdetermined fitting
Fitting N_params from N_samples with N_samples < N_params interpolates
training data but generalizes poorly. Held-out metrics become unreliable.
Diagnostic: "Is n_samples >= 2x n_params (including bias terms)?"

## MP4: Phenomenon-domain mismatch
Some phenomena are specific to certain task structures. Transferring a
hypothesis across domains requires explicit justification.
Diagnostic: "Has this phenomenon been demonstrated in a domain with similar
structure to the one being tested?"

## MP5: Insufficient model capacity for comparison
When comparing conditions, both must have enough capacity to succeed. If both
fail, the comparison is between two failure modes, not two approaches.
Diagnostic: "Can the model succeed at this task at all, independent of the
experimental variable?"

## MP6: Verdict misclassification — "refuted" when method failed
When a metric is uninformative by construction (invariant, saturated, or
operating outside its valid range), a negative result says nothing about
the hypothesis. Recording REFUTED pollutes the PAST EXPERIMENTS record
that brief/plan agents read for context, causing downstream agents to
treat the hypothesis as closed when it was never tested.
Diagnostic: "Did the metric actually vary in a range that could
distinguish confirmation from refutation? Or was it stuck at a trivial
value regardless of the hypothesis being true or false?"

---
Historical references (maintainer context only — not for agent consumption):
MP1 ← cycle 032, MP2 ← cycle 028, MP3 ← cycle 011, MP4 ← cycle 021, MP5 ← cycle 031
MP6 ← cycles 021, 031 (grokking experiments recorded REFUTED on uninformative metrics)
