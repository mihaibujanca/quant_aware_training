status=CORRECTED
old_outcome=confirmed
new_outcome=confirmed_with_caveat
reason=Recovery formula is correct (91.4%), and the result exceeds the 20% CONFIRMED threshold. However, the outcome text claims "corrections are remarkably effective" which is misleading. The corrected PPL of 143.6 is a +149% degradation from baseline (57.7), which is worse than k=5 uncorrected (122.1, +136%). The high recovery % is an artifact of the 997.7-point denominator at k=8 — the metric compresses huge absolute failures into attractive percentages. Absolute corrected PPL trend (59.1 -> 70.1 -> 143.6 for k=4/5/8) shows corrections become less effective as k increases, not more. The outcome text must reflect this.
guardrail_flags=G3_misleading_metric,G4_counterintuitive_explained
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Rewrite outcome to note absolute PPL trend. Future cascade experiments should report corrected_ppl and relative_degradation_pct alongside recovery_pct. Consider adding a "usability threshold" (e.g. corrected_ppl < 2x baseline) as a secondary criterion.
interesting=true
interesting_reason=The recovery % metric breaks down at extreme degradation — 91% recovery sounds excellent but yields an unusable model (143.6 PPL). This highlights a metric design flaw: % recovery from catastrophic baselines is misleading. The absolute PPL comparison (k=4 corrected=59.1, k=5 corrected=70.1, k=8 corrected=143.6) tells the real story — corrections scale sublinearly with damage.
