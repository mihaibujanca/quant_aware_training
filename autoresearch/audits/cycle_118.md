status=PASS
old_outcome=INCONCLUSIVE
new_outcome=INCONCLUSIVE
reason=Verdict is correct. Model reached PPL=78.65 (>50 threshold) after 20M tokens, failing the training quality gate. All other CONFIRMED criteria are met: monotonic pruning degradation, Gini>0.3 in 4/4 layers, REAP decisively beats random by >3 std at 50%. INCONCLUSIVE is the right call — the REAP infrastructure works, but the testbed is undertrained.
guardrail_flags=G5-note
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Lower PPL threshold to 100 (PPL=78.65 is reasonable for 15.5M/4K-vocab), or increase vocab to 8192 and retrain. The REAP machinery is clearly working — all downstream items (129-131) can proceed with the current checkpoint if the threshold is relaxed. Alternatively, retrain with vocab=8192 + hidden=384 for ~30 min and re-evaluate.
interesting=false
interesting_reason=n/a
