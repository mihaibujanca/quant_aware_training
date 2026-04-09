cycle = 367
item = 376
verdict = REFUTED
audit_outcome = PASS
guardrail_flags = G3_minor
notes = Manifest says "Block 10 and 11 rank 6th and 5th most important" but actual ranking is 7th and 2nd. The raw ranking array is correct; only the prose summary is wrong. Implementation is correct (hooks zero MLP output, not weights). 3 batches with mean/std. Baseline PPL ~64-66 explained by fixed-window vs sliding-window. Block 0 delta +3021 is plausible (catastrophic). Importance ranking [1,7,2,6,9,10,8,3,5,4,11,0] is valuable input for item 377 mixed compression.
interesting = no
