status=PASS
old_outcome=REFUTED
new_outcome=REFUTED
reason=Implementation is correct: hook captures pre-GELU activations from c_fc, threshold logic properly classifies dead (max_z < -T) and near-linear (min_z > T) channels, contribution score formula is sound. Results are consistent across 3 seeds — only 5/36864 channels qualify at T=2, zero at T=3/T=4. PPL validation confirms zero impact from pruning the few qualifying channels. This is the 5th refutation of the dead-channel hypothesis (cycles 28, 32, 83, 128, 162).
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Stop revisiting dead-channel census for GPT-2; the hypothesis is decisively closed across 5 cycles. The contribution-score distribution being smooth and log-normal with no low-contribution cluster confirms there is no easy pruning target in the GELU activation space.
interesting=false
interesting_reason=
