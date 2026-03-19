status: PASS
original_outcome: REFUTED
audited_outcome: REFUTED
guardrail_flags: none
reason: The experiment cleanly measures what it set out to measure. Jaccard@256=0.517 is below the 0.60 refutation threshold, and the monotonic Jaccard progression from 0.276 to 0.844 across six probe sizes gives a clear, interpretable curve. The lstsq singularity at probe_size=512 (tokens=hidden_dim) is a genuine numerical hazard, correctly identified and not confounded with the Jaccard finding. Evaluation uses a fixed held-out corpus (seed=42, 200 sequences) separate from the probe corpus — no data leakage.
learned: Centroid-based topk channel selection is data-hungry: 3-way Jaccard grows monotonically with probe size but requires >2048 tokens to approach 0.85 on GPT-2 block 7. Even the 2048-token reference achieves only 0.844 (3-way), revealing inherent inter-seed variability in the scoring function. Secondary hazard: lstsq output-projection refit is numerically unstable when probe_tokens ~ hidden_dim.
next: Test whether a data-free (weight-only) channel scoring (e.g., output_norms alone, or weight_norm_in * weight_norm_out) can match or exceed the data-conditioned score at any probe size. If weight-only scores produce comparable Jaccard, the probe corpus can be eliminated entirely.
recommendations: (1) The 2048-token self-consistency Jaccard of 0.844 is a red flag for the scoring function itself — consider whether std_hidden is the noisy component and whether a more stable statistic (e.g., median or trimmed mean) would improve seed consistency. (2) The lstsq hazard should be addressed in the library with a condition-number check or ridge regularization before any deployment experiment hits the tokens~k regime.
interesting: true
interesting_reason: The 2048-token 3-way Jaccard of only 0.844 reveals that centroid-merge channel scoring has fundamental inter-seed instability beyond just small-sample noise, questioning the scoring formula itself.
