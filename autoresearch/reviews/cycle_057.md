verdict: APPROVE
round: 1
issues: minor (no validity risk)
feedback: |
  Sound decomposition design. The 4096-token probe budget directly addresses the cycle 056
  audit finding (8x overdetermination at 6x/512 channels — adequate). Three clean variants
  isolate selection, refit, and stacking contributions.

  1. The hypothesis is near-trivially confirmable from existing data: cycle 056 single-block
     is +6.6-9.4%, stacked cycle 054 is +48%, so stacking accounts for ~80% by subtraction.
     The plan acknowledges this in Risks. The actually novel signal is Variant B (cross-selection
     refit) — if Shakespeare-selected channels + OWT lstsq matches OWT-native, it proves channel
     selection is domain-robust. Make sure results emphasize the A-vs-B comparison, not just
     the A-vs-C stacking gap that is already known.

  2. Minor framing inconsistency: the agenda item frames this as decomposing the domain transfer
     failure (Shakespeare vs OWT), but the plan's hypothesis focuses on stacking vs single-block.
     Both are useful, but the write-up should be explicit that the stacking question is partially
     answered by prior data, and the domain-transfer question (Jaccard + Variant B) is where new
     evidence comes from.

  3. Shakespeare corpus length: the fallback plan (document actual overdetermination) is fine,
     but if Shakespeare yields <2048 tokens the Jaccard comparison becomes weak. Consider
     asserting a minimum and skipping the Jaccard check if the corpus is too short, rather than
     reporting a low-confidence number.

  Review checks:
  - R1 (premise): Sound. Builds on cycles 054 and 056. Dependencies satisfied. No duplicate.
  - R2 (measurement): Sound. PPL relative change is well-calibrated. Jaccard is deterministic.
    The 60% threshold separates single-bottleneck from distributed failure modes.
  - R3 (statistical): Adequate. 4096 tokens / 513 columns = 8x overdetermination (MP3 cleared).
    3 seeds is standard. Jaccard is deterministic given probes.
  - R4 (method-architecture): Sound. Fork of established pipeline. Single-block vs stacked
    comparison is clean. Cross-selection refit (Variant B) is a well-designed ablation.
  - R5 (pitfalls): No pitfall applies. MP3 cleared. MP6 low risk — stacking PPL gap is large
    enough to produce meaningful signal. MP4 is precisely what Variant B tests.

pitfall_scan: No pitfalls apply. MP3 addressed by 4096-token budget (8x overdetermination).
