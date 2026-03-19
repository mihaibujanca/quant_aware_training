verdict: APPROVE
round: 1

## R1: Duplication

No prior cycle uses OpenWebText. Cycles 044/045 used Shakespeare and WikiText-2 only. This fills the explicit training-distribution gap identified in the agenda. Not a duplicate.

## R2: Known pitfalls

- **MP1**: Not applicable (no detection threshold).
- **MP2**: Not applicable (PPL metric, no activation-regime detection).
- **MP3** (underdetermined fitting): lstsq correction at layer 8 fits ~768×3072 params from 2048 samples. Heavily underdetermined, but identical to all prior cycles (005, 033, 044, 045). This is the pipeline under study, not a design flaw. Noted but not blocking.
- **MP4**: GPT-2 + OpenWebText is the closest domain match possible (GPT-2 was trained on WebText; OpenWebText is the public replica). Pass.
- **MP5**: Pretrained GPT-2. Capacity established. Pass.
- **MP6**: Verdict thresholds well-separated (+15% vs +30%). Prior data: Shakespeare +8.5%, WikiText +44%. OpenWebText result should land in a distinguishable range. Pass.

## R3: Method soundness

- 500 streaming documents yields ~250K+ tokens (avg OWT doc ~500+ words). 450 sequences × 128 tokens = 57.6K tokens needed. Sufficient headroom.
- Agenda item says "first 10K tokens" but plan says "500 documents" — minor inconsistency. The plan's 500-doc approach is implemented and sufficient. Not blocking.
- Clean experimental isolation: identical pipeline to cycle 045, only corpus changes. Blocks, correction layer, seeds, hidden_fraction all held constant.
- Streaming the first 500 docs is deterministic (no pre-shuffle). OWT is not chronologically ordered, so position bias is low risk, but the sample is not random.
- The plan does not clarify whether correction fitting uses the same 200 probe sequences or separate data. If the same data informs both neuron selection and correction coefficients, there is a mild double-dipping risk. This matches the prior-cycle regime so it is consistent, but worth noting for future work.

## R4: Feasibility

Same pipeline as cycle 045 with one corpus swap. ~30 min is a reasonable estimate. Well within budget.

## R5: Verdict criteria

- CONFIRMED: mean corrected degradation <= +15%. Clear, testable.
- REFUTED: mean corrected degradation >= +30%. Well-separated from confirmation.
- INCONCLUSIVE: 15-30%. Covers the ambiguous middle.
- No overlap between verdict regions. Sound.

## Minor notes

1. The "10K tokens" in the agenda vs "500 documents" in the plan is a cosmetic inconsistency — the plan's approach is correct.
2. First-500-docs deterministic sampling is not truly random, but OWT ordering is not problematic. A note in the output about this would be good practice.
