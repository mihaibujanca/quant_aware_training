verdict: APPROVE
round: 1

## R1: Duplication

Cycle 048 computed P1 on Shakespeare probes only. No prior cycle computes P1 on WikiText-2 or performs cross-corpus Spearman comparison of block rankings. Not a duplicate.

## R2: Known pitfalls

- **MP1**: Not applicable (no detection threshold).
- **MP2**: Not applicable (no activation-regime detection).
- **MP3**: Not applicable (no fitting/learning — only Jacobian computation and rank correlation).
- **MP4**: Same domain (GPT-2 FFN) as cycle 048. Pass.
- **MP5**: Pretrained GPT-2. Capacity established. Pass.
- **MP6**: Spearman rho on n=12 blocks is low-power but the test is legitimate. A rho of 0.85 yields p < 0.001 at n=12. Verdict thresholds (0.85 / 0.5) are well-separated and match the agenda criteria. Pass.

## R3: Method soundness

- Clean experimental isolation: identical Jacobian code, only corpus changes. Fork from `gpt2_local_complexity.py` is appropriate.
- Cycle 048 JSON stores P1 values at `proxies_by_layer[block_idx]["p1_frob_std"]`, not as a flat array. The plan says "Load Shakespeare P1 values from cycle_048 JSON" but does not specify this key path. Minor — the implementer should extract `[d["proxies_by_layer"][str(i)]["p1_frob_std"] for i in range(12)]`.
- 128 probes from WikiText-2 valid (~36K tokens) is well within headroom.
- Low statistical power (n=12) is an inherent limitation of GPT-2's block count, not a design flaw. A rho of 0.7 would be statistically significant (p ~ 0.01) but classified "inconclusive" — this is conservative, which is acceptable.

## R4: Feasibility

Same Jacobian computation as cycle 048 with a corpus swap. ~30 min estimated runtime. Within budget.

## R5: Verdict criteria

- CONFIRMED: Spearman rho >= 0.85. Clear, testable.
- REFUTED: Spearman rho < 0.5. Well-separated from confirmation.
- INCONCLUSIVE: 0.5 <= rho < 0.85. Covers the ambiguous middle.
- No overlap between verdict regions. Sound.

## Minor notes

1. Plan should specify the nested key path for extracting P1 from cycle 048 JSON: `proxies_by_layer[str(i)]["p1_frob_std"]` for i in 0..11. Keys are string-typed in JSON.
2. The plan's risk note about block 0/11 dominance inflating rho is good — consider reporting rho with and without extreme blocks (as cycle 048 did with the leverage-point guard).
