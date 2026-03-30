# Item 143: Best Correction Placement at Scale — GPT-2 FFN k=4

## Item
Take best placement from item 142 and 2 runners-up, run with 3 seeds, sweep rank.

## Context
- **Cycle 125** (item 142): REFUTED. P1 (after FFN) is within 2% of best alternative.
  - P1=69.52, P5=70.18 (-1.0% relative), P2=70.97, P4=71.65, P3=80.85
  - No non-P1 placement beats P1 by >=5% → no escalation triggered
- Item 143's premise was "take best placement and validate at scale." Since the sweep found placement doesn't matter (P1 already near-optimal), a 3-seed validation + rank sweep would just confirm the REFUTED finding from 142.

## Assessment
PLAN INVALID — predecessor item 142 was refuted, invalidating the premise. Running 143 would produce a predictable REFUTED outcome with no new information. The correction placement question is closed.
