# Item
233 (P0): Anti-podal channel twin census — does activation pattern match column geometry?
Depends on item 230 (cycle 216). Confirmed criterion: "in blocks with ≥100 anti-podal pairs (from item 230), median corr(z_c, z_{c'}) < −0.3, AND sign disagreement rate > 60%".

# Context
Item 230 (cycle 216, REFUTED) found a TOTAL of ~31 anti-podal pairs (cos<-0.9) across all 12 blocks combined. The maximum any single block has is roughly 30 (block 0) and blocks 1-5 have 0-8 each. NO block has ≥100 anti-podal pairs.

# Dependencies
- 230 (REFUTED, cycle 216) — provides the anti-podal pair list. The list is essentially empty.

# Key Files
- autoresearch/results/cycle_216_wdown_geometry.json (anti-podal counts per block)

# Constraints
The Confirmed criterion is unmeasurable as written: it references "blocks with ≥100 anti-podal pairs" but no such block exists. Item 233's premise is invalid post-cycle-216.

# Handoff
Plan agent should write PLAN INVALID with reason: "P2 dependency failure — item 230 confirmed there are no blocks with ≥100 anti-podal pairs (max is ~30, total ~31 across all 12 blocks). Item 233's Confirmed criterion is unmeasurable. The cycle 216 audit explicitly recommended reframing 233 to use a softer threshold or different signal — but reframing is a new agenda item, not an in-cycle adjustment."
