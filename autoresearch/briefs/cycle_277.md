# Item
287 (P0): 2-block [1, 11] hybrid — K=512 on [1] + rank-22 c_proj on [11].

# Context
- Item 285 / cycle 275 REFUTED: identical recipe ([1] K=512 + block 11 rank-22, trainable 411,648) → best_mean=48.200 ± 0.048.
- Item 287 is a duplicate of item 285 — same BLOCKS, same K, same trainable surface.
- The only new element in 287 is a within-session [1,4] K=512 control. The main hybrid measurement was already performed in cycle 275.

# Dependencies
- Cycle 275 already measured the hybrid.

# Pre-flight
- P1 FAILS: duplicate of cycle 275 / item 285.
- The [1,4] control alone is not worth a full cycle. Cycle 266 already measured [1,4]=43.85.
- Move 287 to PAST as INCONCLUSIVE with reason "duplicate of cycle 275 item 285".

# Handoff
Plan agent: write PLAN INVALID.
