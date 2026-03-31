# Cycle 154 Plan: Joint Training Step Sweep (Item 164)

## PLAN INVALID

**Reason**: Cycle 153 revealed joint training is highly seed-dependent (PPL 91.9 in cycle 153 vs 69.76 in cycle 149 with nominally identical protocol). A step budget sweep on a single seed (as item 164 specifies) would be unreliable. Before sweeping budget, the reproducibility issue must be resolved (likely the cosine schedule interacting with random init differently across seeds). This requires a focused debugging cycle, not a budget sweep.

Additionally, cycle 153 showed CE-trained models resist post-hoc corrections (107.5 catastrophic), which undermines the practical value of joint training — it can't be combined with the correction approach that provides the biggest PPL gain (70.7 → 59).
