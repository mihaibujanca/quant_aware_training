# Cycle 037 Brief: Item 56 — Total FFN Parameter Reduction at <10% PPL

## Item
Heterogeneous per-block compression ratios combining cycle 033+036 findings: 4-block accumulation (blocks 7,5,4,1) at best safe ratio per block (all 6× from cycle 034), then test whether adding block 3 as 5th block at lower ratio (2×-3×) stays <10% PPL. Report total FFN param % saved vs original 12-block budget at <10% boundary.

## Context
- Cycle 033: k=4 uniform 4× = +8.84% PPL (CONFIRMED, <10% safe)
- Cycle 034: All 6 blocks (7,5,4,1,3,6) tolerate 6× individually (30/30 within 5% threshold)
- Cycle 035: 3-seed validation confirms 6× safe on blocks 7,5,4,1,3,6 (5/6 sub-baseline PPL = regularization effect)
- Cycle 036: 6× multi-block k=4 = +9.59% PPL (INCONCLUSIVE), only +0.75pp worse than 4× despite 50% more compression
- **Key insight**: ratio dominates block-count degradation. 4→6× costs <1pp at k=4; k=4→k=5 cliff costs +8pp regardless of ratio.

## Dependencies
- Item 53 (cycle 036): INCONCLUSIVE — 6× multi-block available
- Item 52 (cycle 035): INCONCLUSIVE — per-block 6× confirmed real, no noise

## Key Files
- `autoresearch/results/cycle_036_accumulation_6x.json` — 6× k=4 baseline: +9.59%, ratio=5.995
- `autoresearch/results/cycle_034_ratio_sweep.json` — per-block {2×,3×,4×,5×,6×}; all 6× safe
- `experiments/multiblock_accumulation_6x.py` — baseline code (cycle 036)

## Constraints
- **Budget calculation**: FFN params per block = 3072×2 (W_up/W_down) = 6144. Total 12 blocks = 73,728.
  - k=4 at 6×: 4×(6144/6) + 8×6144 = 4,096 + 49,152 = 53,248 (27.8% reduction)
  - k=5 at [6×,6×,6×,6×,3×]: 4×1024 + 2048 + 8×6144 = 55,296 (25.0% reduction)
- **Success**: total ≥25% reduction AND k=5 at ≤3× stays <12% PPL OR k=5 at ≤2× stays <10% PPL

## Handoff
Run heterogeneous k=5 sweep: [7×6×, 5×6×, 4×6×, 1×6×, 3×{2×,3×,4×}]. Test which 3rd ratio (if any) lets k=5 stay <10% or <12%. If none, characterize k=4→k=5 cliff geometry (block identity vs compounding error).
