# Item
136: Cascade corrections at k=5 stacking — GPT-2 FFN

# Context
- k=5 compression (blocks [1,3,4,7,11]) at 6× degrades PPL by ~60% (cycle 106 baseline: +62.8% uncorrected)
- Cycle 115 confirmed r=8 LoRA cascade corrections recover 74.3% of k=4 degradation (PPL 59.1 from 80.7)
- Cycle 117 (item 133) timed out at 140+ min on both k=5 and k=8 — split into items 136 and 137

# Dependencies
- **Predecessor (cycle 115, item 118)**: k=4 cascade corrections confirmed at 74% recovery
  - Model: topk+lstsq+500 Adam on W_down at 6×, then LoRA C_l(x) = x + U_l @ (V_l @ x) trained 2000 Adam steps
  - Baseline: PPL 51.7; Uncorrected: 80.7; r=8 corrected: 59.1
  - Criterion: r=8 is optimal trade-off (r=4: 72.6%, r=16: 74.1% recovery)
- **Reference (cycle 106, item 101)**: k=5 polish confirms +22-24% reduction vs +62.8% uncorrected
  - Gradient polish shows k=5 error is trainable (not intractable cascade limit)

# Key Files
- Code: `experiments/cascade_correction_k5_k8_gpt2.py` (existing, from cycle 117 timeout attempt)
- Results template: `autoresearch/results/cycle_115_cascade_correction.json` (use for schema)
- Baseline data: `autoresearch/results/cycle_106_k5_polish.json` (k=5 uncorrected ~+60% PPL)

# Constraints
- Single seed (42) first — if reasonable, expand to seeds 123 and 7
- Compute budget: ~15 min per seed (5 blocks × compress+polish + correction training)
- r=8 only (fixed rank, validated by cycle 115 on k=4)
- Report: uncorrected PPL, corrected PPL, recovery fraction ≥30% to confirm
- Total correction params: 5 blocks × (768×8 + 8×768) = 61,440 params

# Handoff
**Question for planner**: k=5 has more blocks (5 vs 4) so cascade might compound worse. Does 74% recovery from k=4 transfer? If r=8 hits <10% recovery, will need to investigate whether deeper stacking breaks the correction mechanism (not just parameter count).

Blocking assumptions:
- Code path exists and compiles (verified)
- Quantization codec (topk+lstsq) is stable from cycles 104-115
- Adam settings (lr=1e-4, 2000 steps) are stable across k values
