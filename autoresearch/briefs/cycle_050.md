# Item
**65: Oracle-guided adaptive compression ratio — GPT-2 FFN**

# Context
Cycle 048 confirmed Jacobian Frobenius std (P1) as a cheap oracle: rho=0.84 at 8×, p<0.001, robust to outliers (rho_without_L0=0.818). Operationalize it by assigning per-block compression ratios inversely proportional to P1 values — easy blocks (low P1) get 10×, hard blocks (high P1) get 4× or skip.

# Dependencies
- None (item 65 has no dependencies)
- Prior results feed into comparison: cycle 036 (k=4@6× uniform: +9.59%), cycle 042 (k=4@10× uniform corrected: +8.45%)

# Key Files
- **Oracle signal**: `autoresearch/results/cycle_048_local_complexity_gpt2.json` (P1 Frobenius std per block)
- **Baselines**: `cycle_036_accumulation_6x.json` (k=4@6×), `cycle_042_multiblock_10x.json` (k=4@10× corrected)
- **Implementation**: `experiments/multiblock_*` series for reference pattern
- **Recipe**: `autoresearch/results/cycle_045_recipe.md` (validated compression pipeline)

# Constraints
- Must use k=4 accumulation (easiest-first order: blocks 7,5,4,1)
- Ratio assignment rule: ratio_i = 10 × (P1_min / P1_i) or hardcap at 4× for hardest blocks
- Compare oracle-guided vs both uniform baselines on PPL and total FFN param reduction
- Eval on Shakespeare (probe corpus; cycle 044 confirmed corpus specificity)
- Success criterion: oracle k=4 ≤ 8% PPL increase AND ≥ total param reduction vs uniform 6×

# Handoff
Compute per-block P1 oracle values from cycle 048. Assign ratios inversely proportional to P1 (normalize to [4,10]). Run k=4 multi-block accumulation with these oracle-assigned ratios. Measure PPL vs float + vs uniform baselines (cycle 036, 042). Report per-block ratio allocation, final PPL, total FFN param reduction, and whether oracle allocation beats both uniform benchmarks.
