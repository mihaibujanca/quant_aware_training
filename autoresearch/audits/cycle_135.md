status=PASS
old_outcome=confirmed
new_outcome=confirmed
reason=Salience-curvature rho=-0.745 exceeds 0.7 threshold. Curvature-damage |rho|=0.645 exceeds 0.5 threshold. Consistent across 3 seeds (within 0.09 rho). p-values significant (<0.05). Clear CONFIRMED.
guardrail_flags=none
recommendations=Proceed to item 146 (pullback error measurement) and item 147 (G-weighted lstsq). The negative curvature-damage correlation (high curvature = low damage) suggests protecting low-curvature late layers with more compression budget.
interesting=true
interesting_reason=First validation of curved inference on GPT-2. The inverted relationship (low curvature = high damage) means compression errors in semantically "straight" trajectory regions are most destructive. This is geometrically non-obvious — straight trajectories should be easier to approximate but in fact they carry the most semantic precision per unit of Euclidean error.
