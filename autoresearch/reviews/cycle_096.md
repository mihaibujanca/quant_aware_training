verdict: REJECT
round: 2

## Round 2 review — fixes from round 1

**I1 (freeze-both ablation):** Fixed. Ablation C added, interaction term computed correctly in step 5.

**A2 (descriptive/causal distinction):** Fixed. Step 3 explicitly labels norm deltas as descriptive, step 4 as causal.

**A1 (sum-to-100%):** Fixed. Step 5 header and text explicitly disclaim the sum-to-1 assumption.

## New blocking issue

**I2 (verdict logic ignores the interaction correction it computes).** Step 5 correctly decomposes into `attn_contribution`, `ffn_contribution`, and `interaction`. But step 6's verdict logic applies the 70% threshold to `attn_frac = attn_contribution / D` and `ffn_frac = ffn_contribution / D` — these are the *raw* ablation deltas that still conflate each sub-block's own error with the interaction term. If interaction is substantial (which the plan acknowledges is likely given the sequential attention-then-FFN architecture), `attn_frac + ffn_frac > both_contribution / D`, and neither fraction is a clean causal attribution. The plan does the hard work of computing the decomposition and then doesn't use it.

Fix: the verdict should threshold on interaction-corrected quantities. For example, define `attn_clean = attn_contribution - interaction` (or split interaction proportionally, or use Shapley values: `attn_shapley = 0.5 * (attn_contribution + both_contribution - ffn_contribution)`). Then apply the 70% threshold to `attn_shapley / D`. Alternatively, add an explicit clause: if `|interaction / D| > 0.15`, verdict is INCONCLUSIVE regardless of individual fractions, since the confound is too large for a clean attribution.

**I3 (per-block dominant undefined).** Step 6 references `per_block_dominant[L]` for the consistency check, but no per-block verdict logic is defined. The per-block ablations are supplementary (step 4 note), so either define how `per_block_dominant` is computed from per-block ablation PPLs, or drop the consistency check from the verdict and keep it as a supplementary diagnostic.
