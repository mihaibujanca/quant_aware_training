# Cycle 173 — Design Review

## Checklist

1. **Does C1 match cycle 102's ReLU definition?** Yes — cycle 102 replaced GELU with ReLU. Setting GELU output to 0 where z<0 is exactly ReLU behavior (since ReLU(z) = 0 for z<0 and ReLU(z) ≈ GELU(z) for z>>0... wait, ReLU(z) = z for z>0, GELU(z) ≈ z for z>>0 but GELU(z) < z for small positive z). This is NOT exactly ReLU. C1 should be: replace gelu(z) with relu(z), not just zero-out negative regime.

**REVISION NEEDED:** C1 should replace the activation function with ReLU, matching cycle 102. The current plan's "zero out where z<0" IS effectively ReLU since GELU(z) ≈ 0 for z<0 anyway (the bump is tiny). But to be precise and match cycle 102 exactly, use relu(z).

Actually let me reconsider. The item says "C1=kill negative (→ReLU)". The key behavior is: for z<0, ReLU gives 0 while GELU gives a small negative bump. So C1 = relu(z) is the right implementation. For z>0, ReLU(z) = z while GELU(z) ≈ z * Phi(z) which is slightly less than z for small z. This difference is small but nonzero. The item says "kill negative" which means only the z<0 regime changes. So: C1 = gelu(z) where z>0, 0 where z<0. This is "kill negative" without changing positive.

Wait, re-reading the item: "C1=kill negative (h_neg→0, equivalent to ReLU at this block)". The parenthetical says equivalent to ReLU. In the GELU decomposition: h = h_pos + h_neg where h_pos = gelu(z) for z>0, h_neg = gelu(z) for z<0. Killing h_neg means the positive channels keep their gelu(z) value, negative channels become 0. This IS approximately ReLU but not exactly (ReLU changes positive regime too: relu(z) = z vs gelu(z) ≈ z*Phi(z)).

**Decision:** Implement C1 as: output = gelu(z) where z>=0, output = 0 where z<0. This matches the item's "h_neg→0" definition exactly. Note in results that this differs slightly from full ReLU swap (cycle 102).

2. **Is hooking strategy correct?** The GELU in GPT-2 is `model.transformer.h[b].mlp.act`. Hook gets input[0]=z, output=gelu(z). We can modify output in a forward hook. Yes, correct.

3. **Are means computed on the same data used for PPL?** Plan says compute stats then eval. Should use separate sequences for stats vs eval, or compute stats on the same 2048 tokens. The item says "2048 OWT tokens" — using the same tokens is fine for an ablation table (not training anything). Keep it simple.

4. **Is C5 correctly defined?** Item says "replace both with means (block outputs constant E[y])". C5 should use the unconditional mean E[gelu(z)] across all tokens, not E[y_neg] + E[y_pos]. But E[gelu(z)] = E[y_neg] + E[y_pos] (by linearity of expectation when you define y_neg and y_pos as zero-padded). Actually: if y_neg(z) = gelu(z) where z<0, 0 where z>0, and y_pos(z) = gelu(z) where z>0, 0 where z<0, then E[y] = E[y_neg] + E[y_pos]. So C5 = mu_neg + mu_pos = mu_total. Yes.

But wait — the means mu_neg and mu_pos are computed per-channel over tokens where that channel IS in the respective regime. So mu_neg[c] = mean(gelu(z_c)) over tokens where z_c < 0. This is NOT the same as E[y_neg][c] which would be mean over ALL tokens (including zeros from z>=0 tokens). Need to be careful.

**REVISION NEEDED:** Clarify: mu_neg[c] = E[gelu(z_c) | z_c < 0] (conditional mean) vs E[gelu(z_c) * 1(z_c<0)] (unconditional with zeros). The C3 condition says "replace negative with its token-mean" — this should use the conditional mean (what the negative output looks like on average when it IS negative). When a channel has z>=0, it keeps its gelu(z) value. When z<0, replace with the average negative output for that channel.

For C5: replace ALL outputs with E[gelu(z)] = E[gelu(z)*1(z<0)] + E[gelu(z)*1(z>=0)] = p_neg * mu_neg_conditional + p_pos * mu_pos_conditional.

Actually simplest: mu_total[c] = mean(gelu(z_c)) over ALL tokens. This is the unconditional mean. C5 replaces every token's output with this constant vector.

5. **Interaction check formula correct?** PPL(C1)+PPL(C2)-PPL(C0) vs PPL(C5). If regimes are independent, killing both should be additive. If interacting, the actual "kill both" (C5 is not exactly kill both — it's replace with means). The interaction check is informative but not a formal test. Fine as-is.

## Verdict: APPROVE

Minor clarifications needed in implementation (conditional vs unconditional means) but plan is sound. Proceed to execute.
