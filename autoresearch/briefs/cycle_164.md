# Brief — Cycle 164 (Item 174)

## Item
**Positive-channel ablation — GPT-2 FFN** [INTERPRETABILITY]
- Complement of cycle 102: where cycle 102 killed negative channels (PPL +8320%), kill POSITIVE channels.
- Goal: isolate whether positive token-variation or positive magnitude drives computation.

## Context
- **Cycle 102 outcome**: GELU→ReLU (kill negative regime) → PPL 22.26 → 1874.11 (+8320%).
- **z_mean ≈ −1.0** across all blocks (learned during training, cycle 178 hypothesis).
- **Negative regime carries 87–98% token-independent signal** (b_leak, cycle 176).
- **Open question**: Is positive regime equally essential, or is most computation in the negative leak?

## Dependencies
- Pretrained GPT-2 baseline (PPL ~22.26).
- OWT validation set (4096 tokens, standard config).
- Per-block GELU decomposition: split z > 0 (positive) vs z ≤ 0 (negative).

## Experimental Design
- **Condition A**: GELU(z) = 0 where z > 0 (keep negative-only).
- **Condition B**: GELU(z) = mean(GELU(z)) where z > 0 (replace positive with token mean).
- Measure per-block PPL (blocks 0–11) and full-model PPL.
- Report: (a) PPL delta, (b) ||y_pos_mean|| vs ||y_pos_token|| (signal magnitude comparison).

## Criteria
- **Confirmed**: Killing positive channels → >+100% PPL (both regimes essential).
- **Refuted**: Killing positive channels → <+20% PPL (negative carries most).
- **Inconclusive**: +20–100% PPL delta.

## Handoff
Ablate positive channels block-by-block. Report per-block and stacked results. If PPL swing is <+20%, conclude negative regime is dominant (cycle 176 bias hypothesis strengthened). If >+100%, conclude symmetric importance and focus next experiments on interaction dynamics.
