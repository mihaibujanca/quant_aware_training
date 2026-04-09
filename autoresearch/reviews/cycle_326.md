# Cycle 326 Review

## Plan summary
Load the parameter-golf 13L_640d_322 checkpoint config, extract `num_encoder_layers` and `skip_weights`, and cross-reference the encoder/decoder boundary to the cycle 318 eff_rank profile (b2->b3 drop from 86 to 4.89). Determine whether the boundary explains the rank collapse.

## R1: Premise validity
Sound. Cycle 318 established the eff_rank profile with a dramatic b2->b3 drop. The hypothesis that this coincides with the encoder/decoder boundary is a natural structural explanation worth checking. The plan already inspected the config JSON and found `num_encoder_layers: 0` (auto = `num_layers // 2 = 6`), placing the boundary at block 6, not block 3. The pre-conclusion (REFUTED-no-correlation) follows logically.

## R2: Measurement soundness
No activation measurement is performed. The plan reads architectural config fields (`num_encoder_layers`, `skip_map`, `skip_weights`) from the checkpoint/config. These are definitive structural facts, not noisy measurements. The skip_weight norms provide a useful secondary check (whether skips are actually used), which could matter for an INCONCLUSIVE edge case.

## R3: Statistical adequacy
N/A -- deterministic config lookup plus Frobenius norms of stored parameters. No statistical inference.

## R4: Method-architecture alignment
N/A -- no hooks, no forward passes, no activation capture.

## R5: Known pitfall scan
- MP1-MP5: Not applicable (no thresholds, no activation detection, no fitting, no domain transfer, no capacity comparison).
- MP6 (verdict misclassification): The plan handles this correctly. REFUTED-no-correlation is appropriate when the boundary is 3 blocks away from the drop. The item's own INCONCLUSIVE clause covers the degenerate case (boundary at b3 but skips unused). No risk of recording a false refutation.
- MP7 (post-LayerNorm): Not applicable -- reading config, not measuring activations.
- MP8 (representation vs functional rank): Not applicable -- no rank measurement in this cycle.

## Issues
- **Minor overengineering**: Two Python scripts for what is essentially a config JSON read + parameter norm print. A single script would suffice. Not blocking.

## Verdict
APPROVE (round 1). The plan is correct, minimal-risk, and the pre-conclusion is sound: encoder/decoder boundary is at block 6, the b2->b3 drop at block 3 is deep inside the encoder, so the boundary does not explain the rank collapse. No pitfall violations.
