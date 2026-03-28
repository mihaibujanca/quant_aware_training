verdict: APPROVE
round: 1
issues: 0 blocking, 1 minor

## R1: Premise
Sound. Cycle 076 was fatally confounded: condition A had n=512 samples for p=513 parameters (underdetermined lstsq, MP3 violated), producing catastrophic PPL ~3.6M. Condition C had n=2048 and gave PPL ~113. The 6-order-of-magnitude gap cannot be attributed to corpus diversity vs sample adequacy. This experiment fixes the confound by matching n=2048 for both conditions. The expected baseline from cycle 044 (Shakespeare-only at 2048 → PPL ~135) provides a strong sanity check.

## R2: Measurement
Raw compressed PPL on WikiText-2 validation, same metric as cycles 044 and 076. The 5% relative threshold for confirmation is appropriate: if Shakespeare-only at 2048 gives ~135 and mixed gives ~113, that is a ~16% relative improvement — well above threshold. The 3% refutation threshold is tight enough to be meaningful. Both thresholds are reasonable given cross-seed std was ~1% in cycle 076 condition C.

## R3: Seeds
3 seeds [42, 123, 7], consistent with all prior cycles. Cross-seed std < 20% guard (tightened from 30% since the underdetermined lstsq variance source is removed). Adequate.

## R4: Method-architecture match
Identical pipeline to cycle 076: same hooks, same `collect_hidden_activations`, same `_fit_output_projection`, same `build_compressed_pack_from_refits`. The only change is condition A' uses 2048 Shakespeare tokens instead of 512. The code path is fully validated from cycle 076. Channel selection from Shakespeare atlas at 2048 tokens matches cycle 076 step 2 exactly.

## R5: Pitfalls
- MP3 (underdetermined fitting): At 6x compression on block 7, hidden_dim=128, p=129 (with bias). n=2048 for both conditions gives N/p=15.9. Well above 2x. This was the ENTIRE POINT of this experiment — fixing the MP3 violation in cycle 076. NOT TRIGGERED.
- MP4 (domain mismatch): The experiment tests cross-domain generalization intentionally. The eval domain (WikiText-2) differs from refit domain (Shakespeare or mixed). This is by design. NOT TRIGGERED.
- MP5 (insufficient capacity): GPT-2 at 6x compression on 4 blocks — validated in cycles 044, 070, 076. Compressed model can achieve PPL ~113 (cycle 076 condition C). NOT TRIGGERED.
- MP6 (verdict misclassification): If both conditions give similar PPL, that is a genuine refutation (corpus diversity doesn't help at adequate n). The metric clearly varies in a discriminating range (PPL 70-200). NOT TRIGGERED.
- MP7 (post-LayerNorm): Hooks on mlp module capture FFN internals, not post-LayerNorm. NOT TRIGGERED.
- MP8 (representation vs functional rank): Not measuring rank; measuring downstream PPL after lstsq refit. NOT TRIGGERED.

## Minor observations

1. **Condition A' vs cycle 044 comparison.** Cycle 044 used `probe_ffn_geometry` (which builds both the atlas and the refit data in one pass). This plan uses `probe_ffn_geometry` for the atlas but `collect_hidden_activations` for the refit data. These are separate forward passes on the same Shakespeare tokens, so results should match, but minor numerical differences from batching/ordering are possible. Not blocking — the within-experiment A' vs C comparison is the primary metric, and cycle 044 is only a sanity check.
