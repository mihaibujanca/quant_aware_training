verdict: APPROVED
round: 1
issues: 0 blocking, 2 minor

## R1: Premise validity

Sound. Cycle 044 found +120% PPL degradation on WikiText-2 with Shakespeare-probed compression. Cycle 057 measured cross-corpus Jaccard of 0.47-0.54 on 2 corpora / 3 blocks. The hypothesis cleanly separates channel selection instability from lstsq refit instability by measuring Jaccard and cross-eval PPL independently. If Jaccard is high but cross-eval PPL still diverges, the problem is pinned to the refit step, which directly informs item 66 (mixed-corpus refit).

## R2: Measurement soundness

Null baselines are adequate. Random Jaccard at k=512 of n=3072 is J = 512/(2*3072 - 512) = 0.091 — correctly computed (P5). Any measured Jaccard well above 0.09 is meaningful. Cross-eval PPL is normalized against uncompressed GPT-2 baselines (step 5). The 4x4 cross-eval matrix is the right design: it separates "which channels were selected" from "how well they generalize."

Within-corpus cross-seed Jaccard as a stability reference (expected ~0.84 from cycle 067) is a good sanity check. The plan also cross-references cycle 057's Shakespeare/WikiText-2 Jaccard range (0.47-0.54) for consistency.

## R3: Statistical adequacy

3 seeds [42, 123, 7] for corpus construction with mean/std reporting. Topk selection is deterministic given corpus, so seeds only enter through corpus sampling — correctly identified in P7. 4 corpora x 4 blocks = 6 unique cross-corpus Jaccard pairs per block, 16 cross-eval PPL values per block. Adequate for the confirmation/refutation thresholds stated.

## R4: Method-architecture alignment

Correctly uses `probe_ffn_geometry` -> `derive_ffn_compression` -> `swap_compressed_ffns` pipeline. GPT-2 GELU FFN and Conv1D transposition via `_standard_weight()` explicitly noted (P6). Block selection [7, 5, 4, 1] matches the progressive stacking configuration from prior cycles. 6x compression (k=512 of 3072) is the standard operating point.

## R5: Known pitfall scan

- MP1 (threshold above trivial): Not applicable — Jaccard threshold (0.7) is well above random baseline (0.091).
- MP2 (activation regime): Not applicable — measuring channel selection indices, not activation ranges.
- MP3 (underdetermined fitting): Not applicable — no fitting in channel selection. Lstsq refit is a separate step and not the focus here.
- MP4 (domain mismatch): Directly addressed — the entire experiment tests domain transfer.
- MP5 (capacity): Not applicable — using pretrained GPT-2, capacity is fixed.
- MP6 (verdict misclassification): The three-way verdict (confirmed/refuted/inconclusive) with the 0.4-0.7 inconclusive zone is well-designed. No risk of false refutation.
- MP7 (post-LayerNorm): Not applicable — measuring channel selection indices, not representation geometry.
- MP8 (representation vs functional rank): Not applicable — measuring set overlap (Jaccard), not dimensionality.

No pitfalls triggered.

## Minor observations (non-blocking)

1. **C4 download reliability:** The fallback (drop to 3 corpora) is reasonable, but C4 streaming from HuggingFace can also be slow or require authentication. The plan should ensure `datasets` is installed and HF_TOKEN is available, or preferably pre-download the sample before the experiment script runs.
2. **Corpus diversity coverage:** Shakespeare, WikiText-2, OpenWebText, and C4 are all English-language text with significant overlap in vocabulary distribution. A more adversarial test would include code or structured text, but for the stated hypothesis (explaining cycle 044's Shakespeare vs WikiText-2 gap), these 4 corpora are sufficient.

## Verdict

APPROVED. The plan is well-structured, correctly scoped, and directly addresses the open question from cycles 044/057. The three-way verdict with the inconclusive zone (0.4-0.7 Jaccard) avoids MP6-style misclassification. Seeds, baselines, and sanity checks are all present.
