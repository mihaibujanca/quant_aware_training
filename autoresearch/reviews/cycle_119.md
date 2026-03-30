# Cycle 119 Review — Item 136: Cascade corrections at k=5 stacking

## R1: Premise validity — PASS
k=4 cascade corrections (cycle 115) confirmed 74% recovery at r=8. Extending to k=5 is a natural next step. The hypothesis (≥30% recovery) is reasonable: one more compressed block adds compounding error, so expecting lower recovery than k=4 is sound.

## R2: Measurement soundness — PASS
- Null baseline: uncorrected k=5 PPL ~93.9 from cycle 106 (seed 42). Float baseline ~51.7 (3-seed mean). Both documented.
- Sanity checks: plan requires float PPL within ±2 and uncorrected within ±10% of priors before proceeding.
- Recovery metric: (uncorrected - corrected) / (uncorrected - baseline) × 100. Standard, well-defined.
- Verdict thresholds (30%/10%) are well below baseline noise and above zero — no pitfall MP1 risk.

## R3: Capacity and statistical adequacy — PASS
3 seeds (42, 123, 7). k=4 std was ~7.5 PPL; target effect is 12.6 PPL (30% of 42-point gap). SNR ~1.7 with 3 seeds — marginal but adequate for a directional finding. Plan correctly notes boundary behavior (25-35%) is reported as-is.

## R4: Method-architecture alignment — PASS
Same LoRA correction architecture as cycle 115 (validated on k=4). Same compression pipeline (topk+lstsq+500 Adam). Same model (GPT-2). No novel architectural choices.

## R5: Known pitfall scan — PASS (all clear)
- **MP1** (trivial threshold): 30% threshold is well above noise. N/A.
- **MP2** (activation mismatch): GELU activations, corrections applied in residual stream. N/A.
- **MP3** (underdetermined fitting): r=8, d=768 → 2×768×8=12288 params per block. Training on 2000 steps × 4 batch × 128 seq = 1M tokens. Adequate.
- **MP4** (domain mismatch): Same domain (GPT-2 FFN on OWT) as prior cycles. N/A.
- **MP5** (capacity): Float model achieves ~51.7 PPL, corrections validated at k=4. N/A.
- **MP6** (verdict misclassification): Recovery metric is continuous and well-separated from trivial. N/A.
- **MP7** (post-LayerNorm): Corrections on FFN output, not post-LN activations. N/A.
- **MP8** (dimension vs rank): Not measuring intrinsic dimension. N/A.

## Issue: Block list discrepancy in agenda
The agenda item 136 says blocks `[1,3,4,7,11]` but the plan correctly overrides this to `[7,5,4,1,6]` from cycle 106 and the existing script. Block 11 is GPT-2's last block (index 11 of 12) and was never in the k=5 set. Block 3 was not in the k=5 selection either. The plan's note about this is correct and the override is justified. **Recommend updating agenda item 136 text to fix the block list.**

## Param count verification
Plan says 5 × 2 × 768 × 8 = 61,440 total correction params. Check: each block gets U (768×8) + V (8×768) = 12,288 params. 5 blocks × 12,288 = 61,440. Correct.

## Verdict: APPROVE
Clean plan. Well-scoped single-variable experiment with proper baselines, sanity checks, and fallbacks. Only minor issue is the stale block list in the agenda (cosmetic, plan handles it correctly).
