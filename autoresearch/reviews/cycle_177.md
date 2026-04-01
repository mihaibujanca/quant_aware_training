# Cycle 177 Review — SoLA-style split compression (Item 187)

## R1: Premise validity — VALID WITH CAVEAT

The premise is sound: topk+lstsq discards all information from 2560/3072 channels. Recovering even partial tail signal via low-rank projection could improve the lstsq fit. The caveat: the split approach requires FULL c_fc (3072x768 per block), which is 3x more params than standard topk+lstsq. The item says "matched total parameter count" but this is physically impossible without also compressing c_fc (which introduces nonlinear distortion via GELU). The plan addresses this by: (a) comparing at same topk=512 (split has more params), (b) honestly reporting param counts, (c) noting that the extra params are the ORIGINAL model's c_fc (no new params created).

## R2: Measurement soundness — OK

Baseline reproduction (topk+lstsq at k=512) is included as r=0 case. Baseline PPL sanity check [25, 65]. 3 seeds, 200 eval sequences. Polish applied uniformly to all configs.

## R3: Capacity and statistical adequacy — OK

SVD of h_tail (4096 x 2560) is well-conditioned for rank 2-16. lstsq with (512+r) features from 4096 tokens is overdetermined. 3 seeds sufficient for PPL differences > 2.

## R4: Method-architecture alignment — MINOR NOTE

SVD of h_tail is DATA-DEPENDENT: the projection V_r is computed from calibration data and may not generalize to eval data. But this is the same situation as lstsq fitting (also data-dependent). The eval/cal data split (probe_train vs eval_val) handles this.

The SplitCompressedMLP must compute full 3072-dim hidden at inference (not just top-k), which costs more FLOPs. This is fine for PPL comparison but would matter for actual deployment.

## R5: Known pitfall scan — OK

- MP3 (underdetermined lstsq): 4096 samples >> 512+16 features. No risk.
- MP5 (nonlinear distortion): SVD of activations (post-GELU) is valid — no nonlinearity between SVD and lstsq.
- Param mismatch: honestly reported. Not a bug, an inherent property of the split approach.

## Summary

No blocking issues. The param mismatch is inherent and honestly handled. The experiment tests whether tail channel information, even at low rank, improves lstsq fitting quality.

**Verdict: ACCEPT.**
