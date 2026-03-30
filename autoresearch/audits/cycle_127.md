# Audit — Cycle 127 (Item 144)

**SVD of learned corrections: geometric alignment — GPT-2**

```
status=PASS
old_outcome=inconclusive
new_outcome=inconclusive
reason=Verdict criteria have a 0.3-0.5 gap but the code handles it correctly via the "Mixed" else branch. Max error alignment 0.46 (block 5) is 3.6x null p99 — real signal, but below the 0.5 CONFIRMED/REFUTED thresholds. P1 PPL matches cycle 125 reference within 0.13. Implementation, null baseline, and hook placement are all correct.
guardrail_flags=none
recommendations=Compare correction SVD against PROPAGATED error (cumulative from upstream compressed blocks) rather than LOCAL single-block error. Block 7 has the largest correction norm but lowest error alignment — likely compensating for upstream propagation, which the current local-error SVD cannot capture.
interesting=true
interesting_reason=Corrections are 3-4x above null alignment with compression error but only in middle blocks (4-5), while block 7 (largest correction) has weak local-error alignment — suggesting corrections split between local error undoing and propagated error compensation.
```

## Guardrail Detail

- **G1 (Implementation):** PASS. Hook on `model.transformer.h[bid]` output captures residual stream after full block (attn+FFN+residual), which is the pre-LayerNorm input to next block. Correction applied there is post-FFN pre-LayerNorm as intended. LowRankCorrection is `x + (x @ V^T) @ U^T` with zero-init U — correct residual parameterization. Null baseline uses orthogonalized random rank-4 subspaces in R^768, 1000 trials — appropriate.
- **G2 (Reproducibility):** PASS. Seed=42 throughout, commit f41a7e7 verified. P1 PPL=69.39 vs cycle 125 ref=69.52 (delta=0.13, well within 2.0 tolerance).
- **G3 (Interpretation):** PASS with note. The criteria define three buckets (>0.5 confirmed, >0.5 refuted, <0.3 inconclusive) but blocks 4-5 fall in the 0.3-0.5 gap. The code's else branch labels this "Mixed: INCONCLUSIVE" which is the conservative correct call. However, the result IS informative: moderate error alignment + weak attention alignment means corrections partially undo compression error but not via attention amplification channels.
- **G4 (Counterintuitive):** PASS. Block 1 attn_delta=0 is physically correct (no upstream compression). Block 7's large correction norm with low error alignment is explained by propagated upstream error (not captured by local error SVD).
- **G5 (Learning quality):** PASS. 69.3% PPL recovery, matching cycle 125.
- **G6 (Method vs genuine):** PASS. Alignments 3-4x above null p99 are genuine subspace overlap, not noise. The moderate magnitude is a real finding about correction structure.
