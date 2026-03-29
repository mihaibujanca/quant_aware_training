# Audit — Cycle 095

- **Item**: 109 — Attention compression v2 — GPT-2
- **Cycle**: 095
- **Auditor verdict**:

```
status=PASS
old_outcome=inconclusive
new_outcome=inconclusive
reason=5/12 blocks pass at <=+5% PPL (blocks 1,3,4,9,10). Nearest miss is block 8 at +5.58% — not a rounding issue. Block 0 catastrophic (+102%) is expected (first-layer token mixing). Phase 2 multi-block (top-3 simultaneous): +13.0% for both rank-11 and rank-16 — interaction effects dominate, not per-block approximation quality. INCONCLUSIVE is correct: 1 short of CONFIRMED, clearly not REFUTED.
guardrail_flags=minor_manifest_errors
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The rank-16 = rank-11 finding in multi-block is the most informative result. It mirrors cycle 093's ratio-independent c for FFN. Next step: test attention + FFN joint compression on the 5 compressible blocks to see if attention and FFN error interact or are orthogonal. Also: block 0 skip (compress 11/12 blocks, skip block 0) as an easy win.
interesting=true
interesting_reason=Rank-independence in multi-block attention compression parallels FFN ratio-independence (c~0.41). Both point to a shared mechanism: inter-block error propagation is the bottleneck, not per-block approximation fidelity. This is converging evidence for a fundamental compression ceiling set by network depth, not width.
```

## Guardrail Details

**G1 (Implementation)**: PASS. Per-head SVD correctly extracts each head's Q/K/V slice from the fused Conv1D weight (768, 2304), transposes to (2304, 768), indexes `[base + h*64 : base + (h+1)*64, :]` to get (64, 768) per head, applies `torch.linalg.svd(full_matrices=False)`, truncates to rank-11, reconstructs via `(U[:,:r] * S[:r]) @ Vt[:r,:]`. Write-back correctly re-transposes for Conv1D. W_O (`c_proj`) is never touched — only `c_attn` is modified. PPL uses the shared `_evaluate_model` from `aleph/lm_geometric_compression.py`, same as all prior cycles.

**G2 (Seeds)**: PASS. 3 seeds [42, 123, 7]. Per-seed deltas stored in `block_all_deltas`. Standard deviations are small (0.18-0.76% for non-catastrophic blocks), indicating seed variance is not a concern. Block 0's std of 7.8% is large in absolute terms but small relative to its +102% mean.

**G3 (Threshold analysis)**: PASS. 5/12 blocks pass at <=+5%. The nearest miss is block 8 at +5.58% (all 3 seeds: +6.45%, +5.45%, +4.83%). Even the best single-seed value for block 8 (4.83%) wouldn't change the mean. Block 2 (+6.48%) and block 7 (+6.24%) are also clearly above threshold. The gap between pass and fail is genuine, not a borderline rounding issue.

**G4 (Block 0)**: Expected. Block 0 handles initial token mixing — the first attention layer must attend across all positions to establish context. Full-rank QKV is structurally necessary here. This is consistent with the literature (early transformer layers are most sensitive to pruning/compression).

**G5 (Rank-16 = rank-11)**: The most interesting finding. Phase 2: rank-11 gives +12.98% +/- 1.35%, rank-16 gives +13.00% +/- 1.51%. These are statistically identical. This means going from 3.7x to 5.4x QKV compression costs nothing additional when multiple blocks are compressed simultaneously. The damage is from inter-block interaction, not per-block approximation error — same mechanism as FFN c~0.41 ratio-independence (cycle 093). Converging evidence that compression ceilings are depth-determined.

**G6 (Method)**: No failure. The experiment cleanly tests its hypothesis. Design is improved from v1 (cycle 090, rejected for wrong compression ratio). W_O exclusion is explicitly noted and justified (QKV feasibility scan, not full-block compression).

## Minor Errors

1. **Manifest top-3 blocks**: Manifest says "blocks 1, 4, 3" but JSON records [1, 4, 9]. Block 9 has delta +4.04% vs block 3's +4.07% — block 9 is the correct third member. Cosmetic error, does not affect Phase 2 execution (code uses sorted deltas).
2. **Real block compression ratio**: Manifest claims "~3.1x with W_O included" but correct calculation gives ~2.6x (QKV compressed: 329K params + W_O unchanged: 590K = 919K; original: 2.36M; ratio = 2.57x). Does not affect any results — the 5.4x QKV-only ratio is correctly computed and is what the experiment actually tests.
