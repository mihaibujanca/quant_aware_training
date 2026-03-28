verdict: APPROVE
round: 1
issues: 0 blocking, 1 minor

## R1: Premise
Sound. Cycle 069 had a genuine implementation bug: `output_hidden_states[block_idx]` gives the block input (pre-attention), not the FFN input (post-attention + ln_2). The Jacobian J(x) = W_out @ diag(GELU'(W_in @ h + b)) @ W_in depends on h, so using the wrong h gives wrong Jacobians. The fix (register_forward_hook on mlp.c_fc) captures the exact tensor that c_fc receives, which is correct by construction. This re-run is well motivated.

## R2: Measurement
Subspace alignment via mean principal-angle cosine (svdvals of Vh_i @ Vh_j.T) is the standard metric for comparing subspaces. k=384 (agenda specification, 8x compression of d=768). The plan includes both raw alignment (for agenda verdict compatibility) and normalized excess (primary interpretive metric). The two-layer verdict system correctly separates "passes the 0.8 threshold" from "is meaningfully above random." At k=384/d=768, random alignment is ~0.50-0.58 — well below the 0.8 threshold, so trivial confirmation risk is low (unlike cycle 069 at k_99=470 where random was 0.716).

## R3: Seeds
3 seeds [42, 123, 7], 256 tokens per seed, 1000 random pairs per block. Consistent with prior cycles. Cross-seed std < 0.05 guard is reasonable since Jacobians depend on model weights (fixed) and token-dependent pre-activations (vary only with corpus). 200 random baseline pairs per block is adequate for estimating the random mean.

## R4: Method-architecture match
Hook-based FFN input capture is the same approach cycle 068 used successfully. Conv1D transposition (`.T`) is handled correctly per GPT-2 convention. GELU derivative via autograd is correct and validated in cycle 069. The only change from cycle 069 is the input capture method — all other computation (SVD, alignment, random baseline) is identical.

## R5: Pitfalls
- MP1 (threshold above trivial): At k=384, random baseline ~0.55 vs threshold 0.8 — gap of 0.25. The threshold is well above trivial. The plan also computes normalized excess to catch any edge cases. NOT TRIGGERED.
- MP3 (underdetermined fitting): No fitting; SVD is exact decomposition. NOT TRIGGERED.
- MP6 (verdict misclassification): The plan explicitly includes the two-layer verdict (raw + excess) to avoid misclassification. If all confirmed blocks are trivial (excess < 0.1), it reports "trivially_confirmed" rather than "confirmed." NOT TRIGGERED.
- MP7 (post-LayerNorm): The hook captures the input to c_fc, which is the output of ln_2 (LayerNorm). This IS post-LayerNorm. However, this is correct for this experiment: we want the Jacobian of the FFN map as it actually operates, and ln_2's output is the FFN's actual input. The Jacobian J = W_out @ diag(GELU'(z)) @ W_in is computed relative to this input, so the normalization is part of the ground truth. Unlike MP7 cases where we measure geometric properties OF the representations (which LayerNorm distorts), here we measure geometric properties of the MAP from its actual input. NOT TRIGGERED for this use case.
- MP8 (representation vs functional rank): The experiment measures functional subspace alignment (Jacobian SVD), not representation geometry. This is the correct approach per MP8 guidance. NOT TRIGGERED.

## Minor observations

1. **Comparison with cycle 069 results.** The validation step correctly requires that results differ from cycle 069 — if identical, the hook fix is not working. However, the plan does not specify a quantitative threshold for "different enough." Given that the FFN input differs from the block input by the full attention computation + ln_2, differences should be substantial for most blocks. A pragmatic check: if mean alignment differs by < 0.01 for any block, log a warning.
