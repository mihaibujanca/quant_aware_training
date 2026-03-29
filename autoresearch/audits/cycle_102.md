# Audit: Cycle 102 — Zero-shot GELU→ReLU swap — GPT-2

**Item:** 107
**Claimed outcome:** REFUTED
**Audited outcome:** REFUTED
**Status:** PASS

## Phase 1: Verdict Check

Refutation gate: full-model swap PPL > +30% relative. Result: baseline PPL 22.26, full-swap PPL 1874.11, delta = +8320%. Far above the +30% threshold. **Refutation gate clearly triggered.**

## Phase 2: Quality Checks

### G1: Implementation correctness

**Swap target: CORRECT.** GPT-2's `GPT2MLP.forward()` calls `self.act(hidden_states)` where `self.act` is `NewGELUActivation()`. The experiment replaces `model.transformer.h[i].mlp.act` with `nn.ReLU()`. Verified via `inspect.getsource(GPT2MLP)` that `forward()` uses the instance attribute, not a hardcoded function call. The swap is correctly intercepted.

**PPL computation: CORRECT.** Standard cross-entropy loss with `reduction="sum"`, divided by total tokens, then `exp()`. No off-by-one: input is `batch[:, :-1]`, target is `batch[:, 1:]`.

**Per-block isolation: CORRECT.** Each block is swapped, evaluated across 3 seeds, then restored via saved reference to the original `NewGELUActivation` instance before proceeding to the next block (lines 222-233). No cross-contamination.

**Sparsity measurement: CORRECT.** Hooks on `.mlp.act` post-activation output while model is in full-swap state. Counts `out == 0` (exact zeros from ReLU). This is the right place and the right check.

### G2: Reproducibility

3 seeds [42, 123, 7], 200 sequences each, seq_len=1024. Results are tight:
- Baseline: 22.26 ± 0.76 (CV=3.4%)
- Full swap: 1874.1 ± 19.7 (CV=1.1%)
- Per-block results consistent across seeds (all deltas in same direction, similar magnitudes)

Fully reproducible.

### G3: Results interpretation

+8320% is massive but the evidence fully supports it. Three seeds with CV=1.1% for the full-swap measurement. The magnitude makes sense: GELU and ReLU differ most in the negative regime. GELU passes small negative signals (the "bump" around z ∈ [-2, 0]), while ReLU kills them entirely. In GPT-2, where every weight, bias, LayerNorm parameter, and attention pattern was trained against GELU semantics, zeroing all negative activations simultaneously is a catastrophic architectural change.

The per-block results are informative and internally consistent:
- Block 0: +2664% alone — this block handles the initial embedding-to-feature transformation, where GELU's soft gating is most critical
- Blocks 1-2: +8.7% to +22% — moderate
- Blocks 3-6: +22% to +57% — the "working" middle layers are moderately sensitive
- Blocks 7-10: +5.3% to +12.3% — late layers are surprisingly tolerant
- Block 11: +29.7% — the final block is more sensitive again (output-facing)

The U-shaped sensitivity (high at start, lower in middle-late, spike at end) is plausible: the first and last blocks have less residual-stream "slack" to absorb activation changes.

### G4: Counterintuitive?

Not counterintuitive. A zero-shot activation swap without any finetuning is a major architectural perturbation. The weight matrices were trained with GELU semantics — `W_proj @ gelu(W_fc @ x)` bakes in the expected output distribution of GELU, including its non-zero response to negative inputs. Swapping to ReLU changes the effective rank and magnitude of intermediate representations. The +8320% result is large but not surprising.

**Baseline PPL check:** This cycle uses seq_len=1024 (vs 128 in prior cycles). Baseline PPL = 22.26 on OWT with 1024-token sequences is correct for GPT-2 small. Prior cycles with seq_len=128 got ~51.7 — the difference is expected since longer sequences give the model more context for prediction. The baseline sanity check in the code (range [15.0, 65.0]) was set appropriately for the 1024 context length.

### G5: What did we learn?

1. **Zero-shot GELU→ReLU is not viable for GPT-2.** The GELU bump carries essential information that ReLU cannot replicate without retraining. This closes the door on "free" piecewise-linear compression by simple activation swap.

2. **Block-level sensitivity is highly non-uniform.** Block 0 is catastrophically sensitive (+2664%), while blocks 9-10 tolerate the swap well (+5-7%). This heterogeneity is useful: if a finetuning-based swap were pursued, blocks 9-10 could be swapped first with minimal retraining.

3. **Post-ReLU sparsity is extremely high.** 56-98% zeros per block. If the swap could be made to work (via finetuning), the resulting ReLU network would have massive activation sparsity — directly exploitable for inference-time compression and sparse matmul.

4. **No correlation with cycle_048 complexity.** Spearman rho = -0.24, p = 0.44. The GELU→ReLU sensitivity of a block is not predicted by its weight complexity. This suggests the sensitivity is driven by the activation distribution (how much mass is in the negative-but-near-zero range that GELU passes and ReLU kills), not by weight structure.

### G6: Method failure or genuine refutation?

**Genuine refutation of the zero-shot swap hypothesis.** The implementation is correct — the right attribute is swapped, the forward pass uses it, and the restore logic works (verified by per-block isolation producing correct baselines between blocks). The +8320% is not an artifact.

However, this does NOT refute GELU→ReLU swaps in general — only the zero-shot (no finetuning) version. Published work (e.g., ReluLLM, ShiftedReLU) shows that with modest finetuning (a few hundred steps), the swap can be made nearly lossless. The experiment correctly scoped its claim to zero-shot only.

## Verdict

REFUTED is correct. Zero-shot GELU→ReLU swap in GPT-2 produces catastrophic PPL degradation (+8320%). The per-block analysis provides useful secondary data showing non-uniform sensitivity and high post-ReLU sparsity.
