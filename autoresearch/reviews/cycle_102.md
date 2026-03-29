verdict: APPROVE
round: 1

## R1: Premise validity
The gap is genuine: no published zero-shot GELU->ReLU swap on pretrained GPT-2 without finetuning. Survey covers 4 relevant papers, all either train from scratch or finetune. The hypothesis (GELU smoothness is training-time, not inference-time) is well-motivated by SUGAR and ReLU Strikes Back. Sound premise.

## R2: Measurement soundness
PPL evaluation on OWT + WikiText-2 with 200 seqs and 3 seeds is adequate. Per-block swap methodology (one FFN at a time) is correct. One issue: **[I1]** The plan says "hook each block's post-activation output" for sparsity measurement, but this is ambiguous. Sparsity must be measured after `mlp.act` (the ReLU output) but BEFORE `mlp.c_proj` (the down-projection). After c_proj, the zeros are mixed by the linear map and sparsity is lost. The plan must specify the exact hook point.

Correlation with cycle 048 P1_frob_std is a good cross-check. Spearman is the right choice for 12 data points.

## R3: Statistical adequacy
3 seeds with mean +/- std is sufficient for a directional PPL comparison. 200 seqs per eval is consistent with prior cycles. No issues.

## R4: Method-architecture alignment
The plan correctly identifies `NewGELUActivation` at `model.transformer.h[i].mlp.act`. Verified: HF GPT-2 uses `NewGELUActivation` (tanh approximation), not `nn.GELU()`. Replacing with `nn.ReLU()` is correct.

**[I2]** Missing consideration: NewGELU has a small negative lobe (minimum ~-0.17 at x~-0.67). Weights in `c_proj` were trained expecting these small negative activations. ReLU clips them to 0, creating a systematic positive bias in the down-projection output. This bias accumulates through the residual stream. The plan should measure mean activation shift per block (mean of ReLU(x) - GELU(x) over the eval data) to quantify this effect and disentangle it from the nonlinearity change.

## R5: Known pitfall scan
- MP1 (threshold above trivial): N/A -- comparing PPL, not detecting patterns.
- MP2 (detection method mismatched to activation regime): **Relevant.** ReLU output range [0, inf) vs GELU (-0.17, inf). The sparsity metric (fraction of exact zeros) is well-defined for ReLU but the magnitude distribution differs. Addressed by I2.
- MP3 (underdetermined): N/A -- no fitting.
- MP4 (phenomenon-domain mismatch): N/A -- testing GPT-2 on GPT-2.
- MP6 (verdict misclassification): PPL is a direct task metric, not a proxy. Pass.
- MP7 (post-LayerNorm): PPL measurement, not geometric. Pass.
- MP8 (representation vs functional rank): N/A.

## Issues

**I1 (moderate):** Sparsity hook point ambiguous. "Post-activation output" could mean after act or after the full MLP. Must hook `mlp.act` output (the hidden_states after GELU/ReLU, before c_proj). Fix: register hook on `model.transformer.h[i].mlp.act` specifically.

**I2 (minor):** No measurement of systematic activation shift from GELU's negative lobe. GELU(x) < 0 for x in ~(-0.67, 0), ReLU clips to 0. This positive bias propagates through residual stream. Fix: log mean(ReLU(x) - GELU(x)) per block as a diagnostic. Not blocking, but useful for interpreting results.

## Advisory

**A1:** If full-swap PPL is in the inconclusive range (+10-30%), the per-block sensitivity profile becomes the main deliverable. Consider also reporting a "cumulative swap" curve: swap blocks 0..k for k=0..11, measuring how damage accumulates. This distinguishes "each block adds +2%" (linear) from "blocks 6-8 cause a cascade" (nonlinear interaction).

## Notes
- Abort condition (baseline PPL outside [25, 65]) is well-calibrated for GPT-2 on OWT.
- Verdict logic thresholds (10% confirmed, 30% refuted) are reasonable given survey evidence that finetuning closes to 1-2%.
- The experiment cleanly separates from prior cycles and directly enables the geometric compression pivot if confirmed.
