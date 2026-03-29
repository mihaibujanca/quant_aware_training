# Cycle 102 — Item 107: Zero-shot GELU→ReLU Swap — GPT-2

## Item
Load pretrained GPT-2, swap every `nn.GELU()` → `nn.ReLU()`, evaluate PPL on OWT and WikiText-2 with zero finetuning. Measure per-block sensitivity and post-ReLU sparsity.

## Context
Cycles 099–101 closed three correction paths: subspace projection, Jacobian factorization, GELU' clustering—all refuted. Research is geometrically stuck in GELU smoothness. Item 107 is the unmeasured baseline: nobody has published raw zero-shot GELU→ReLU on a pretrained model. Survey `gelu_relu_inference.md` cites 4 papers that train from scratch or finetune after swap. If swap PPL gap < +10%, piecewise-linear MASO theory applies and geometric compression thesis survives.

## Dependencies
None. Independent empirical measurement.

## Key Files
- Pretrained GPT-2 (HF: `gpt2`)
- OWT / WikiText-2 eval splits (already in repo if used in prior cycles)
- Survey: `/autoresearch/surveys/gelu_relu_inference.md` — literature evidence that smoothness is training, not inference

## Constraints
- Zero finetuning (immediate swap evaluation)
- Full-model PPL on OWT + WikiText-2
- Per-block single-channel swap (identify sensitive blocks)
- Report activation sparsity post-ReLU (% zeros per block)
- Threshold: confirmed if ≤+10% PPL, refuted if >+30%, inconclusive if +10-30%

## Handoff
Experiment is straightforward: load model, patch activations, eval. No novel methods. Report PPL table (full + per-block), sparsity histogram, correlation with cycle 048 compression difficulty (P1_frob_std). Key output: does MASO theory apply (ReLU viable) or is GELU bump critical?
