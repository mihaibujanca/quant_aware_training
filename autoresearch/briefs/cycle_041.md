# Cycle 041 Brief — Item 59

## Task
**Generalization probe: single-block 4x geometric compression on GPT-2-medium -- FFN**

All compression results so far are on GPT-2-small (12 layers, d=768, FFN=3072). Item 59 tests whether topk + lstsq refit transfers to GPT-2-medium (24 layers, d=1024, FFN=4096). Single-block swaps at 4x on 6 evenly spaced blocks (0, 4, 8, 12, 16, 20), 3 seeds, 200 eval sequences.

## Hypothesis
The geometric compression method is not GPT-2-small-specific. At least 4/6 medium blocks tolerate 4x with mean relative PPL <=10%.

## Success Criteria
- **Confirmed**: >=4/6 blocks tolerate 4x at mean relative PPL <=10%.
- **Refuted**: <=1/6 blocks pass 10% threshold.
- **Inconclusive**: 2-3/6 blocks pass, partial transfer.

## Dependency
- Item 52: DONE (cycle 035, multi-seed 6x confirmed on GPT-2-small)

## Key Context
- GPT-2-medium: 24 layers, d=1024, FFN=4096 (~345M params, ~1.4GB float32)
- Pipeline: `aleph/lm_geometric_compression.py` with `hf_model_name="gpt2-medium"` -- auto-discovers 24 blocks via `transformer.h`, same `c_fc`/`c_proj` structure
- GPT-2-small reference: all 6 easiest blocks tolerate 6x within 5% PPL (cycle 035)
- Conservative 4x ratio chosen because medium FFN is 33% wider -- matching effective difficulty
