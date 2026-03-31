# Item 169: Correction-First Compression

## Item
**Correction-first compression: train corrections, then compress residual — GPT-2 FFN k=4**
- Flip pipeline: compress then correct (current best PPL ~59) → correct then compress residual
- Type: empirical
- Success: PPL < 57 (≥3% over current best)
- Failure: PPL > 62 or corrections don't transfer
- Inconclusive: PPL 57–62

## Context
- **Working recipe** (cycle 158): topk+lstsq + LoRA corrections → PPL ~59 (1.6M + 49K params)
- **Alternative topk pipelines** (158): topk+polish ~70.7, random-init ~91.1
- **Joint training exhausted** (155–158): 5 consecutive refutations, composition barrier confirmed
- Last result: cycle 158 warm-start joint → PPL 73.0 (worse than polish), proved joint can't beat corrections

## Dependencies
- None (cycle 115 correction protocol already proven in results/)
- Requires: cycle 115 LoRA correction code + topk+lstsq compression from cycle 142

## Key Files
- `experiments/topk_init_joint_cycle158.py` (reference for topk+lstsq init)
- `autoresearch/results/cycle_115_cascade_correction.json` (LoRA correction baseline)
- `autoresearch/STATUS.md` (recipes section for baseline metrics)

## Constraints
- **Protocol** (4 steps): (a) train LoRA on topk+lstsq model, (b) extract C = W_corrected − W_compressed, (c) compute R = W_float − C, (d) compress R via topk+lstsq k=4
- **Eval**: 3 seeds, 200 OWT eval sequences (match cycle 115–158 eval protocol)
- **Baseline**: topk+lstsq+corrections PPL = 57.65 (float), 59.0 (stacked)

## Handoff
Test if reversing pipeline (correct → compress residual) beats current (compress → correct). Hypothesis: learning W_ffn − W_correction before compression preserves subspaces corrections can't reach. Protocol is fully specified; dependencies are complete.
