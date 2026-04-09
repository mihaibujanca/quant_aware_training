# Cycle 338 Plan

## Item
347: GPT-2 full 12-block weight spectrum census — why does LASER work on GPT-2?

## Approach
Single PyTorch script. Load `gpt2` from HuggingFace, loop all 12 blocks, SVD each `h[i].mlp.c_proj.weight`, report per-block spectrum stats. Direct comparison table with pgolf census (cycle 337).

## Steps

1. **Write `experiments/gpt2_weight_spectrum_census_cycle347.py`** (~90 lines)
   - `from transformers import GPT2Model`
   - Load `gpt2` (no GPU needed — weights only)
   - Loop `i in range(12)`:
     - Extract `model.h[i].mlp.c_proj.weight` (Conv1D, shape 3072x768)
     - `U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)`
     - Compute: top-1, top-5, top-13 energy fractions; rank for 50%/90%/95% energy
   - Print formatted table to stdout
   - Write `autoresearch/results/cycle_338_spectrum_census.json`
   - Write `autoresearch/results/cycle_338_summary.md` — table of all 12 blocks + pgolf comparison
   - Write `autoresearch/results/cycle_338_manifest.txt`

2. **Run**: `.venv/bin/python experiments/gpt2_weight_spectrum_census_cycle347.py`

3. **Verdict logic** (in script):
   - median block top-5 >= 8% → CONFIRMED (widespread concentration, explains LASER)
   - only b1 has top-5 >= 8%, all others < 5% → REFUTED (b1 is outlier)
   - else → INCONCLUSIVE

## Notes
- GPT-2 Conv1D stores weight as (n_input, n_output) = (3072, 768). SVD energy fractions are transpose-invariant, so no special handling needed.
- Cycle 280 measured b1 only: top-1=4.59%, top-5=11.67%. Expect similar or higher for other blocks if LASER viability is widespread.
- pgolf comparison baseline: max top-5 across all 13 blocks = 5.32% (b12), typical ~2.4%.

## Risks
- None significant. SVD of 12 matrices of size 768x3072, runs in <10s.
- Model download may take a moment if not cached, but gpt2-small is ~500MB.

## Expected outcome
CONFIRMED — pretrained LLMs typically develop low-rank structure through gradient descent on natural language. Multiple blocks should show top-5 energy well above pgolf's flat spectra.
