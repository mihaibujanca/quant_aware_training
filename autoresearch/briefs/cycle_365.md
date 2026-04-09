# Item
374. Adaptive k per block — equalize Frobenius error across GPT-2 blocks (R4)

# Context
- **Composition barrier**: 3-block PPL=47.43 (good); 6-block PPL=65.54 (fails). Uniform k=512 across blocks is wasteful.
- **Cycle 362 finding**: Frobenius error varies 4.5x across blocks under uniform k: b1=1385, b4=1860, b3=1822, b2=1637, b10=5424, b11=6293. Some blocks dominate, others underutilize their budget.
- **Per-block errors orthogonal** (c362): mean off-diagonal cosine=0.016, max=0.070. Errors already decorrelated; direct decorrelation won't help.
- **Hypothesis**: allocate k proportional to Frobenius error. High-error blocks (b10, b11) get larger k; low-error blocks (b1, b4) get smaller k. Hold total budget constant at 6×512=3072 across 6 blocks.

# Dependencies
- **362 (CONFIRMED)**: Error correlation matrix + Frobenius norms per block. Data in `cycle_362_error_correlation.json`.
- **355 (CONFIRMED)**: topk+lstsq+adapter works on all blocks; baseline PPL 48.47 (b2 single block).

# Key Files
- `experiments/gpt2_*.py` — baseline topk+lstsq+adapter scripts (e.g., c357, c360)
- `autoresearch/results/cycle_362_error_correlation.json` — Frobenius norms: b1=1385, b4=1860, b3=1822, b2=1637, b10=5424, b11=6293
- `aleph/qgeom/` — quantization library (topk selection, lstsq fitting)

# Constraints
- 6-block composition only (PPL threshold < 60 vs uniform 65.54).
- Hold 3072 total channels across 6 blocks (6×512 uniform baseline).
- Allocate k_i ∝ Frobenius_i, normalized by sum.
- Use existing topk+lstsq+adapter recipe; only vary k per block, not method.
- Run 3 seeds; report mean+std PPL.

# Handoff
**Task**: Implement adaptive-k allocation. Compute k_i = round(3072 × Frobenius_i / sum(Frobenius)). Run 6-block topk+lstsq+adapter [b2,b1,b10,b4,b3,b11] with per-block k values. Measure PPL. If PPL < 60, hypothesis CONFIRMED (adaptive k breaks composition barrier). If PPL >= 60, hypothesis REFUTED (error norms not the bottleneck).
