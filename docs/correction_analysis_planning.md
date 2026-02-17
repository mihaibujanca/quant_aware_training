# Geometric Correction: Status and Decision Point

## End goal

A low-precision correction network that recovers a solid amount of
quantization error on GPT-2 (pre-trained, quantized to 4-bit). The
correction network must be tiny relative to the model (<1% of params).

## What we've learned

### MLP findings (Experiments 0-3) — strong results

| Finding | Numbers |
|---------|---------|
| Metric distortion dominates | 88-98% of error energy |
| Metric-only correction recovers float accuracy | All configs tested |
| c_local = -E@a (0 learned params) recovers float | Uniform-width MLPs |
| Error is low-rank | rank₉₅ ≈ 2 at most layers |
| Pre-ReLU correction cascades perfectly | Residual ~1e-6 |
| Propagated error dominates at depth | 94% at layer 12 |

### Transformer findings (Phase 1 analysis) — mixed results

Config: MiniGPT, d_model=128, d_ff=512, 4 blocks, Shakespeare char-level.
Float ppl=5.46, quantized ppl=5.69 (+0.23 degradation).

| Analysis | Finding | Verdict |
|----------|---------|---------|
| **SVD decay** | rank₉₅ = 35→94 across blocks (out of 128). Block 3: top-32 = 69% | **Bad.** Corrections are high-rank. V_L with k=32 misses 31% of energy at deep blocks. |
| **Cross-block structure** | Mean principal angles 56-71° (random = 90°) | **Bad.** Subspaces are nearly orthogonal. No free weight-sharing. |
| **Cascade coupling** | Correcting L-1 reduces L's error by 20-37% | **Good.** Sequential correction has leverage. Effect grows with depth. |
| **Metric vs topological** | 80-84% metric, 16-20% topological | **OK.** Lower than MLPs (88-98%) but still dominantly metric. GELU softer than ReLU. |
| **Residual containment** | Error grows 5.2x in 4 blocks (vs 470x/12 layers in MLPs) | **Good.** Roughly linear growth, not exponential. Residual stream works. |
| **c_local** | Only 22-29% error reduction per block | **Bad.** Only corrects W2 error, misses W1 quantization effect on GELU activations. |

### Honest assessment

The geometric framework (metric/topological, polyhedral partitions) gave
genuine understanding. But the specific correction architecture designed
around it (V_L + tiny shared MLP) has two problems in transformers:

1. **Corrections are high-rank.** The "pre-compute low-rank basis,
   predict coefficients" idea needs nearly full-rank bases at deep
   blocks, defeating the compression.

2. **c_local is weak.** The cheap analytical correction that was the hero
   in MLPs barely helps here because FFNs have an internal dimension
   change (128→512→128). c_local only fixes the down-projection; the
   up-projection error propagates through GELU and isn't captured.

What IS working: residual containment (linear error growth, not
exponential) and cascade coupling (corrections compound). The error
structure is correctable — just not cheaply low-rank.

## Decision: two quick experiments before committing or pivoting

Both are ~50 lines added to the existing script, run in minutes.

### Experiment A: Activation-aware SVD

Raw SVD of correction matrices treats all directions equally. LQER showed
that weighting by activation statistics drops needed rank from 600 to
32-64. We should check: does activation-weighted SVD of our corrections
show faster decay?

```
# Weight corrections by how much each direction matters for the task
# S_diag = diag(activation_norms_per_direction)
# SVD of C @ S_diag instead of C
```

**If rank₉₅ drops from 94 to ~15 after weighting:** V_L idea is back on.
The raw SVD was misleading — most high-rank directions are data-irrelevant.

**If rank stays high:** The error genuinely occupies many dimensions and
low-rank correction won't work without rethinking the approach.

### Experiment B: Full per-block c_local (both W1 and W2)

Current c_local = -E_W2 @ a only fixes the down-projection. The full
local correction also needs the W1 effect:

```
c_full = ffn_float(x) - ffn_quant(x)   # full oracle at block level
c_W2_only = -E_W2 @ a_float            # what we currently compute
c_W1_component = c_full - c_W2_only    # what W1 error contributes
```

**If c_W2_only explains 80%+ of c_full:** W2 correction is sufficient,
the W1 component is small (GELU absorbs most of the W1 error).

**If c_W1_component is 50%+:** The FFN's internal nonlinearity creates
substantial error that simple c_local can't touch. Need to either (a)
correct at the pre-GELU level inside the FFN, or (b) accept that
analytical correction won't work and go fully learned.

## What happens after A and B

| Exp A result | Exp B result | Direction |
|-------------|-------------|-----------|
| Low rank after weighting | W2 dominates | **Continue:** V_L + activation weighting, small k |
| Low rank after weighting | W1 matters | **Continue with modification:** correct at two points per block (pre-GELU + post-FFN) |
| Still high rank | W2 dominates | **Consider pivoting:** error is genuinely high-dimensional. Weight-space correction (LQER-style LoRA) may be more natural |
| Still high rank | W1 matters | **Pivot:** the geometric framework gives understanding but not a viable architecture. Pursue LQER/QAT/other |

## Alternative directions (if pivoting)

1. **LQER-style weight correction**: SVD of E with activation-aware
   scaling → LoRA-format additive correction. Well-validated, scales
   to LLMs. Our geometric analysis could still inform the scaling.

2. **QAT with geometric regularization**: Use the metric/topological
   decomposition as a regularizer — penalize topological distortion
   during quantization-aware training.

3. **Selective precision**: Use geometric analysis to identify which
   layers/blocks need higher precision. Budget-constrained mixed
   precision informed by correction rank.

## Completed experiments

| # | Experiment | Key finding | Files |
|---|-----------|-------------|-------|
| 0 | Canonical error analysis | Error compounds 470x, oracle correction to 1e-6 | `notebooks/canonical_error_correction.py` |
| 1 | Low-rank correction | SVD rank₉₅ ~ 2 but task needs rank 5+ | `notebooks/canonical_learned_correction.py` |
| 2 | Correction distillation | c_local (0 params) recovers float acc, skip arch essential | `notebooks/correction_distillation.py` |
| 3 | Geometric decomposition | Metric-only correction recovers float acc everywhere | `notebooks/canonical_learned_correction.py` |
| 4 | Per-block transformer | See findings table above | `experiments/transformer_correction_analysis.py` |
| - | Variable-dim (broken) | c_local insufficient for AE/transformer | `experiments/correction_distillation_architectures.py` |

## Key references

- [LQER (ICML 2024)](https://arxiv.org/abs/2402.02446): Activation-aware SVD. Rank 32-64 suffices.
- [QEP (2025)](https://arxiv.org/abs/2504.09629): Explicit error propagation across layers.
- [LoftQ (ICLR 2024)](https://arxiv.org/abs/2310.08659): LoRA init compensating quantization error.
- [Low-Rank Correction (ICLR 2025)](https://openreview.net/forum?id=FA3iYp1y6z): 10% rank → 50% gap closed.

## Files

- Research brief: `docs/geometric_error_correction_brief.md`
- Geometric figures: `notebooks/geometric_correction_figures.py`
- Transformer analysis: `experiments/transformer_correction_analysis.py`
- Results: `docs/transformer_correction_analysis.json`
