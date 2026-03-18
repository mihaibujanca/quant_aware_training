# Plan — Cycle 025: GELU PWL Approximation on GPT-2

## Pre-flight checks
- P1: `.venv/bin/python -c "import transformers, torch; print('ok')"` — verify deps
- P2: verify Shakespeare probe is cached: `ls ~/.cache/huggingface/` or check `aleph/lm_geometric_compression.py` for cache path
- P3: confirm `model.transformer.h[0].mlp.act` is nn.GELU (not functional call)
- P4: confirm `compute_perplexity` function is importable from prior experiment pattern

## File to write
`experiments/gelu_pwl_gpt2.py`

## Implementation

### PWL activation class
```python
class PiecewiseLinearGELU(nn.Module):
    """K-piece PWL approximation of GELU over [lo, hi].
    Outside: 0 for x < lo, x for x > hi (GELU asymptotes).
    Breakpoints: uniformly spaced in [lo, hi].
    """
    def __init__(self, n_pieces: int, lo: float = -5.0, hi: float = 5.0):
        super().__init__()
        # breakpoints and values are fixed (no grad)
        bps = torch.linspace(lo, hi, n_pieces + 1)
        vals = F.gelu(bps)
        self.register_buffer("bps", bps)
        self.register_buffer("vals", vals)
        self.lo = lo
        self.hi = hi

    def forward(self, x):
        # torch.searchsorted needs 1D breakpoints
        idx = torch.searchsorted(self.bps, x.contiguous()) - 1
        idx = idx.clamp(0, len(self.bps) - 2)
        x0 = self.bps[idx]
        x1 = self.bps[idx + 1]
        y0 = self.vals[idx]
        y1 = self.vals[idx + 1]
        t = (x - x0) / (x1 - x0 + 1e-12)
        interp = y0 + t * (y1 - y0)
        # outside range handling
        result = torch.where(x < self.lo, torch.zeros_like(x),
                    torch.where(x > self.hi, x, interp))
        return result
```

### GPT-2 loading
Use `aleph.lm_geometric_compression.load_nanochat_teacher` and `build_probe_corpus` / `ProbeCorpusSpec` / `TeacherLoadSpec`.

### Perplexity function
Copy from `experiments/procrustes_linearity_gpt2.py` lines 39–60.

### Experiment flow
1. Load model, get baseline ppl
2. Full-model sweep: for K in [3, 5, 10, 20]:
   - Replace all 12 blocks' `.mlp.act` with `PiecewiseLinearGELU(K)`
   - Measure ppl, compute relative increase
   - Restore original act
3. Block-by-block sweep (K=3 only):
   - For i in range(12): replace only block i's GELU, measure ppl, restore
4. Save JSON: `autoresearch/results/cycle_025_gelu_pwl_gpt2.json`
5. Save PNG: `autoresearch/results/cycle_025_gelu_pwl_gpt2.png`
   - Top panel: ppl vs K (full-model)
   - Bottom panel: per-block ppl increase for K=3

## Evaluation
- Apply confirmed/refuted/inconclusive criteria from agenda item 33
- Report: baseline ppl, 3-piece ppl and relative increase, 10-piece relative increase
- Block-by-block: which blocks are most sensitive (top-3 by ppl delta)

## Output files
- `experiments/gelu_pwl_gpt2.py`
- `autoresearch/results/cycle_025_gelu_pwl_gpt2.json`
- `autoresearch/results/cycle_025_gelu_pwl_gpt2.png`
- `autoresearch/results/cycle_025_manifest.txt`
