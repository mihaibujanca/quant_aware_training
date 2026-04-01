# Cycle 194 Implementation Plan: Self-Correction Depth Analysis

## Overview

Rerun hybrid progressive sweep (cycle 190 config) with residual error collection.
For each k=7,8,9,10,11: collect residuals from float and hybrid_k models, compute
per-layer error metrics. Determine whether PPL decrease at k=8-10 is due to error
cancellation (magnitude decrease) or redirection (magnitude grows, cos decreases).

**Script**: `experiments/self_correction_analysis_cycle194.py`
**Output**: `autoresearch/results/cycle_194_self_correction.json`

---

## Config

Match cycle 190:
```python
SEQ_LEN = 1024;  N_SEQS = 200;  BATCH_SIZE = 4;  SEED = 42
```

Run full k=0..11 for progressive PPL, but only collect residuals for k=6..11 (to save time).

---

## Hybrid Activation

```python
class GELUnegReLUpos(nn.Module):
    def forward(self, z): return torch.where(z < 0, F.gelu(z), F.relu(z))
```

(Same as cycle 190.)

---

## Residual Collection

For each k in [6, 7, 8, 9, 10, 11]:
1. Build GELUnegReLUpos model with blocks 0..k swapped
2. Collect residuals from ALL 12 layers (float and swapped)
3. Compute e_l[k] = res_float[l] - res_hybrid_k[l]  for l in 0..11
4. Metrics per layer:
   - frob[k][l] = ||e_l[k]||_F
   - cos_vs_k7[k][l] = mean_cos_similarity(e_l[k], e_l[7])  (alignment with k=7 error)

---

## Key Analysis

For blocks l=8,9,10 specifically:
- Compare frob[k][l] for k=8,9,10 vs k=7
- Compare cos_vs_k7[k][l]

If frob decreases k=7→k=10: CONFIRMED (partial cancellation)
If frob grows but cos changes significantly: REFUTED (redirection)

---

## Output JSON

```json
{
  "cycle": 194, "item": 204,
  "verdict": "...",
  "progressive_ppl": [...],  // full k=0..11 PPL (recomputed to confirm)
  "error_metrics_by_k": {
    "6": {"frob_per_layer": [...], "cos_vs_k7_per_layer": [...]},
    "7": {...},
    ...
  },
  "config": {...}
}
```
