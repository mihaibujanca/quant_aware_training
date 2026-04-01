# Cycle 190 Implementation Plan: GELU-neg + ReLU-pos Hybrid Activation

## Overview

Test a hybrid activation function that uses GELU for negative inputs (preserving negative
regime) and ReLU for positive inputs (changing positive-regime shape). Apply to all 12 GPT-2
blocks simultaneously. Compare PPL to baseline (~22.7), C3 all-12 (27.314), and ReLU+bias
all-12 (~42.6) to isolate the cost of positive-regime shape change alone.

**Output**: `autoresearch/results/cycle_190_gelu_neg_relu_pos.json`
**Script**: `experiments/gelu_neg_relu_pos_cycle190.py`

---

## Configuration

Match cycle 188 exactly for fair comparison:
```python
SEQ_LEN = 1024
N_SEQS = 200
BATCH_SIZE = 4
SEED = 42
DEVICE = "mps" if available else "cpu"
OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
```

---

## Hybrid Activation

```python
class GELUnegReLUpos(nn.Module):
    def forward(self, z):
        return torch.where(z < 0, F.gelu(z), F.relu(z))
```

No bias correction needed — GELU(z) for z<0 is the exact same output as the float model
in the negative regime. Only z>=0 is changed.

---

## Implementation Steps

1. Load GPT-2, tokenizer, OWT data (same as cycle 188)
2. Compute baseline PPL on N_SEQS with SEQ_LEN=1024
3. Deepcopy float model
4. For all 12 blocks: replace `swapped.transformer.h[b].mlp.act = GELUnegReLUpos()`
5. Compute hybrid PPL
6. Also compute progressive sweep (k=0..11 blocks swapped) to match cycle 188 format
7. Write JSON results

---

## Verdict Criteria

- **Confirmed**: hybrid all-12 PPL > 35 (+54% over ~22.7) — positive-regime shape IS dominant
- **Refuted**: hybrid all-12 PPL < 28 (+23%) — negative-regime loss dominates
- **Inconclusive**: 28-35 PPL range

---

## Output JSON

```json
{
  "cycle": 190, "item": 200,
  "verdict": "...",
  "baseline_ppl": 22.7,
  "hybrid_all12_ppl": ...,
  "hybrid_pct_increase": ...,
  "progressive": [{"k": 0, "ppl": ...}, ...],
  "config": {...}
}
```

---

## Key Comparisons to Include

- baseline_ppl (~22.7)
- hybrid_all12_ppl (this experiment)
- c3_all12_ppl = 27.314 (cycle 188, for reference)
- relu_bias_all12_ppl = ~42.6 (cycle 181, for reference)
