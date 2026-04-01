# Cycle 193 Implementation Plan: C3+Hybrid Combined Modification

## Overview

Apply BOTH C3 (negative regime → per-channel mean) AND Hybrid (positive regime → ReLU)
to all 12 GPT-2 blocks simultaneously. Measure all-12 PPL.

**Script**: `experiments/c3_hybrid_combined_cycle193.py`
**Output**: `autoresearch/results/cycle_193_c3_hybrid_combined.json`

---

## Config

Match cycle 188 for fair comparison (baseline ~22.7):
```python
SEQ_LEN = 1024;  N_SEQS = 200;  BATCH_SIZE = 4;  SEED = 42
```

---

## Combined Activation

```python
class C3HybridActivation(nn.Module):
    def __init__(self, mu_neg: torch.Tensor):
        super().__init__()
        self.register_buffer('mu_neg', mu_neg)  # (3072,)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z < 0: constant mu_neg  (C3)
        # z >= 0: ReLU(z) = z    (Hybrid positive)
        shape = z.shape
        mu = self.mu_neg.expand_as(z)
        return torch.where(z < 0, mu, F.relu(z))
```

---

## Implementation Steps

1. Load float GPT-2, tokenizer, OWT data (N=200, SEQ_LEN=1024)
2. Baseline PPL
3. Calibrate mu_neg per block (same as C3 cycle 188):
   - Hook on mlp.act input z, collect GELU(z) where z<0, take per-channel mean
   - Result: mu_neg[b] shape (3072,)
4. Deepcopy model, replace all 12 blocks' act with C3HybridActivation(mu_neg[b])
5. Evaluate all-12 PPL
6. Also run progressive sweep (k=0..11) to compare shape with C3 and hybrid progressives

---

## Calibration Details (same as cycle 188)

```python
sum_neg = {b: torch.zeros(3072, dtype=torch.float64) for b in range(12)}
count_neg = {b: 0 for b in range(12)}
# Hook on act: where z < 0, collect GELU(z); accumulate sum and count
# mu_neg[b] = sum_neg[b] / count_neg[b]
```

---

## Verdict

Reference baseline = 22.700 (cycle 188 baseline):
```python
pct = (ppl_all12 - baseline) / baseline * 100
if pct > 98:  verdict = "CONFIRMED"
elif pct < 54: verdict = "REFUTED"
else:          verdict = "INCONCLUSIVE"
```

Note: the verdict thresholds are relative to the expectation of +48% (simple sum of C3+Hybrid).
Confirmed means ABOVE what ReLU+bias gives (~88%) — truly general superadditivity.
Refuted means below the expected sum (~48%) — no superadditivity without zero-point.
