# Cycle 338 Review: GPT-2 full 12-block weight spectrum census (item 347)

## Checklist

| Check | Status | Notes |
|---|---|---|
| Correct model | PASS | `gpt2` from HuggingFace — same model as cycle 280 |
| All 12 blocks covered | PASS | Loop `range(12)`, each block's `mlp.c_proj.weight` |
| Conv1D weight shape handled | PASS | Shape (3072, 768) — energy fractions are transpose-invariant |
| SVD correctness | PASS | `torch.linalg.svd(W.float(), full_matrices=False)` — cast to float32 |
| Top-k energy computation | PASS | `sum(S[:k]^2) / sum(S^2)` — standard energy fraction |
| Verdict thresholds reasonable | PASS | median top-5 >= 8% = CONFIRMED; only b1 >= 8% = REFUTED |
| Cross-check with cycle 280 | PASS | Script should reproduce b1 top-1=4.59%, top-5=11.67% |
| pgolf comparison | PASS | Summary includes pgolf numbers from cycle 337 |
| Output files specified | PASS | JSON, summary.md, manifest.txt — standard pattern |
| Time budget | PASS | 12 SVDs on 768x3072 matrices + model load — trivial, <30s total |

## Issues

None. Straightforward census matching the cycle 337 pattern, adapted for HuggingFace GPT-2.

## Decision
APPROVE
