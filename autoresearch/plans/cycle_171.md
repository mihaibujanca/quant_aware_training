# Cycle 171 Plan — Item 181: Wider-Shallower Compression

**Script:** `experiments/wider_shallower_compression_cycle171.py`

## Pre-flights

**P1 — Null hypothesis:** Diffuse error (cycle 163) means stacking damage is proportional to
total compression, not block count. 12@3x will be WORSE because every residual-stream addition
gets a small error, and 12 small errors compound to more than 4 larger ones (c=0.41/block).

**P2 — Criteria (verbatim):**
- Confirmed: 12@3x PPL < 4@6x PPL (70.7) by >= 2 PPL points
- Refuted: 12@3x PPL >= 4@6x PPL
- Inconclusive: within 2 PPL

**P3 — Novelty:** Cycle 094 tested non-uniform allocation across 4 fixed blocks (REFUTED).
Cycle 089/093 tested stacking accumulation at fixed 6x. This is new: varying BOTH block count
AND per-block ratio simultaneously.

**P4 — Confounds:**
- Polish 500 steps may over/underfit at 3x vs 6x (more channels to tune). Mitigated: same
  config for fair comparison.
- Block selection for 8-block config (C): using [0..7] is arbitrary. Mitigated: uniform subset.
- Total compressed params differ across configs. Not param-matched, but tests the principle.

## Implementation

1. Load GPT-2 teacher, build OWT corpus (200 eval, 200 probe seqs, seq_len=128)
2. Probe geometry for all 12 blocks (reuse atlas across configs within seed)
3. For each seed (42, 137, 7):
   a. Baseline: 4 blocks [1,4,7,11], hidden=512 (6x), eval PPL
   b. Config A: 12 blocks [0..11], hidden=1024 (3x), eval PPL
   c. Config B: 12 blocks [0..11], hidden=1536 (2x), eval PPL
   d. Config C: 8 blocks [0,1,2,3,4,5,6,7], hidden=768 (4x), eval PPL
4. Polish: 500 Adam steps, lr=1e-4 on c_proj weights per block
5. Corrections test (r=4 uniform LoRA, 2000 steps) on configs A and baseline only
6. Save JSON results, print summary, verdict

## Output
- `autoresearch/results/cycle_171_wider_shallower.json`
- Summary table: config | mean_ppl | std | delta_vs_baseline

## Verdict logic
```
delta = mean_ppl_A - mean_ppl_baseline
if delta <= -2.0: CONFIRMED
elif delta >= 0.0: REFUTED
else: INCONCLUSIVE
```
