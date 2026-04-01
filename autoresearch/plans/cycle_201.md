# Cycle 201 Implementation Plan: Block 11 Geometry Deep-dive

## Overview

Analyze block 11's anomalously high delta_norm (335 vs median 30) and borderline droppability
using three sub-analyses: (1) output rank, (2) token-type delta breakdown, (3) delta direction.

**Script**: `experiments/block11_geometry_cycle201.py`
**Output**: `autoresearch/results/cycle_201_block11_geometry.json`

---

## Config

```python
N_SEQS = 50
SEQ_LEN = 64
SEED = 42
ENERGY_THRESH = 0.90
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FOCUS_BLOCKS = [3, 7, 10, 11]  # 3,7 for comparison; 10,11 for anomaly analysis
OUTPUT_PATH = "autoresearch/results/cycle_201_block11_geometry.json"
```

---

## Sub-analysis 1: Input and Output Rank

Use block forward hooks to capture inp[0] (residual input) and out[0] (residual output).

```python
block_in = {b: [] for b in FOCUS_BLOCKS}
block_out = {b: [] for b in FOCUS_BLOCKS}

for b in FOCUS_BLOCKS:
    def mk_hook(bb):
        def fn(m, inp, out):
            block_in[bb].append(inp[0].detach().float().cpu().reshape(-1, 768))
            out_t = out[0] if isinstance(out, tuple) else out
            block_out[bb].append(out_t.detach().float().cpu().reshape(-1, 768))
        return fn
    hooks.append(model.transformer.h[b].register_forward_hook(mk_hook(b)))
```

Compute:
- `input_rank_90`: effective_rank of centered block_in
- `output_rank_90`: effective_rank of centered block_out
- `delta = block_out - block_in`: per-token residual change
- `delta_rank_90`: effective_rank of centered delta matrix

---

## Sub-analysis 2: Token-type Breakdown

For each token position, record: delta_norm, token_id.

```python
# Collect token IDs alongside delta norms
token_ids_flat = []  # (N_tokens,)
delta_norms_flat = []  # (N_tokens,)

for seq_ids in calib_ids:
    token_ids_flat.extend(seq_ids[0].tolist())

# Group by token type: top-100 most frequent tokens
from collections import Counter
token_counts = Counter(token_ids_flat)
top_tokens = [t for t, c in token_counts.most_common(100)]

# Mean delta norm per token type
delta_by_token = {}
for tok in top_tokens:
    mask = [i for i, t in enumerate(token_ids_flat) if t == tok]
    if mask:
        delta_by_token[tok] = float(block11_delta_norms[mask].mean().item())
```

Also: mean delta norm by position within sequence (position 0-63).

```python
delta_by_position = []
N_per_pos = N_SEQS
for pos in range(SEQ_LEN):
    pos_norms = block11_delta_norms[pos::SEQ_LEN]  # every SEQ_LEN-th element
    delta_by_position.append(float(pos_norms.mean().item()))
```

---

## Sub-analysis 3: Delta Direction Coherence

```python
delta_11 = block_out[11] - block_in[11]  # (N_tokens, 768)
delta_centered = delta_11 - delta_11.mean(0)
S_delta = torch.linalg.svdvals(delta_centered)
delta_rank = effective_rank(delta_centered)

# Top singular vector alignment: cos(delta_i, v_1) for each token
U, S, Vh = torch.linalg.svd(delta_centered, full_matrices=False)
# U: (N, min(N,768)), Vh: (min(N,768), 768)
v1 = Vh[0]  # top right singular vector: (768,)
cos_v1 = F.cosine_similarity(delta_11, v1.unsqueeze(0).expand_as(delta_11), dim=1)
mean_cos_v1 = float(cos_v1.abs().mean().item())

# Compare to other blocks
```

---

## Verdict Logic

```python
# Confirmed if: block 11 has anomalous output rank OR concentrated token-type delta
# Specifically:
b11_output_rank = output_rank_90[11]
median_other_output_rank = median([output_rank_90[b] for b in [3, 7, 10]])

# Check 1: output rank anomaly
output_rank_anomaly = b11_output_rank > 3 * median_other_output_rank

# Check 2: delta concentration by token type
# Top-20 token delta mean >> all-token delta mean (>2x)
top20_deltas = sorted(delta_by_token.values(), reverse=True)[:20]
top20_mean = sum(top20_deltas) / 20
overall_mean_delta = float(block11_delta_norms.mean().item())
delta_token_concentration = top20_mean > 2 * overall_mean_delta

# Check 3: delta direction coherence
delta_dir_coherent = mean_cos_v1 > 0.5  # most tokens' deltas align with top direction

outlier_count = sum([output_rank_anomaly, delta_token_concentration, delta_dir_coherent])
if outlier_count >= 2:
    verdict = "CONFIRMED"
elif outlier_count == 0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
```

---

## Output JSON

```json
{
  "cycle": 201, "item": 211,
  "verdict": "...",
  "per_focus_block": [
    {"block": 3, "input_rank_90": ..., "output_rank_90": ..., "delta_rank_90": ..., "mean_delta_norm": ...},
    {"block": 7, ...},
    {"block": 10, ...},
    {"block": 11, ...}
  ],
  "block11_token_analysis": {
    "delta_by_position": [...],  # 64 values
    "top20_token_mean_delta": ...,
    "overall_mean_delta": ...,
    "top10_tokens_by_delta": [{"token_id": ..., "token": "...", "mean_delta_norm": ...}]
  },
  "block11_delta_direction": {
    "delta_rank_90": ...,
    "mean_cos_top_sv": ...
  },
  "outlier_flags": {...},
  "config": {...}
}
```
