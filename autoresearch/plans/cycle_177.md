# Cycle 177 Plan — SoLA-style split compression (Item 187)

## Pre-flights
- P1: OWT data file exists at `data/openwebtext_sample.txt`
- P2: GPT-2 loads and baseline PPL in [25, 65]
- P3: probe_ffn_geometry returns atlas with hidden_sample (3072-dim) for all 4 blocks
- P4: torch.linalg.svd works on hidden_sample[:, tail] matrices

## Design

### Core idea
Standard topk+lstsq: select 512 top channels, fit lstsq from h_topk → output. Drops all tail info.
Split: select 512 top channels + SVD-compress tail (2560 channels) to rank r. Fit lstsq from [h_topk, z_tail_r] → output. Preserves partial tail information.

### Per-block compression steps
1. Probe geometry to get hidden_sample (N x 3072), output_sample (N x 768)
2. Select top-k=512 channels by activation importance (standard topk)
3. h_topk = hidden_sample[:, topk] (N x 512)
4. h_tail = hidden_sample[:, tail] (N x 2560)
5. SVD of h_tail: U, S, Vh = torch.linalg.svd(h_tail, full_matrices=False)
6. Keep top-r components: V_r = Vh[:r, :].T (2560 x r projection matrix)
7. z_tail = h_tail @ V_r (N x r compressed tail features)
8. Combined features: [h_topk, z_tail] (N x (512+r))
9. lstsq: fit c_proj_combined (768 x (512+r)) from combined → output_sample

### Inference module: SplitCompressedMLP
- Stores: full c_fc (3072x768), c_fc_bias, V_r (2560xr), topk_indices (512), c_proj_combined (768x(512+r)), c_proj_bias
- Forward: z = c_fc @ x + b → h = GELU(z) → h_top = h[topk], z_tail = h[~topk] @ V_r → y = c_proj @ cat(h_top, z_tail)

### Parameter accounting
- Split per block: c_fc=3072*768 + c_proj=768*(512+r) + V_r=2560*r + biases
  - r=2: 2,359,296 + 394,752 + 5,120 = 2,759,168
  - r=16: 2,359,296 + 405,504 + 40,960 = 2,805,760
- Standard topk+lstsq: c_fc=512*768 + c_proj=768*512 = 787,200

Split uses ~3.5x more params due to full c_fc. For FAIR comparison:
- Also run topk+lstsq with k chosen to match split's total params (k ≈ 1800 channels, effectively no compression)
- Primary comparison: split vs topk+lstsq at SAME topk=512 (split has more total params but uses them for tail info)

### Sweep
- r in {2, 4, 8, 16} for SVD rank of tail activations
- Baseline: standard topk+lstsq (r=0, same as current pipeline)
- Each config: unpolished + 500 Adam polish

### Verdict logic
- Compare mean polished PPL across 3 seeds
- CONFIRMED: best-r PPL <= 68.7 (>= 2 PPL better than topk+lstsq 70.7)
- REFUTED: all r configs PPL >= 72.7
- INCONCLUSIVE: best-r within +/-2 of 70.7

### Output
- `experiments/split_compression_cycle177.py`
- `autoresearch/results/cycle_177_split_compression.json`
- `autoresearch/results/cycle_177_split_compression.png`
