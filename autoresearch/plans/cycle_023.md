# Cycle 023 Plan — Local Complexity as Compression Difficulty Predictor

## Pre-flight Checks
- P1: `.venv/bin/python` exists and has torch — verified by run command prefix
- P2: `autoresearch/results/` directory exists — write results there
- P3: cycle 010 r values available in `autoresearch/results/cycle_010_single_affine.json`
- P4: single output file `experiments/local_complexity_toy.py`, results JSON + PNG

## Architecture & Data
- Dataset: spirals, n=2000, n_turns=3, scale=2.0 (matches cycle 010)
- Network: 2→64→64→64→64→1, ReLU, lr=0.001 (matches cycle 010 exactly)
- Probe set: N=1000 points drawn from spirals data (reuse training set, subsample)
- ε = 0.39 (targets ~30 neighbors per ball in 2D input space ~[-2,2]²)

## Algorithm

### Training
- Train to epoch 2000 with Adam lr=0.001
- Save checkpoints at epochs 100, 500, 2000 (state_dict)

### Local Complexity Computation (per checkpoint, per layer)
For each layer l in {1,2,3,4}:
1. Forward-pass all N probe points through layers 1..l, collecting binary activation patterns
   (pattern[i] = (pre_relu_activation[i] > 0).int() as a 64-bit tuple or hash)
2. For each probe x_i, find all x_j with ||x_i - x_j||_2 ≤ ε in 2D INPUT space
   (precompute pairwise distances once — 1000×1000 float32 ≈ 4MB, fine)
3. Count distinct patterns among neighbors of x_i → local_count[i]
4. local_complexity[l] = mean(local_count) over all i

### Correlation with Cycle 010 r Values
- r_values = [0.424, 0.047, 0.031, 0.022] (layers 1–4 from cycle 010 r_held)
- Compute Spearman correlation between [lc_l1, lc_l2, lc_l3, lc_l4] and r_values
- Only 4 points — report p-value too, note low power

### Training Dynamics
- Compute local complexity (all layers) at each checkpoint
- Report mean local complexity over layers at each checkpoint
- Check: (early - late) / early ≥ 0.30

## Output Files
- `autoresearch/results/cycle_023_local_complexity.json` — all metrics
- `autoresearch/results/cycle_023_local_complexity.png` — 2-panel: (1) lc vs layer per
  checkpoint, (2) scatter lc vs r with Spearman annotation

## Execution
```
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/local_complexity_toy.py
```
