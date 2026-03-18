# Cycle 021 Plan — Grokking as Compression Catalyst: Toy ReLU

## Pre-flight
- P1: `.venv/bin/python` available (confirmed by MEMORY.md convention)
- P2: `from aleph.datasets import make_spirals` available
- P3: `aleph/relu_geometric_compression.py` exports `build_region_atlas`, `coarsen_region_atlas`
- P4: `aleph/models.py` exports `MLP`

## File
`experiments/grokking_compression_catalyst.py`

## Config
```python
N_TRAIN = 200       # small dataset → enables grokking
N_TEST  = 2000      # large test set for stable metrics
TURNS   = 3
HIDDEN  = 32
DEPTH   = 4         # 4 hidden layers: 2→32→32→32→32→1
EPOCHS  = 10000
CKPT_EVERY = 500    # 20 checkpoints
LR     = 0.001
WD     = 0.01       # weight decay → essential for grokking
SEED   = 42
```

## Algorithm

### Training loop with checkpoints
- Train MLP(2, [32,32,32,32], 1) with BCEWithLogitsLoss, Adam(lr=LR, weight_decay=WD)
- At every CKPT_EVERY epoch: save model state_dict in memory list
- Record train_acc, test_acc at each checkpoint

### Per-checkpoint geometric signals
For each saved checkpoint:
1. **Pattern count**: forward pass on X_test, collect hidden masks from each ReLU layer,
   concatenate into binary code string, count distinct codes → N_patterns
2. **Pattern entropy**: compute frequency distribution of patterns,
   H = -sum(p * log2(p)) where p are frequencies
3. **Single-affine residual**: fit one affine map (least squares) to all
   (X_test, logit) pairs: r = ||f(x) - (Ax+b)||² / ||f(x)||² per layer

### Detect phase transition
- Smooth N_patterns with window=3, find epoch where drop ≥ 10% from running max
- If no single large drop: use the checkpoint with the steepest decline slope
- Define: pre_grokking_idx = last checkpoint before drop, post_grokking_idx = first after

### Compression comparison
Load pre- and post-grokking checkpoints. For each:
- Run `build_region_atlas(model, X_test_tensor, cfg)` with minimal config
- Run `coarsen_region_atlas(atlas, target_n_regions=16)` → compressed model
- Evaluate compressed model accuracy on X_test
- Record: n_patterns, atlas_size, compressed_acc, original_acc

### Output
JSON: `autoresearch/results/cycle_021_grokking.json`
PNG:  `autoresearch/results/cycle_021_grokking_trajectory.png` (4-panel:
       train_acc, test_acc, n_patterns, single_affine_residual vs epoch)

## Verdict Logic
```python
drop_frac = (pre_patterns - post_patterns) / pre_patterns
acc_improvement = post_compressed_acc - pre_compressed_acc
if drop_frac >= 0.40 and acc_improvement >= 0.10:
    verdict = "confirmed"
elif drop_frac < 0.10 or abs(acc_improvement) < 0.02:
    verdict = "refuted"
else:
    verdict = "inconclusive"
```

## Notes
- Use `torch.no_grad()` for all checkpoint evaluations
- If grokking is absent (N_train too large or WD too small), report as refuted
- coarsen_region_atlas may fail if region atlas is too small — catch and report
