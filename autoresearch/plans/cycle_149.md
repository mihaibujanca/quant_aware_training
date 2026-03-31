# Plan — Cycle 149 (Item 159)

## Goal
Test whether training all 4 compressed FFN blocks jointly under the stacked model's CE loss breaks the composition barrier (cycle 138: independent PPL 74 vs topk 71).

## Method
Fork structure from `experiments/ae_stacking_k4_cycle138.py` into `experiments/joint_stacking_training_cycle149.py`.

### Compression architecture
Linear factorization (no ReLU, per cycle 148 finding). For each block, replace `c_proj` with `W_enc @ W_dec`:
- `W_enc`: Linear(3072, 64, bias=False) — maps GELU-activated hidden to bottleneck
- `W_dec`: Linear(64, 768, bias=True) — maps bottleneck to output (bias copied from original c_proj)
- `c_fc` frozen throughout (only compressing the output projection)
- Param count per block: 3072×64 + 64×768 + 768 = 246,528. Total: ~986K for 4 blocks.

### SVD initialization (conditions a)
For each block's `c_proj.weight` (shape 768×3072 in GPT-2 Conv1D = 3072×768 in standard):
1. `U, S, Vt = torch.linalg.svd(W_cproj, full_matrices=False)`
2. `W_enc.weight = (Vt[:64, :])^T` — shape (3072, 64)
3. `W_dec.weight = (U[:, :64] @ diag(S[:64]))^T` — shape (64, 768)
4. Verify: `W_dec(W_enc(h))` ≈ `c_proj(h)` for probe data (Frobenius check).

### Random initialization (condition b)
Standard PyTorch init (Kaiming uniform for encoder, same for decoder). Bias from original c_proj.

### Conditions
| Condition | Init | Steps | Trainable params |
|---|---|---|---|
| (a) joint-SVD | SVD of c_proj | 2000 | all 4 blocks' W_enc + W_dec |
| (b) joint-random | random | 2000 | all 4 blocks' W_enc + W_dec |
| (c) topk-baseline | topk+lstsq | 500 | c_proj weights of 4 blocks |

### Training
- All conditions train under stacked CE loss (all 4 blocks swapped simultaneously).
- AdamW lr=1e-4, cosine schedule to lr=0 over training steps.
- Batch size 4, seq len 128, probe_holdout data.
- Condition (c): standard topk+lstsq+polish recipe (500 steps, Adam, no schedule — matching historical protocol exactly).

### Factorization module
```python
class LinearFactorFFN(nn.Module):
    def __init__(self, original_mlp, d_bn, init_mode="svd"):
        # c_fc + act frozen, encoder + decoder replace c_proj
        # init_mode: "svd" or "random"
```

### Eval
- `_evaluate_model()` on 200 OWT eval sequences after training.
- Also evaluate at steps [500, 1000, 1500, 2000] for conditions (a) and (b) to get learning curves.
- Report: PPL per condition, PPL at intermediate steps, loss curves.

## Verdict criteria (from agenda)
- **Confirmed** if joint-SVD stacking PPL beats topk+lstsq+polish (70.0) by >=2% relative (i.e., PPL <= 68.6).
- **Refuted** if joint-SVD stacking PPL is within 2% of topk+lstsq+polish or worse (i.e., PPL >= 68.6).
- **Inconclusive** if better by 2-5% but seed variance unknown (single seed screening).

## Output
- `autoresearch/results/cycle_149_joint_stacking.json` — PPL per condition, intermediate checkpoints, loss curves
- stdout summary table
- Learning curve plot saved to `autoresearch/results/cycle_149_learning_curves.png`

## Verification
- Float PPL should match prior cycles (~57.65).
- Condition (c) topk baseline PPL should match cycle 138 topk post-polish (~70-71). If it doesn't, something is wrong with data or eval.
- SVD init condition (a) at step 0 should give a valid PPL (SVD captures most of c_proj energy). If PPL at step 0 is catastrophic (>200), SVD init failed — check Conv1D weight transposition.

## Runtime
~45 min. Condition (a): 2000 steps × forward+backward through full GPT-2 with 4 swapped blocks. Condition (b): same. Condition (c): topk fitting (~2 min) + 500 polish steps. 4 intermediate evals for (a) and (b) add ~8 min.

## Key risk
The topk+polish baseline trains c_proj weights that are FULL 3072x768 matrices — much more expressive than the rank-64 factorization. Condition (a) has ~986K params while condition (c) has ~9.4M params. If joint training loses, it could be the rank constraint, not the training protocol. Mitigation: the SVD init starts from the best rank-64 approximation of the ORIGINAL c_proj (not the lstsq-fitted one), so it begins closer to the float model. The question is whether 2000 CE steps from this init can compensate for the rank limitation.
