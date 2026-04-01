# Cycle 184 — Item 197: Residual Stream Trajectory Divergence: GELU vs ReLU+bias

## Task

Measure how and where the residual stream diverges across GPT-2's 12 blocks when GELU is replaced by ReLU+bias. Five sub-analyses: per-layer divergence norm, SVD rank of divergence, growth rate fit, transition-zone concentration, and alpha interpolation.

## Hypothesis

The +63% PPL failure of ReLU+iterative bias (cycle 182) is caused by GELU vs ReLU shape mismatch near z=0 that compounds exponentially through layers and concentrates in a low-rank subspace (rank <= 32). Alpha interpolation between GELU and ReLU should degrade smoothly if shape is the driver.

## Pre-flight Results

- **P1 Duplicate**: No prior trajectory divergence analysis exists. Cycles 181-182 measured PPL only, not per-layer residual divergence.
- **P2 Dependencies**: Item 192 (iterative bias) is in PAST (INCONCLUSIVE, cycle 182). Cycle 182 experiment file at `experiments/iterative_relu_bias_swap_cycle182.py` provides the iterative bias implementation pattern.
- **P3 Concrete**: 5 specific sub-analyses with quantitative criteria.
- **P5 Null baseline**: If GELU approx ReLU everywhere, divergence norms would be near zero at all layers.
- **P6 Model**: No saved checkpoint needed. Iterative bias swap is recomputed inline (sequential block-by-block, same as cycle 182).

## Criteria (verbatim from agenda)

- Confirmed if: divergence grows exponentially (R² > 0.8 for exp fit) AND divergence rank ≤ 32 (concentrated) AND α interpolation is smooth.
- Refuted if: divergence is diffuse (rank > 256) or α has a sharp transition.
- Inconclusive if: growth is linear or rank varies wildly by layer.

## Steps

### Step 1: Write `experiments/trajectory_divergence_cycle184.py`

Config: DEVICE=mps/cuda/cpu, N_SEQS=16, SEQ_LEN=128 (=2048 tokens), SEED=42, OWT_PATH same as cycle 182. Reuse `load_owt_tokens`, `eval_ppl`, `compute_mu_neg_single`, `ReLUActivation` patterns from cycle 182.

**Three models** (all from fresh `GPT2LMHeadModel.from_pretrained("gpt2")`):
- (a) Float GELU — unmodified
- (b) ReLU+iterative bias — sequential swap: for l in 0..11, compute mu_neg on current state, replace `.mlp.act` with ReLU, add mu to `c_proj.bias`
- (c) ReLU no bias — replace all `.mlp.act` with ReLU, no bias correction

**Capture residual streams**: Register forward hooks on each `model.transformer.h[l]` block and capture the block's OUTPUT tensor (return value). In GPT-2, this is the residual stream after adding the MLP output, i.e. `x + attn(x) + mlp(ln2(x))` — the full residual connection output of block l. For each model, run the 2048 tokens, collect `r_l` at each of 12 block boundaries. Store as `(12, N_tokens, 768)` tensor.

**Analysis 1 — Per-layer divergence**:
```
div_norm_l = ||r_gelu_l - r_relu+bias_l|| / ||r_gelu_l||  (per-token, then mean)
```
Also compute for ReLU-no-bias vs GELU (sanity: should be worse).

**Analysis 2 — SVD rank**:
```
D_l = r_gelu_l - r_relu+bias_l   # (N_tokens, 768)
U, S, V = torch.linalg.svd(D_l, full_matrices=False)
rank_90 = (cumsum(S**2) / sum(S**2) < 0.9).sum() + 1
rank_99 = (cumsum(S**2) / sum(S**2) < 0.99).sum() + 1
```

**Analysis 3 — Growth rate**:
```
mean_div_l = ||D_l||_F / sqrt(N_tokens)
Fit: log(mean_div_l) = a*l + b  (exponential) and mean_div_l = a*l + b (linear)
Report R² for both fits.
```

**Analysis 4 — Transition zone**:
For each block l, register a forward hook on `model.transformer.h[l].mlp.c_fc` and capture its OUTPUT (this is z, the pre-activation value: z = W_fc @ LN(x) + b_fc, i.e. post-linear, pre-GELU). Run on the GELU model. For the top-50 most-divergent tokens (by ||r_gelu - r_relu+bias||) and bottom-50: compute fraction of channels with |z| < 1. Report mean fraction for top-50 vs bottom-50.

**Analysis 5 — Alpha interpolation**:
Create `AlphaActivation(alpha)` module: `forward(z) = alpha*gelu(z) + (1-alpha)*relu(z)`.
For each alpha in {0, 0.25, 0.5, 0.75, 1.0}: fresh model, iterative bias swap using AlphaActivation, eval PPL on full N_SEQS=16 sequences. Use 500 tokens for mu_neg estimation to save time.

**Alpha bias formula** (single canonical definition): For GELU_alpha(z) = alpha*gelu(z) + (1-alpha)*relu(z), the per-block bias correction is computed iteratively (same structure as cycle 182):
```
For l in 0..11:
    z_l = W_fc @ h_l + b_fc          # pre-activation (output of c_fc linear layer)
    h_l = block input AFTER all previous blocks have been swapped
    bias_l = E_x[gelu(z_l)] - E_x[gelu_alpha(z_l)]   # mean output bias correction
    c_proj.bias += c_proj.weight @ bias_l   # apply correction through down-projection
    Swap block l's activation to AlphaActivation(alpha)
```
Where `gelu(z_l)` is the original GELU output and `gelu_alpha(z_l) = alpha*gelu(z_l) + (1-alpha)*relu(z_l)`. This yields: bias_l = 0 at alpha=1 (AlphaActivation IS GELU, no correction needed) and bias_l = E[gelu(z_l) - relu(z_l)] at alpha=0 (standard mu_neg from cycle 182). The expectation is estimated over 500 tokens on the current (partially-swapped) model state.

**Outputs**:
- JSON: `autoresearch/results/cycle_184_trajectory_divergence.json` with all numeric results
- Plot: `autoresearch/results/cycle_184_trajectory_divergence.png` — 2x3 grid:
  - (0,0) Per-layer relative divergence (both ReLU+bias and ReLU-no-bias vs GELU)
  - (0,1) SVD rank_90 and rank_99 per layer
  - (0,2) Growth rate fit (log scale, exp and linear fits overlaid)
  - (1,0) Transition zone fraction: top-50 vs bottom-50
  - (1,1) Alpha interpolation PPL curve
  - (1,2) Singular value spectra for layers {0, 3, 6, 11}
- Print summary table and verdict.

### Step 2: Run

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/trajectory_divergence_cycle184.py
```

Expected runtime: ~10-15 min on MPS (3 model loads, 2048-token forward passes, alpha sweep with 5 configs).

### Step 3: Review and write results manifest

Write `autoresearch/results/cycle_184_manifest.txt` with verdict, key numbers, artifact paths.

### Step 4: Update agenda.md

Move item 197 from NEXT/P0 to PAST under `### Cycle 184` with status and one-line summary.

### Step 5: Commit

```bash
git add experiments/trajectory_divergence_cycle184.py autoresearch/results/cycle_184_* autoresearch/plans/cycle_184.md autoresearch/briefs/cycle_184.md autoresearch/agenda.md
git commit -m "[autoresearch] cycle 184: residual stream trajectory divergence GELU vs ReLU+bias — <VERDICT>"
```

## Validation

- Baseline GELU PPL must be in [22, 26] (GPT-2 on OWT with 16 seqs). If outside, data loading bug.
- ReLU+iterative bias PPL should be near 37 (cycle 182 measured 37.0 with 200 seqs — 16 seqs will have higher variance, accept 30-50).
- Divergence at layer 0 should be small (first block error only), grow through layers.
- Alpha=1.0 (pure GELU with zero bias correction) PPL must equal baseline. Alpha=0.0 must match ReLU+iterative bias.
- SVD rank sanity: rank_90 must be <= 768 (trivially true); if rank_90 > 256 at any layer, that's the REFUTED signal.

## Artifacts

- `experiments/trajectory_divergence_cycle184.py`
- `autoresearch/results/cycle_184_trajectory_divergence.json`
- `autoresearch/results/cycle_184_trajectory_divergence.png`
- `autoresearch/results/cycle_184_manifest.txt`

## Fallbacks

- If MPS OOM on 3 simultaneous model copies: run models sequentially, store residuals to disk between runs (each model's residuals: 12*2048*768*4 bytes = 72 MB).
- If alpha sweep too slow: reduce to alpha in {0, 0.5, 1.0} (3 instead of 5 configs). Still sufficient to detect sharp vs smooth.
- If transition zone hook interferes with residual capture: run analysis 4 in a second pass with only the GELU model (z values come from the original model's MLP inputs).
