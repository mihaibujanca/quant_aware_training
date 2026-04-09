# Cycle 333 Plan — Item 332: LASER on tiny GPT activation variants

## What
Spectrum preflight on all blocks × 3 activations, then LASER rank sweep on the best-spectrum block per variant. Report whether LASER transfers across activations.

## Script
Create `experiments/tiny_gpt_laser_cycle333.py` (~300 lines, single file).

### Imports & Config (~20 lines)
Import model class from `experiments/tiny_gpt_act_compare_cycle331.py` (or copy the model class inline — the cycle 331 script has `N_LAYER`, `N_HEAD`, etc. at module level and the model as a class).

```python
VARIANTS = ["gelu", "relu2", "lrelu2"]
CKPT_DIR = "experiments/tiny_gpt_act_compare_checkpoints"
SEED_FOR_SPECTRUM = 42  # one seed suffices for weight SVD (weights are deterministic per seed)
RANKS = [1, 2, 5, 10]
N_STEPS = 500
LR = 1e-4
EVAL_EVERY = 50
SEEDS = [42, 137, 2024]
N_EVAL = 100  # eval sequences
BLOCK_SIZE = 128
SPECTRUM_GATE = 0.10  # top-5 energy ≥ 10%
```

### Step 1: Spectrum Preflight (~40 lines)
For each variant, load seed-42 checkpoint. For each of 4 blocks, SVD `block.mlp.c_proj.weight` (shape 256×1024):
1. Compute `U, S, Vh = torch.linalg.svd(W, full_matrices=False)` — S has 256 values
2. Report: `top_1_energy = S[0]^2 / sum(S^2)`, `top_5_energy = sum(S[:5]^2) / sum(S^2)`, `top_10_energy`
3. Print 3×4 table: variants × blocks → top-5 energy fraction
4. Per variant, select target block = block with highest top-5 energy fraction
5. **Gate**: if best block's top-5 energy < SPECTRUM_GATE for any variant → that variant gets verdict "REFUTED-flat-spectrum", skip retrain for it

### Step 2: Baseline PPL (~30 lines)
For each variant, for each seed: load checkpoint, eval PPL on 100 val sequences. Report baseline PPL per (variant, seed). This gives the "float baseline" to compare LASER against.

Eval: standard CE loss → exp(mean_loss). Use the same data loading as cycle 331 (wikitext-2 val, GPT2TokenizerFast, seq_len=128).

### Step 3: LowRankCProj for nn.Linear (~30 lines)
Adapt cycle 280's LowRankCProj for `nn.Linear` semantics:
```python
class LowRankCProj(nn.Module):
    """Rank-r replacement for nn.Linear c_proj (no bias).

    c_proj.weight shape: (out_features=256, in_features=1024)
    SVD: U(256,256), S(256,), Vh(256,1024)
    A = U[:, :r] * S[:r]  → (256, r), frozen
    B = Vh[:r, :]          → (r, 1024), trainable
    Forward: F.linear(x, A @ B) = x @ (A @ B).T
    """
    def __init__(self, A_frozen, B_init):
        super().__init__()
        self.register_buffer("A", A_frozen.contiguous())
        self.B = nn.Parameter(B_init.clone().contiguous())

    def forward(self, x):
        W = self.A @ self.B  # (256, 1024)
        return F.linear(x, W)
```

### Step 4: LASER Rank Sweep (~80 lines)

**Step 4.0: Full-rank sanity (MANDATORY)**. For GELU seed-42, construct LowRankCProj with r=256 (full rank). Eval PPL. Assert matches baseline within 0.05 PPL. If mismatch: there is a transposition bug in the nn.Linear adaptation — abort and debug. Do not proceed to rank sweep until this passes.

For each variant (that passed spectrum gate), for each rank in {1, 2, 5, 10}:

1. **No-retrain**: load seed-42 checkpoint, replace target block's c_proj with LowRankCProj(A_frozen, B_init), eval PPL. This gives the "SVD-only" baseline.

2. **Retrain**: for each seed in {42, 137, 2024}:
   - Load seed-42 checkpoint (always same base model — only B is retrained with different seeds)
   - Replace c_proj with LowRankCProj
   - Adam optimizer on `[B]` only (no bias — tiny model has no c_proj bias)
   - 500 steps, lr=1e-4, eval every 50 steps, best-state restore
   - Calib data: first 200 sequences from wikitext-2 val, batch size 4
   - Report best PPL per seed, mean±std across seeds

### Step 5: Output (~40 lines)
- JSON: `autoresearch/results/cycle_333_laser.json`
  - Spectrum table: `{variant: {block: {top1, top5, top10}}}`
  - Baseline PPL: `{variant: {seed: ppl}}`
  - No-retrain PPL: `{variant: {rank: ppl}}`
  - Retrain PPL: `{variant: {rank: {seed: best_ppl, mean, std}}}`
- Console: spectrum table, 3×4 retrain PPL table
- Manifest: `autoresearch/results/cycle_333_manifest.txt`

## Verdict Logic (per agenda item 332)
- **CONFIRMED-LASER-generic**: all 3 variants have at least one rank r where retrained PPL ≤ baseline + 0.5
- **REFUTED-LASER-GELU-specific**: only GELU variant matches baseline, ReLU²/lrelu² need r ≥ 100
- **REFUTED-flat-spectrum**: all 3 variants have flat spectrum (top-5 < 10%), LASER inapplicable
- **INCONCLUSIVE**: otherwise

Note: "baseline" here means the UNCOMPRESSED val PPL of that variant (not GPT-2's baseline). The question is: does LASER maintain PPL, not whether it improves it. On GPT-2, LASER at b1 r=2 actually IMPROVED PPL by 4.86 (regularization effect). On tiny GPT, improvement is unlikely but matching baseline is the bar.

## Sanity Checks
- Float baseline PPL should match cycle 332: GELU ~198, ReLU² ~185, lrelu² ~185 (±2 due to eval split)
- No-retrain PPL should be WORSE than baseline (SVD truncation loses info)
- Retrain PPL should improve over no-retrain (Adam recovers lost directions)
- Trainable params per seed: r × 1024 (B only). At r=2: 2048. At r=10: 10240.

## Risks
- Tiny model weight spectrum may be flat everywhere → all variants get REFUTED-flat-spectrum. This is still a valid result (LASER needs concentrated spectrum, tiny random-init models may not have it).
- 500 steps may be insufficient for the tiny model's higher PPL scale (~185-200 vs GPT-2's ~54). Could increase to 1000 if time permits.
- nn.Linear vs Conv1D transposition bug — the plan explicitly handles this, but verify with a no-retrain sanity: rank-256 (full rank) should recover exact baseline PPL.

## Time Budget
- Spectrum preflight: 3 variants × load + SVD ≈ 30 sec
- Baseline PPL: 3 variants × 3 seeds × eval ≈ 2 min
- LASER retrain: 3 variants × 4 ranks × 3 seeds × 500 steps ≈ 36 runs × 20 sec ≈ 12 min
- Total: ~15 min (within cap)
