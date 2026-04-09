# Cycle 332 Plan — Item 331 (revised per review)

## What
Train tiny GPT (4L 256d) on wikitext-2 with {GELU, ReLU², LeakyReLU²} × 3 seeds, then eff_rank census. Report mean±std. No curvature framework (dropped to fit time cap).

## Script
Create `experiments/tiny_gpt_act_compare_cycle331.py` (~250 lines, single file).

### Model (~80 lines)
nanoGPT-shaped: `n_layer=4, n_head=4, n_embd=256, n_inner=1024, vocab_size=50257, block_size=128`.
Pre-LN (LayerNorm before attn and MLP). No bias on nn.Linear. Causal mask via `torch.triu`.
MLP: `c_fc` (256→1024), activation, `c_proj` (1024→256). Attention: standard multi-head with `c_attn` (256→768) and `c_attn_proj` (256→256).
Final LN + `lm_head` (256→50257, tied with wte).
Activation passed as constructor arg: one of `{"gelu": F.gelu, "relu2": lambda z: F.relu(z).square(), "lrelu2": lambda z: F.leaky_relu(z, 0.5).square()}`.

### Data (~30 lines)
Load wikitext-2-raw-v1 train/val via HF `datasets`. Tokenize with `GPT2TokenizerFast.from_pretrained("gpt2")`.
Concatenate all tokens into a flat tensor, slice into (batch_size=32, seq_len=128) windows.
Val: 100 sequences for eval PPL (cross-entropy → exp).

### Training loop (~60 lines)
For each activation in `["gelu", "relu2", "lrelu2"]`, for each seed in `[42, 137, 2024]`:
1. `torch.manual_seed(seed)`, init model, move to MPS/CPU
2. AdamW: lr=3e-4, wd=0.01, betas=(0.9, 0.95)
3. Cosine LR schedule over 3000 steps (warmup 200 steps)
4. Train 3000 steps, log loss every 100, eval PPL every 500
5. **LR fallback**: if val PPL > 200 at step 2000, restart from scratch with lr=1e-4 (same seed, 3000 steps). Mark variant as "fallback-lr" in output.
6. Save best checkpoint to `experiments/tiny_gpt_act_compare_checkpoints/{variant}_s{seed}.pt`
7. Print final val PPL

### Eff_rank census (~40 lines)
For each saved checkpoint:
1. Load model, set eval mode
2. Hook each block's `mlp.c_proj` output (4 blocks)
3. Forward 200 calib sequences (from val or train)
4. Covariance `C = mlp_out.T @ mlp_out / N` (256×256), SVD, compute `eff_rank = (sum(s))^2 / sum(s^2)`
5. Report per-block eff_rank mean±std across seeds for each variant

### Output
- JSON: `autoresearch/results/cycle_332_eff_rank.json` — `{variant: {block: {mean, std, per_seed: [...]}}}`
- Console: 3×4 eff_rank table (mean±std), per-variant val PPL (mean±std), timing
- Checkpoints: `experiments/tiny_gpt_act_compare_checkpoints/{variant}_s{seed}.pt`

## Verdict logic
- **CONFIRMED-rank-profile-shape**: any pair of variants differs by ≥0.5 eff_rank (in mean) at ≥3 blocks AND std < 0.3 at those blocks AND all 3 converge (val PPL ≤ 200)
- **REFUTED-rank-profile-converges**: all 3 within ±0.3 mean at every block
- **INCONCLUSIVE**: otherwise (e.g. high variance, convergence failure)

## Sanity checks
- All 3 models must reach val PPL ≤ 200 (if any fails even with fallback lr, that variant is excluded)
- Eff_rank values must be > 0 and ≤ 256 (the embedding dimension)
- Training loss should decrease monotonically in the first 1000 steps

## Time budget
9 runs × 3000 steps ≈ 27 min on MPS. Eff_rank census ≈ 5 min. Total ≈ 32 min (within 45-min cap).

## Risks
- LR fallback adds up to 3000 extra steps per variant (unlikely for all seeds)
- Tiny model may not differentiate activations enough (4 layers, 256d is small)
