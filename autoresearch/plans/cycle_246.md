# Cycle 246 Plan — item 256: gelu_noneg + μ + LoRA r=8 + K=128 W_out on [1,4,7]

# Hypothesis
> "gelu_noneg + μ + LoRA r=8 + K=128 W_out compression — does the bump-removed
> activation absorb channel compression?" Combining cycle 239's recipe
> (gelu_noneg + μ_l + LoRA r=8 on c_proj, all 12 blocks, 47.692 ± 0.143) with
> cycle 223's K=128 W_out topk+lstsq on BLOCKS=[1,4,7] (48.989): the bump-removed
> block output may concentrate signal into fewer channels, so K=128 topk is a
> tighter fit AND the LoRA residual absorbs what lstsq cannot. Tests whether
> activation fix and channel compression are orthogonal or trade.

# Method
- Base: GPT-2 small, WikiText-2, disjoint calib/eval splits (cycle 239 loader).
- Phase 1: precompute μ_l on CALIB for all 12 blocks (cycle 228/239 recipe).
- Phase 2: patch all 12 `mlp.act` with `GeluNonegCorrected(μ_l)`.
- Phase 3: for BLOCKS=[1,4,7], run K=128 topk+lstsq against the current
  (gelu_noneg+μ corrected) block I/O; replace `mlp` with a `CompressedMLP`
  variant that carries `W_fc[:,topk], b_fc[topk], W_out (128,768), b_out (768)`
  and a built-in LoRA residual (A ∈ R^(128,8), B ∈ R^(8,768)) added to W_out.
- Phase 4: install LoraConv1D (cycle 239 wrapper) on the c_proj of the other
  9 float blocks (A ∈ R^(3072,8), B ∈ R^(8,768), A zero-init).
- Phase 5: joint Adam LR=1e-4 over {LoRA A/B × 12 blocks, W_out + b_out on [1,4,7]}.
  N_STEPS=2000, EVAL_EVERY=200, best-state snapshot + restore.
- Seeds: 42 first (P0 scout). Add 137, 2024 if wall-clock budget allows.

# Install order (critical — LoRA on compressed W_out, not float)
1. gelu_noneg + μ patch (12 blocks).
2. K=128 topk+lstsq fit on [1,4,7] against gelu_noneg-corrected I/O → replace mlp.
3. Install LoRA r=8 on the remaining 9 float blocks' `mlp.c_proj`.
4. CompressedMLP carries its own (A,B) LoRA for its 128→768 projection.
5. Collect trainable params → Adam.

# Hypothesis predictions
- Confirmed: held-out best_mean ≤ 47.0 (beats cycle 239 47.692 AND cycle 223 48.989).
- Refuted: best_mean ≥ 49.5.
- Inconclusive: 47.0 < best_mean < 49.5.

# Expected artifacts
- `experiments/gelu_noneg_lora_k128_cycle246.py`
- `autoresearch/results/cycle_246_gelu_noneg_lora_k128.json` (per-seed init/best/final, param counts)
- `autoresearch/results/cycle_246_summary.md`
- `autoresearch/results/cycle_246_manifest.txt` (written by execute)

# Sanity checks / validation
- μ ≤ 0 elementwise across 12 blocks (max ≤ 1e-6).
- LoRA-zero init eval PPL should be the K=128-lstsq-init floor on [1,4,7] plus
  gelu_noneg+μ on the other 9; expected roughly 50–65 PPL.
- Trainable param count asserted ≈ 595,200:
  - 9 float blocks LoRA: 9 × (3072×8 + 8×768) = 276,480
  - 3 compressed blocks LoRA: 3 × (128×8 + 8×768) = 21,504
  - 3 compressed W_out + b_out: 3 × (128×768 + 768) = 297,216
  - Total = 595,200.
- Disjoint calib/eval token sets asserted (cycle 239 assertion).

# Fallbacks
- If single-seed budget only: report 1 seed and flag as scout; decide confirm/refute
  only if |best − 47.0| > 1.5 or ≥ 49.5.
- If CompressedMLP-internal LoRA is clunky to plumb: drop LoRA on the 3 compressed
  blocks (train only W_out + b_out there), keep LoRA on the 9 float blocks. Flag
  this as a reduced-surface fallback in the summary; trainable count becomes
  276,480 + 297,216 = 573,696.
- If gelu_noneg+μ patch on [1,4,7] perturbs the topk scoring substantially, still
  run topk on the patched activations (not raw GELU) — that is the correct target
  subspace given the install order.

# Bookkeeping
- Move item 256 from `agenda.md` NEXT/P0 → PAST/Cycle 246.
- Add TSV row to `autoresearch/cycles.tsv`.
- Update STATUS.md and program.md cycle pointer.
- Commit plan + bookkeeping together.
