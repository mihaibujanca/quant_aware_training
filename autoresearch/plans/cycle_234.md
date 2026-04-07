# Task

Cycle 234 / item 244 (P2). Build `experiments/gelu_noneg_lora_cycle234.py`: patch all 12 GPT-2 small MLP activations with cycle 228's `GeluNonegCorrected(μ_l)`, then add a LoRA r=8 adapter on each block's `mlp.c_proj` weight (W_out, HF Conv1D shape (3072, 768)) trained jointly via Adam on CE against WikiText-2 val (N=200, SEQ_LEN=128), LR=1e-4, 2000 steps, seed 42. Compare final PPL against cycle 228's 61.664 and float baseline 45.064.

# Hypothesis

Verbatim item 244:
> gelu_noneg + first-moment + LoRA r=8 W_out residual — eat the 16.6 PPL residual?
> Cycle 228 showed per-channel first-moment correction (μ_l) recovers ~34× of the cycle 220 gelu_noneg catastrophe but leaves +16.6 PPL residual (61.66 PPL vs 45.06 baseline). The audit attributed residual to higher-order curvature beyond the per-channel mean. Test: on top of the all-12-block `relu(GELU(z)) + μ_l` patch, add a LoRA r=8 adapter on each block's W_out (A ∈ R^{768×8}, B ∈ R^{8×3072}, initialized zero) trained jointly via Adam on CE against baseline logits, LR=1e-4, 2000 steps, seed 42 (1 seed for scouting). Compare final PPL against cycle 228's 61.664.
> - Confirmed if: final_ppl ≤ 48.0 PPL.
> - Refuted if: final_ppl ≥ 58.0.
> - Inconclusive if: final_ppl ∈ (48.0, 58.0).

Note: brief specifies LoRA on Conv1D layout `(3072, 768)` with A ∈ R^(3072, 8), B ∈ R^(8, 768), delta = A @ B (shape (3072, 768)), B init zero so init behavior matches cycle 228.

# Pre-flight Results

Clean. NEW combination of cycle 228 (gelu_noneg + μ_l patch, 61.664 PPL) and LoRA r=8 residual adapter. No prior file `experiments/gelu_noneg_lora_cycle234.py`. Dependencies (234, 238) done.

# Steps

1. Create `experiments/gelu_noneg_lora_cycle234.py`.
2. Reuse cycle 228 loader (`load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")`, N_EVAL=200, SEQ_LEN=128, seed 42, DEVICE mps-or-cpu) and `eval_ppl(model, eval_ids)`.
3. Reuse cycle 228's PHASE 1 μ_l precomputation loop verbatim: hook each `model.transformer.h[l].mlp.c_fc`, stash LN2 outputs, recompute preact = x @ c_fc.weight + c_fc.bias, gelu_l = F.gelu(preact, 'tanh'), noneg_l = relu(gelu_l), μ_l = (gelu_l - noneg_l).mean(0), to(DEVICE).
4. Patch all 12 blocks with distinct `GeluNonegCorrected(μ_dict[l])` instances (copy class from cycle 228).
5. LoRA wrapper. Use a Conv1D-compatible wrapper holding the original `c_proj` and two trainable params per block:
   - `class LoraConv1D(nn.Module)`: stores `base: Conv1D` (frozen), `A: Parameter (3072, 8)` init N(0, 0.01), `B: Parameter (8, 768)` init zero, `bias` reused from base.
   - forward: `x @ (base.weight + A @ B) + base.bias`. Matches HF Conv1D semantics (out = x @ W + b).
   - Freeze `base.weight` and `base.bias` via `requires_grad_(False)`.
6. Install: `for l in range(12): model.transformer.h[l].mlp.c_proj = LoraConv1D(model.transformer.h[l].mlp.c_proj).to(DEVICE)`. Then set `model.requires_grad_(False)` first, then re-enable `A`, `B` for each LoraConv1D (freeze everything except LoRA params).
7. Optimizer: `torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)`. Assert trainable count = 12 × (3072×8 + 8×768) = 368640.
8. Training: 2000 steps. Calibration set = the same 200 eval_ids. At each step, sample one seq (cycled) and compute `out = model(ids, labels=ids); loss = out.loss; loss.backward(); opt.step(); opt.zero_grad()`. Seed 42 for both torch and sampling order. EVAL_EVERY=200: run `eval_ppl` in no_grad, print step, loss, ppl. Track best_ppl + best_state (dict of CPU clones of {A, B} per block). Restore best_state at end.
9. Final eval with restored best state → `final_ppl`.
10. Verdict:
    - CONFIRMED if final_ppl ≤ 48.0
    - REFUTED if final_ppl ≥ 58.0
    - INCONCLUSIVE if 48.0 < final_ppl < 58.0
11. Save JSON `autoresearch/results/cycle_234_gelu_noneg_lora.json` with: cycle, item, verdict, config (N_EVAL, SEQ_LEN, SEED, DEVICE, LR, STEPS, RANK), baseline_ppl, cycle228_ref_ppl=61.664, float_ref_ppl=45.064, init_ppl (post-LoRA-zero init, should ≈ 61.664), final_ppl, best_step, ppl_history (list of (step, ppl)), delta_vs_cycle228, delta_vs_float, thresholds, trainable_params=368640.
12. Save summary MD `autoresearch/results/cycle_234_summary.md` with table (float | cycle 228 μ-only | cycle 234 μ+LoRA) and verdict paragraph.

# Validation

- Post-μ-patch, pre-training eval must equal cycle 228's 61.664 ± 0.01 (B init zero → LoRA delta zero → identical forward to cycle 228). If not, bug in LoraConv1D forward. Assert `abs(init_ppl - 61.664) < 0.5`.
- Trainable params assertion: `sum(p.numel() for p in model.parameters() if p.requires_grad) == 368640`.
- After training, the frozen base.weight for each block must be unchanged (snapshot a copy of `base.weight.data.clone()` at block 0 pre-train, compare post-train with `torch.equal`).
- ppl_history should be monotone-ish decreasing; if final > init, training diverged → fallback.

# Artifacts

- `experiments/gelu_noneg_lora_cycle234.py`
- `autoresearch/results/cycle_234_gelu_noneg_lora.json`
- `autoresearch/results/cycle_234_summary.md`
- `autoresearch/results/cycle_234_manifest.txt` (runner writes)

# Fallbacks

- If `init_ppl != 61.664`: LoraConv1D forward is wrong. Verify delta = A @ B is added to base.weight in the same einsum as HF Conv1D (`x.view(-1, nx) @ W + b`).
- If Adam diverges (loss → NaN or ppl increases >10% above init): lower LR to 5e-5 and retry; if still diverges, note and mark INCONCLUSIVE with diagnostic.
- If best_state restore yields different ppl from eval during training: bug in state snapshotting (ensure `.detach().clone()` and reload via `.data.copy_()`).

# Update bookkeeping

- Move item 244 from NEXT/P2 to PAST/Cycle 234 in `autoresearch/agenda.md` (update header block).
- Append TSV row to `autoresearch/results/cycles.tsv` with cycle=234, item=244, verdict, final_ppl, delta_vs_cycle228.
- Commit with message `[autoresearch] cycle 234: gelu_noneg + μ + LoRA r=8 W_out — <VERDICT>`.
- Update `autoresearch/STATUS.md` with one-line result.
