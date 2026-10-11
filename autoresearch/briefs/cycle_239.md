# Item
249 (P0): Cycle 234 rerun with disjoint calib/eval split — is the LoRA residual real?
Re-execute cycle 234 with `calib_ids = all_ids[:200]` and `eval_ids = all_ids[200:300]` (disjoint splits, like cycles 206/210/223/225/230). All training, μ_l, LoRA Adam loop on calib only; eval and best-state selection on held-out only.

# Context
- Cycle 234 (item 244, INCONCLUSIVE — audit-corrected from CONFIRMED): gelu_noneg + μ_l + LoRA r=8 → 32.411 PPL on the SAME 200 wikitext-2 val sequences used for training. DATA LEAKAGE.
- This cycle fixes the leakage and re-tests.
- Cycle 234 audit recommended this exact follow-up.

# Dependencies
None (cycle 228 μ recipe is on disk; cycle 234 has the LoRA scaffolding to copy from).

# Key Files
- experiments/gelu_noneg_lora_cycle234.py — has the LoRA wrapper, μ precomputation, training loop. Just need to swap the data split.
- autoresearch/audits/cycle_234.md — audit note explaining the bug.

# Constraints
- BLOCKS: all 12 (relu(gelu) + μ_l on every block, LoRA r=8 on every block's c_proj).
- Trainable: A ∈ R^(3072,8) init N(0, 0.01), B ∈ R^(8,768) init zero, all 12 c_projs → 368,640 trainable params.
- Adam LR=1e-4, 2000 steps.
- DATA SPLIT (the fix): calib_ids = all_ids[:200], eval_ids = all_ids[200:300]. Need ≥300 wikitext-2 val sequences at SEQ_LEN=128. Verify the loader provides at least that many.
- μ_l precomputation on calib only.
- LoRA Adam training on calib only.
- Eval (every 200 steps + final) on eval_ids only.
- Best-state snapshot+restore using eval PPL.
- Sanity: LoRA-zero init eval on eval_ids should reproduce cycle 228's eval PPL or a held-out equivalent (~61.66 if the μ-only setup generalizes; otherwise different but reasonable).
- 3 seeds (42, 137, 2024) if budget allows; 1 seed otherwise.
- Confirmed if eval_ppl ≤ 48.0.
- Refuted if eval_ppl ≥ 58.0.
- Inconclusive if 48 < eval_ppl < 58.

# Handoff
Plan agent: design via direct rerun. New script `experiments/gelu_noneg_lora_disjoint_cycle239.py`. Output JSON + summary, comparing against cycle 234 leaked result (32.411) and cycle 228 (61.664).
