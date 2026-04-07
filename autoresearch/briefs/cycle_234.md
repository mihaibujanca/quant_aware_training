# Item
244 (P2): gelu_noneg + first-moment + LoRA r=8 W_out residual — eat the 16.6 PPL residual?
On top of cycle 228's all-12-block `relu(GELU(z)) + μ_l` patch, add LoRA r=8 adapter on each block's c_proj (W_out, shape (3072, 768) for HF GPT-2 Conv1D layout), trained via Adam on CE 2000 steps seed 42.

# Context
- Cycle 228 (INCONCLUSIVE): gelu_noneg + per-channel μ_l → 61.664 PPL (vs baseline 45.064, +16.6 residual). Recovered ~34× from cycle 220 catastrophe but residual is higher-order.
- This cycle: add learnable LoRA correction on top.

# Dependencies
- 234 / cycle 220 (done) — gelu_noneg base.
- 238 / cycle 228 (done) — first-moment correction.

# Key Files
- experiments/gelu_noneg_corrected_cycle228.py — has the gelu_noneg + μ_l infrastructure.

# Constraints
- All 12 blocks patched with `relu(gelu(z)) + μ_l`.
- LoRA r=8 on each block's c_proj.weight (shape (3072, 768) Conv1D).
- LoRA decomposition: c_proj_corrected = c_proj + (A @ B), where A ∈ R^(3072, 8) and B ∈ R^(8, 768). A initialized standard, B initialized to ZERO so initial behavior matches gelu_noneg+μ baseline.
- Trainable params: 12 blocks × (3072×8 + 8×768) = 12 × 30720 = 368,640.
- Adam, LR=1e-4, 2000 steps, CE loss.
- Seed 42 (1 seed for scouting).
- Calibration data: same N=200 wikitext-2 val sequences as cycles 220/228.
- Eval split: same as cycles 220/228 (note: cycles 220/228 use the WHOLE 200 sequences as eval; verify by reading cycle 220 script).
- Confirmed if final_ppl ≤ 48.0.
- Refuted if final_ppl ≥ 58.0.
- Inconclusive if 48 < final < 58.

# Handoff
Plan agent: design via combining cycle 228 patch + a LoRA adapter on c_proj. New script `experiments/gelu_noneg_lora_cycle234.py`. Output JSON + summary, comparing against cycle 228 baseline 61.664 and float baseline 45.064.
