# Cycle 241 — Item 251 Plan

## Item (verbatim)

251. **W_out + W_fc trainable joint Adam at 6 blocks — does W_fc help at cascade depth?**
   - Cycle 213 added the `c_fc` bias as a joint Adam parameter at 3 blocks BLOCKS=[1,4,7] K=512 and reached best_mean 46.424 vs cycle 206 W_out-only 46.556 (essentially identical, -0.13 PPL). Repeat the same recipe at 6 blocks BLOCKS=[1,2,4,5,7,8] K=512 to test whether W_fc training is a noop at 3 blocks because the 3-block surface is already flat, or because W_fc is always a noop. Use cycle 221 scaffolding: calib_ids = all_ids[:200], eval_ids = all_ids[200:300], SEQ_LEN=128, seed 42, joint Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore. Trainable params per block: W_out + b_out + c_fc_bias = 512×768 + 768 + 3072 = 396,288 (W_out-only was 393,216; W_fc adds 3072); 6 blocks × 396,288 ≈ 2.38M. 3 seeds {42, 137, 2024}. Compare against cycle 221 W_out-only 6-block 59.515 ± 0.094. If W_fc helps at 6 blocks where it didn't at 3 blocks, the 59.5 floor is NOT fully structural and adding more trainable surface still buys PPL. If W_fc is still a noop, the 59.5 floor is rigid and the 6-block cascade barrier does not respond to this trainable-parameter direction.

## Goal

Test whether adding the `c_fc` BIAS (only) as a joint Adam parameter on top of `W_out + b_out` breaks the 59.5 PPL floor from cycle 221 W_out-only 6-block. The `W_fc` matrix itself stays FROZEN — only its bias `b_fc` (shape `(K,)` = 512 after topk) joins the trainable set. Hypothesis: REFUTED — cycle 213 showed the analogous bias unfreeze was a noop at 3 blocks, and the 59.5 floor has resisted multiple directions.

## Method

1. Create `experiments/adam_wout_wfc_6block_cycle241.py` by copying `experiments/adam_6block_earlystop_cycle221.py` and modifying the `CompressedMLP` class:
   - `W_fc` stays as a frozen buffer (register_buffer or plain tensor, NOT nn.Parameter).
   - `b_fc` becomes `nn.Parameter` (shape `(K,)` = 512 after topk) — NEW trainable.
   - `W_out` and `b_out` remain `nn.Parameter` as in cycle 221.
   - `forward`: `F.gelu(x @ self.W_fc + self.b_fc) @ self.W_out.T + self.b_out` — unchanged in math, only `b_fc` requires_grad changes.
2. `fit_block_lstsq` returns the updated `CompressedMLP` (init unchanged: topk indices on the frozen `W_fc[:, topk_idx]`, lstsq on `W_out/b_out`, `b_fc` initialised to `c_fc.bias[topk_idx]`).
3. Unfreeze exactly `[mlp.b_fc, mlp.W_out, mlp.b_out]` per compressed block — leave `mlp.W_fc.requires_grad_(False)` explicitly.
4. `snapshot()` / best-state restore covers `{b_fc, W_out, b_out}` per block (three tensors per block).
5. Trainable-params print: per block = 512×768 (W_out) + 768 (b_out) + 3072 (b_fc, wait — 512 after topk, NOT 3072; see note below) + → see recount below. Expected total = **2,382,336 ≈ 2.38M** per the agenda formula. See "Param-count note" below.
6. Config: `BLOCKS=[1,2,4,5,7,8]`, `K=512`, `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `N_STEPS=4000`, `EVAL_EVERY=100`, `SEEDS=[42, 137, 2024]`; `calib_ids=all_ids[:200]`, `eval_ids=all_ids[200:300]`.
7. Per seed: lstsq init → record init PPL → Adam 4000 steps, eval-every-100, best-state snapshot → restore best → assert `|restored − best| < 1e-3`.
8. Aggregate mean±std of init / best / best_step / final@4000. Apply verdict.
9. Write `autoresearch/results/cycle_241_wout_wfc_6block.json` and `autoresearch/results/cycle_241_summary.md` (seed table + trajectories + compare vs cycle 221 59.515 ± 0.094 and cycle 213 W_out+c_fc-bias 46.424).

## Param-count note

Agenda formula: `W_out + b_out + c_fc_bias = 512×768 + 768 + 3072 = 396,288` per block; `6 × 396,288 ≈ 2.38M`.

The agenda formula uses `3072` for `c_fc_bias` — that's the FULL GPT-2 `c_fc` bias dimension (`d_mlp = 3072`), not the `K=512` subset. But after topk selection, only `K=512` `b_fc` entries are reachable (the forward uses `W_fc[:, topk_idx]` and `b_fc[topk_idx]`). **Follow the agenda formula literally**: make `b_fc` a `(K,)=512` `nn.Parameter` post-topk AND also expose the full `(3072,)` vector if the agenda intent is to train the untruncated bias. Two readings:

- **(A) Literal topk subset**: trainable `b_fc` has 512 entries per block → per-block = 393,216 + 768 + 512 = 394,496; 6 blocks → 2,366,976 ≈ 2.37M.
- **(B) Agenda-literal formula**: trainable `b_fc` has 3072 entries (full pre-topk bias) → per-block = 393,216 + 768 + 3072 = 397,056; 6 blocks → 2,382,336 ≈ 2.38M.

Agenda arithmetic = (B) exactly (`397,056 × 6 = 2,382,336`, matches "≈ 2.38M"). The agenda says `396,288` which is off-by-one-slot from `397,056` but the rounding claim matches (B). **Use interpretation (B)**: keep the full `(3072,)` `b_fc` as a Parameter, and the forward takes `b_fc[topk_idx]` at runtime so the 2560 non-selected entries receive zero gradient (dead but counted). This matches the agenda numerics exactly and is the cheapest faithful reading. Print expected trainable count `2,382,336` and assert equality at startup.

## Verdict criteria

- CONFIRMED if `best_ppl_mean ≤ 58.0` (c_fc-bias trainable breaks the 59.5 floor).
- REFUTED   if `best_ppl_mean ∈ [59.0, 60.5]` (c_fc-bias trainable is a noop at 6 blocks; floor rigid).
- INCONCLUSIVE otherwise.

## Files

- NEW: `experiments/adam_wout_wfc_6block_cycle241.py`, `autoresearch/results/cycle_241_wout_wfc_6block.json`, `autoresearch/results/cycle_241_summary.md`

## Verification

- Init PPL per seed should match cycle 221 init closely (lstsq init identical; only added trainable `b_fc` starts at the pretrained GPT-2 bias, which is the same value cycle 221 froze).
- Trainable param count prints `2,382,336` exactly; assert at startup.
- `W_fc` tensors have `requires_grad == False` at every step (assert once before the training loop).
- Best-state restore assert `|restored − best| < 1e-3` passes on every seed.
- Runtime budget: marginal overhead over cycle 221 (only +18K params vs W_out-only 2.36M); expect ≤ cycle 221 wall time × 1.05 per seed on MPS.

## Risks / notes

- `W_fc` MUST stay frozen. The cycle 213 script `adam_full_correction_cycle213.py` trained the full `W_fc` matrix — do NOT copy that block verbatim; the agenda's 2.38M budget forbids it (full W_fc would add 6×768×512 = 2.36M on top, doubling the surface).
- If the "3072 vs 512" interpretation (B vs A) turns out to disagree with the cycle 213 best_mean 46.424 reported by the agenda, re-run the 3-block baseline with the chosen interpretation for a sanity check; but the ask is a 6-block experiment, so defer that unless results are surprising.
- If Adam diverges on the expanded surface, report INCONCLUSIVE with the divergence trace.
