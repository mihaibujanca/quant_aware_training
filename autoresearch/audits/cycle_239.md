# Cycle 239 audit — item 249 (gelu_noneg + μ_l + LoRA r=8 disjoint)

**Status: PASS** — old_outcome=confirmed → new_outcome=confirmed.

**Phase 1 (literal threshold).** Item 249: CONFIRMED if `eval_ppl ≤ 48.0`,
REFUTED if `≥ 58.0`. Result: `final_eval_ppl_mean = 47.692 ± 0.143` over 3
seeds (47.666 / 47.531 / 47.879). 47.692 ≤ 48.0 by 0.31 PPL → CONFIRMED.

**Phase 2 (quality).**
- G1 (split discipline): code at `experiments/gelu_noneg_lora_disjoint_cycle239.py`
  builds `calib_ids = all_ids[:200]`, `eval_ids = all_ids[200:300]`, asserts
  object-id disjointness AND tensor-equality on the first pair. μ
  precomputation hooks run only over `calib_ids` (line 144–146); the LoRA
  Adam loop iterates only `calib_ids[idx]` (line 312); every `eval_ppl()`
  call inside training uses `eval_ids`. The init-eval sanity check
  (`init_eval_ppl == post_mu_eval_ppl` within 1e-3) catches any state leak
  between seeds.
- G2 (variance): 3 seeds, std 0.143 PPL — ~0.3% of the mean. All 3 seeds
  individually under 48.0. Tight enough to trust the threshold call.
- G3 (interpretation): cycle 234's 32.411 (calib==eval) → cycle 239's 47.692
  (held-out) is +15.28 PPL of pure calib overfit, exactly what the cycle 234
  audit predicted. The residual 47.692 still clears 48.0 → the LoRA r=8
  W_out residual on top of `gelu_noneg + μ_l` does generalise.
- G4 (surprising claim): the gelu_noneg + μ + LoRA model beats the held-out
  float baseline (53.895) by 6.20 PPL (-11.5%). This is the SAME phenomenon
  as cycles 206 / 210 / 223 / 225 / 230 (Adam W_out correction beats float
  on held-out): a small learned correction on top of a quantised /
  activation-modified base regularises against the eval slice's harder
  distribution. The eval slice is structurally harder for fp32 GPT-2 (53.9)
  than the calib slice (45.1, cycle 220 ref), so beating fp32 here is
  consistent rather than miraculous. Worth flagging but not an extraordinary
  claim relative to the family.
- G5 (manifest): `learned` and `next` both reflect the loop closure and
  enumerate sensible follow-ups (rank sweep, SVD of A@B, longer training).
- G6 (metric choice): single literal threshold on a single quantity — no
  metric flexibility to abuse.

**Phase 3.** No corrections needed. Old outcome stands.

**Comparison context.**
- cycle 210 (K=512 W_out Adam, held-out): 46.556
- cycle 223 (K=128 W_out Adam, held-out): 48.989
- cycle 239 (gelu_noneg + μ + LoRA r=8, held-out): 47.692
The gelu_noneg + μ + LoRA recipe lands between K=128 and K=512 W_out
correction at a comparable 368K trainable params, via a structurally
different mechanism (activation change + low-rank c_proj residual rather
than channel selection on W_out).
