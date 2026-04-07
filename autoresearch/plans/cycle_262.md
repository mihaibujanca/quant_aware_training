# Cycle 262 Plan — Item 268

## Item (verbatim)
268 (P2): K=1 + W_fc bias trainable on 3 blocks — does the c_fc bias become useful at the smallest K? BLOCKS=[1,4,7], K=1, FullCompressedMLP-style trainable {W_out (768), b_out (768), c_fc bias slice (1 scalar per block)} per block. 3 seeds. Cycle 230 K=1 3-block W_out-only baseline = 59.245. Hypothesis: at K=1 the W_out path is starved enough that the bias becomes useful.

## Hypothesis
At K=1 the c_fc bias finally has a non-trivial effect because the W_out path is starved → best_mean ≤ 58.5 (beats cycle 230 K=1 3-block 59.245 by ≥0.7 PPL).

## Baselines
- Cycle 230 K=1 3-block W_out-only: best_mean=59.245.
- Cycle 213 K=512 W_out+c_fc bias: noop (3-block).
- Cycle 241 K=512 6-block W_out+c_fc bias: noop (59.533 vs 59.515).
- Cycle 240 K=1 6-block: 96.886 (catastrophic).
- Float PPL: 53.895. Cycle 210 K=512 [1,4,7]: 46.556.

## Setup
Create `experiments/k1_wfc_bias_cycle262.py` based on cycle 230 K=1 3-block. BLOCKS=[1,4,7], K=1. Use FullCompressedMLP-style class (parameter `b_fc` on the 1-channel slice) — but keep `W_fc` as a buffer (frozen) since the brief asks for c_fc *bias* trainable, not W_fc. Three trainable params per block: `W_out` (1×768=768), `b_out` (768), `b_fc` (1 scalar). Total: 3 × (768 + 768 + 1) = **4,611 trainable params**. Joint Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore (snapshot W_out, b_out, b_fc). 3 seeds {42, 137, 2024}, disjoint splits N_CALIB=200 / N_EVAL=100, SEQ_LEN=128. Reuse cycle 230 calib loader, hooks, fit_block_lstsq scoring (topk on `act_std × W_proj.norm`). Single forward pass per seed for collection. Same per-K aggregation. trainable_params reported in JSON.

## Verdict
**CONFIRMED** if best_mean ≤ 58.5.
**REFUTED** if best_mean ≥ 59.5.
**INCONCLUSIVE** otherwise.

## Execute
Working dir `/Users/mihai/Projects/quant_aware_training`. Run with `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/k1_wfc_bias_cycle262.py 2>&1 | tee autoresearch/logs/cycle_262_run.log`. Output JSON: `autoresearch/results/cycle_262_k1_wfc_bias.json` (per-seed init/best/best_step/final/history, aggregate, cycle 230 baseline 59.245, verdict). Summary md: `autoresearch/results/cycle_262_summary.md`.

## Bookkeeping
Stage cycle 262 files ONLY (NOT pre-existing dirty files like program.md, aleph/pullback_geometry.py, prior cycle artifacts). Commit `[autoresearch] cycle 262: K=1 + c_fc bias 3-block — <verdict>`. Update agenda.md: delete item 268 from NEXT/P2; prepend `### Cycle 262` PAST entry above `### Cycle 261`. Append TSV row. Manifest, bookkeeping commit, exec_result.

## Risks
The c_fc *bias* on a 1-channel slice is a single scalar — its derivative path goes through GELU' at exactly one z-value per token then into W_out, so it should be highly active under Adam. Risk: Adam may push the bias to a runaway value that destabilises GELU; mitigated by 4000-step early-eval-100 best-restore and the small LR=1e-4. Cycle 240 6-block K=1 crashed at 96.886, but cycle 230 3-block K=1 works at 59.245 — adding 1 scalar per block should not destabilise. If b_fc is frozen instead of trainable by mistake, the result reduces exactly to cycle 230 (59.245) — this is the negative-control sanity. Run cap ~10 minutes.
