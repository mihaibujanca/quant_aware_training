# Cycle 279 Plan — Item 289: [1,4] K=512 W_out + W_fc trainable

## Item (verbatim)
289 (P2): Cycle 266 [1,4] K=512 + W_out + W_fc trainable — does unfreezing W_fc help at the 2-block Pareto winner?

## Hypothesis (item 289 verbatim)
W_out absorbs the DOF at 2 blocks the same way it does at 3-6 blocks → best_mean ∈ (43.6, 44.1) (within ±0.25 PPL of cycle 266 [1,4]=43.852).

## Sliced-vs-full discrepancy (CRITICAL)
Item 289 setup line claims trainable = `W_fc_l (768×3072), b_fc_l (3072)` ⇒ 5,512,704 params total. This REPEATS the cycle 248 agenda error: cycle 213's `FullCompressedMLP` only trains the SLICED `(768, K)` columns selected by topk (K=512), NOT the full `(768, 3072)` matrix. Faithful continuation of the cycle 213/241/248/262 W_fc retraining family REQUIRES the sliced semantics. We use the SLICED form: W_fc slice `(768, 512)` + b_fc slice `(512)` per block. Per-block trainable = `512×768 + 768 + 768×512 + 512 = 787,712`. Total 2 blocks = **1,575,424** (not 5,512,704). Apples-to-apples with cycles 213/241/248/262.

## Verdict thresholds (item 289 verbatim)
- CONFIRMED (W_fc noop) if best_mean ∈ (43.6, 44.1).
- REFUTED (W_fc helps) if best_mean ≤ 43.5.
- INCONCLUSIVE if best_mean ∈ (43.5, 43.6) — marginal — OR best_mean ≥ 44.1.

## Setup
New script `experiments/two_block_wout_wfc_cycle279.py`. BLOCKS=[1, 4], K=512. Combines cycle 213 `FullCompressedMLP` (W_fc/b_fc/W_out/b_out all trainable on the topk-sliced (768, K) columns) with cycle 266 2-block infrastructure. Joint Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore (snapshot all 4 tensors per block), 3 seeds {42, 137, 2024}, disjoint N_CALIB=200/N_EVAL=100, SEQ_LEN=128, MPS GPT-2 small wikitext-2 val. fit_block_lstsq init from cycle 213 (topk by `act.std(0) * W_proj.norm(dim=1)`, lstsq init for W_out/b_out, W_fc/b_fc init = c_fc weight slice). 3 fits total, ~5 min/fit ≈ 15 min wall.

## Steps
1. Create `experiments/two_block_wout_wfc_cycle279.py` (cycle 266 2-block scaffold + cycle 213 FullCompressedMLP + sliced W_fc trainable).
2. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/two_block_wout_wfc_cycle279.py 2>&1 | tee autoresearch/logs/cycle_279_run.log`.
3. Save `autoresearch/results/cycle_279_two_block_wout_wfc.json` with per-seed best/init/history + aggregate best_mean ± std + verdict + cycle 266 [1,4] (43.852), cycle 213 3-block W_out+W_fc (46.424), cycle 248 6-block K=128 W_out+W_fc (67.609), float (53.895) baselines + sliced semantics note.
4. Save `autoresearch/results/cycle_279_summary.md` with verdict, per-seed table, baselines, sliced-vs-full discrepancy note.

## Bookkeeping
Delete item 289 from NEXT/P2 (P2 then empty; P0/P1 already empty); insert `### Cycle 279` at top of PAST EXPERIMENTS (above `### Cycle 278`) with verdict + sliced-semantics note. Append TSV row to `autoresearch/logs/experiments.tsv`. Stage cycle files only (no pre-existing dirty files). Primary commit, hash, manifest, bookkeeping commit. Audit at `autoresearch/audits/cycle_279.md` + `autoresearch/results/cycle_279_audit.txt`. Exec result at `autoresearch/results/cycle_279_exec_result.txt`.
