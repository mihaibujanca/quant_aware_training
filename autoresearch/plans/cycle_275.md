# Cycle 275 Plan — Item 285: Hybrid [1] K=512 + block 11 rank-22 c_proj compose?

## Item (verbatim)
285 (P0): Hybrid Pareto recipe — 1-block [1] K=512 + block 11 rank-22 c_proj compose? Cycle 271 [1] K=512 = 43.821 (394K). Cycle 270 block 11 rank-22 c_proj retrain = 54.469 (17.7K). DISJOINT blocks. Joint Adam trainable = block 1 W_out (512×768) + b_out (768) + block 11 right factor B (22×768) + bias (768) = 411,648.

## Hypothesis
The two Pareto winners compose nearly additively → best_mean ≤ 43.5 (≥0.3 PPL below cycle 271 [1] alone at 43.821, essentially free addition of block 11 compression at a 4.5% trainable param increase).

## Verdict thresholds (item 285 verbatim)
- CONFIRMED iff best_mean ≤ 43.5.
- REFUTED iff best_mean ≥ 45.0.
- INCONCLUSIVE iff 43.5 < best_mean < 45.0.

## Setup
New script `experiments/hybrid_block1_block11_cycle275.py` combining cycle 270 (block 11 LowRankCProj r=22) and cycle 271 (block 1 K=512 CompressedMLP). Per seed: load fresh GPT-2 small; collect MLP I/O for block 1; install K=512 topk+lstsq CompressedMLP at block 1 (cycle 271 `fit_block_lstsq`); SVD c_proj_11.weight once (cached across seeds); install LowRankCProj(A_frozen=U[:,:22]·Σ[:22], B=Vh[:22,:], bias=c_proj_11.bias) at block 11. Freeze all params; unfreeze block 1 W_out+b_out AND block 11 B+bias. Joint Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore (snapshot includes BOTH blocks). 3 seeds {42,137,2024}. Disjoint N_CALIB=200/N_EVAL=100, SEQ_LEN=128, MPS. Trainable=393,984+17,664=**411,648**. Per-seed assert restore<1e-3 and trainable count.

## Steps
1. Create `experiments/hybrid_block1_block11_cycle275.py` — copy `CompressedMLP`, `collect_mlp_io`, `fit_block_lstsq`, `eval_ppl` from cycle 271 verbatim; copy `LowRankCProj` from cycle 270 verbatim. Single SVD of c_proj_11 outside seed loop. Per-seed: install both compressors, eval init PPL, freeze all/unfreeze 4 tensors, Adam loop with snapshot of (block_1.W_out, block_1.b_out, block_11.B, block_11.bias), restore best, assert.
2. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/hybrid_block1_block11_cycle275.py 2>&1 | tee autoresearch/logs/cycle_275_run.log`.
3. Save JSON `autoresearch/results/cycle_275_hybrid_block1_block11.json` with all per-seed history + aggregate (best_mean, best_std, best_min, best_max, init_mean, final_mean) + baselines {cycle 270: 54.469, cycle 271: 43.821, cycle 266: 43.852, float: 53.895} + verdict + verdict_logic.
4. Save markdown summary `autoresearch/results/cycle_275_summary.md`.

## Baselines
cycle 271 [1] K=512 = 43.821 (394K, lowest-PPL Pareto winner); cycle 270 block 11 rank-22 retrain = 54.469 (17.7K); cycle 266 [1,4] K=512 = 43.852 (788K, dominated by 271); cycle 273 hetero K_11=22 = 52.637 (REFUTED, channel-K not weight-SVD); float = 53.895.

## Bookkeeping
Delete item 285 from NEXT/P0; insert `### Cycle 275` block at top of PAST EXPERIMENTS with verdict + Pareto reading. Append TSV row to `autoresearch/logs/experiments.tsv`. Stage cycle files only (not pre-existing dirty files). Primary commit, get hash, manifest, bookkeeping commit. Audit at `autoresearch/audits/cycle_275.md` + `autoresearch/results/cycle_275_audit.txt`.
