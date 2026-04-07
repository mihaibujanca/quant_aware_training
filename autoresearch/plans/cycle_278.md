# Cycle 278 Plan — Item 288: All-blocks c_proj output eff_rank census

## Item (verbatim)
288 (P1): All-blocks c_proj output eff_rank census — extend cycle 267 to blocks {0, 2, 3, 4, 6, 7, 8, 9, 10}.

## Hypothesis
The "U-shape rank-concentration" pattern holds across all 12 blocks: at LEAST ONE of {block 4, block 7} has c_proj output eff_rank ≤ 50 (matching block 11's 22.49 scale more than block 5's 179) AND ALL blocks {2, 3, 6, 8, 9, 10} have eff_rank ≥ 100 (mid-block high-rank pattern).

## Verdict thresholds (item 288 verbatim)
- CONFIRMED iff (block 4 eff_rank ≤ 50 OR block 7 eff_rank ≤ 50) AND ALL of blocks {2, 3, 6, 8, 9, 10} have eff_rank ≥ 100.
- REFUTED iff blocks 4 AND 7 BOTH have eff_rank ≥ 100.
- INCONCLUSIVE otherwise.

## Setup
New script `experiments/all_blocks_eff_rank_cycle278.py`. Verbatim reuse of cycle 267 SVD machinery: load GPT-2 small, register forward hooks on `model.transformer.h[l].mlp.c_proj` for l ∈ {0, 1, 2, ..., 11} (all 12 blocks; cycle 267 values for {1, 5, 11} reproduce as sanity check on the same 200-doc calib slice). Standard calib: wikitext-2 validation, first 200 sequences with len ≥ SEQ_LEN=128, all_ids[:200], expected 25,600 tokens. For each block: collect Y_l ∈ R^{25600 × 768}, compute C_l = Y_l^T Y_l / N (768×768), torch.linalg.eigh → eff_rank = exp(H(p)) where p = eigvals/Σeigvals; top-{1, 5, 20, 50, 100, 200} energy fractions. Also per-block c_proj.weight SVD (3072×768) → top-20 singular values (operator-side comparison). Pure analysis, 0 trainable params, no Adam, no retraining. Wall ≈ 5 min on MPS.

## Steps
1. Create `experiments/all_blocks_eff_rank_cycle278.py`. BLOCKS = list(range(12)). Cross-check blocks {1, 5, 11} reproduce cycle 267 values (eff_rank match within 1e-3).
2. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/all_blocks_eff_rank_cycle278.py 2>&1 | tee autoresearch/logs/cycle_278_run.log`.
3. Save JSON `autoresearch/results/cycle_278_all_blocks_eff_rank.json` with per-block {eff_rank, total_energy, top1_eigval, topk_energy{1,5,20,50,100,200}, top20_mlp_eigvals, wproj_top20_sv}, full 12-block profile, verdict + verdict_logic, sanity-check deltas vs cycle 267.
4. Save markdown summary `autoresearch/results/cycle_278_summary.md`.

## Baselines
Cycle 267 block 1 = 1.94 (92.94% top-1), block 5 = 179.19, block 11 = 22.49 (48.95% top-1); cycle 270 block 11 c_proj.weight rank-100 top-22 energy 14.14%; float 53.895.

## Bookkeeping
Delete item 288 from NEXT/P1 (item 289 becomes new top of P1; P0 stays empty); insert `### Cycle 278` at top of PAST EXPERIMENTS (above `### Cycle 276`) with verdict + 12-block profile reading. Append TSV row to `autoresearch/logs/experiments.tsv`. Stage cycle files only. Primary commit, get hash, manifest, bookkeeping commit. Audit at `autoresearch/audits/cycle_278.md` + `autoresearch/results/cycle_278_audit.txt`. Exec result at `autoresearch/results/cycle_278_exec_result.txt`.
