# Cycle 292 Plan — Item 302

## Dependencies
None (cycle 280 template + cycle 278 census + cycle 289 sub-eff_rank precedent on disk).

## Hypothesis (item 302 verbatim)
Block 1 c_proj rank-1 weight-SVD retrain. Under-eff_rank test like cycle 289 did for block 10. Cycle 280 confirmed block 1 r=2 → 49.03 PPL @ 2,304 params (eff_rank 1.94, r/eff = 1.03). Cycle 289 confirmed block 10 r=5 → 51.47 PPL @ 4,608 params (eff_rank 16.70, r/eff = 0.30) — first sub-eff_rank success. Symmetric question for block 1: does r=1 weight-SVD + Adam still beat float at 1,536 params (1×768 + 768)? Cycle 280 no-retrain showed r=1 → 59.75, r=2 → 57.66 (Δ=2.09) — Adam might absorb the gap. Block 1 eff_rank=1.94 puts r=1 at r/eff=0.52, the sub-eff_rank regime cycle 289 opened. Hypothesis: r=1 retrain reaches ≤51 PPL at 1,536 params, new lowest-param Pareto anchor 33% below cycle 280.

## Approach
Adapt `experiments/block1_lowrank_cycle280.py` verbatim with `RANKS_NORETRAIN=[1]`, `RETRAIN_RANK=1`, thresholds from item 302. Stage A: re-verify cycle 280 r=1 noretrain (59.75). Stage B: retrain r=1, joint Adam over (B,bias)=(1,768)+(768)=1,536 params, 3 seeds {42,137,2024}, best-state restore.

## Files
`experiments/block1_rank1_cycle292.py`; `autoresearch/results/cycle_292_block1_rank1.json` `_summary.md` `_manifest.txt` `_exec_result.txt`; `autoresearch/logs/cycle_292_run.log`; `autoresearch/audits/cycle_292.md`, `autoresearch/results/cycle_292_audit.txt`.

## Config
TARGET_BLOCK=1; RANKS_NORETRAIN=[1]; RETRAIN_RANK=1; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = 1,536. Wall ~7 min.

## Verification
Float PPL within 0.5 of 53.895; SVD shapes U(3072,768), S(768,), Vh(768,768); restore c_proj.weight between stages assert ≈float (<1e-3); per-seed init PPL deterministic == noretrain r=1; trainable count ==1,536 asserted; per-seed |restored−best|<1e-3; noretrain r=1 cross-check vs cycle 280 (59.75) within 0.01.

## Verdict logic (item 302 verbatim)
CONFIRMED iff r=1 retrained best_mean ≤ 51.0 (new lowest-param Pareto anchor); REFUTED iff ≥ 53.5 (r=2 is the floor); INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 280 b1 r=2 best_mean=49.0324 (2,304); cycle 289 b10 r=5 best_mean=51.4659 (4,608); cycle 280 b1 r=1 noretrain=59.75; cycle 278 b1 c_proj eff_rank=1.94, top-1 energy=0.9294.
