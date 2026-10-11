# Cycle 245 audit — item 255 (drop block 5 from 6-block compression)

**Status: PASS | interesting: true**

## Claim
BLOCKS=[1,2,4,7,8] K=512 joint Adam W_out+b_out, LR=1e-4, 4000 steps, EVAL_EVERY=100,
best-state restore, 3 seeds (42/137/2024), disjoint calib=all_ids[:200]/eval=all_ids[200:300],
SEQ_LEN=128. Claimed best_mean=53.826 ± 0.032, CONFIRMED (≤54.0), beats float 53.895 by
0.069, Δ vs cycle 221 6-block 59.515 = −5.689.

## Verdict logic (item 255)
- CONFIRMED if best_mean ≤ 54.0: 53.826 ≤ 54.0 → **CONFIRMED** ✓ matches claimed.
- Refuted band ≥57.0, inconclusive 54–57: not triggered.

## Verified numbers
Per-seed best: [53.86760, 53.78889, 53.82295] → recomputed mean 53.82648, std(unbiased=False)
0.03223 ✓. Per-seed init [74.85103, 74.85103, 74.85103] → init_mean 74.85103 (all seeds
identical, expected since fresh-fit is deterministic given frozen calibration). Per-seed
final@4000 [60.45450, 59.96061, 60.26975] → 60.2283 ± 0.2037 ✓. Best steps [1700, 1600,
1600] → 1633 ± 47 ✓. Δ vs c221 = −5.6885 ✓. Δ vs float 53.8949 = −0.0684 ✓ (statistical
tie, NOT a meaningful gain; matches the cycle 206/210/223/225/230/239 family where
Adam-optimised W_out slightly over-fits the 100-seq calibration eval slice).
Gap-closure: 5.6885 / (59.515 − 53.8949) = 101.2% of gap (claim "103%" uses the rounded
5.50 gap; both defensible).

## Method checks
- **Block count.** BLOCKS=[1,2,4,7,8] is 5 blocks, not 6 (line 19 of
  `experiments/drop_block5_cycle245.py`). Trainable param count 1,969,920 = 5 × (768×512 +
  768) confirms only 5 blocks are parameterised; block 5 is untouched (float). ✓
- **Cycle 221 match.** All hyperparameters (K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128,
  LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]) and the calib/eval splitting
  (`all_ids[:200]` vs `all_ids[200:300]`) are byte-identical to
  `experiments/adam_6block_earlystop_cycle221.py`. Fair apples-to-apples comparison. ✓
- **Compression recipe.** `fit_block_lstsq` uses `scores = act_all.std(0) * W_proj.norm(dim=1)`
  topk selection + full lstsq including bias column — identical to c221/c242. ✓
- **Best-state restore** asserted within 1e-3 per seed (L188-189). ✓

## Scientific content (interesting = true)
Dropping the single cycle-243 rank-1 contributor (block 5, Δ=+5.10 in fixed-fit restore)
closes the **entire** 5.5 PPL cycle-221 joint-cascade gap — the 5-block set re-fit without
block 5 lands at float parity. This forces a re-reading of cycle 243: its 4.4×
non-additivity (Σ delta 24.14 vs 5.50 total gap) was an artifact of the fixed-fit
attribution protocol (other 5 blocks stay compressed while measuring the single restore),
NOT an intrinsic joint-cascade property. Once the dropped block is excluded from the
trainable surface AND the other blocks are re-joint-optimised, one block accounts for
>100% of the gap. Block 5 is therefore a genuine structural bottleneck at K=512, not a
symmetric cascade contributor. This also explains cycle 233's middle-subset [3..8]
degradation and cycle 243's mid-block restore ordering under a single consistent story:
block 5's c_proj output is simply the hardest to replicate in a K=512 topk subspace
within the c221 set. Init PPL 74.851 is *higher* than c221's ~86.7 yet the floor is
*lower*, confirming that init quality does not drive the final floor; joint cascade
structure does. Seed std 0.032 is the tightest multi-block result on record.

## Caveats
- 5-block is a strict subset compression of c221's 6-block and does NOT beat c210 3-block
  46.556 (Δ=+7.27), so it is not a new Pareto winner on absolute PPL. It IS the strictest
  superset compression yet recorded at near-float parity (5 of 12 MLP blocks compressed).
- The −0.069 vs float is NOT a compression "beating float" — it is the standard
  calibration-overfit slack on a 12.7k-token eval slice; all Adam W_out fine-tunes in the
  c206/c210/c223/c225/c230/c239 family show it. Do not report as a gain.
- Single-seed calibration determinism: init_mean std 2.6e-6 is because the fresh topk+lstsq
  init is deterministic given identical calib ids; only Adam training introduces seed
  variance. Expected.
- Follow-ups already queued: (a) [1,2,4,7] and [1,2,4,8] to test whether 4-block also
  matches float; (b) drop block 4 instead (Δ=4.90 in c243) as replication check;
  (c) heterogeneous K — block 5 at K=1024/2048 with the other 5 at K=512 to recover full
  6-block coverage at float parity; (d) diagnose block 5's c_proj input distribution
  (std, kurtosis, effective rank) vs blocks 1/2/4/7/8.
