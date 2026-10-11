# Cycle 244 audit — item 254 (sliding 3-block sweep)

**Status: PASS | interesting: true**

## Claim
Sliding 3-block sweep at matched c242 recipe (K=512, W_out+b_out joint Adam, LR=1e-4,
4000 steps, best-state restore, 3 seeds, disjoint calib/eval, SEQ_LEN=128, 1.18M trainable
per config). Float held-out 53.895.

## Verified numbers
- [3,4,5]: per-seed [49.411, 49.483, 49.538] → mean 49.477 ± 0.052 (init 65.853, final@4000 50.337). Recomputed mean = 49.4773 ✓.
- [5,6,7]: per-seed [50.871, 50.860, 50.964] → mean 50.899 ± 0.047 (init 66.538, final@4000 53.310). Recomputed mean = 50.8983 ✓.
- [6,7,8]: per-seed [52.068, 52.025, 52.022] → mean 52.038 ± 0.021 (init 67.651, final@4000 56.220). Recomputed mean = 52.0383 ✓.

## Verdict logic (item 254)
- CONFIRMED if any window mean ≤ 48.5: `49.477, 50.899, 52.038` → none ≤ 48.5 → False.
- REFUTED if all window means ≥ 52.0: only [6,7,8]=52.038 clears, [3,4,5]/[5,6,7] do not → False.
- Else INCONCLUSIVE → **INCONCLUSIVE** ✓ matches claimed verdict.

## Scientific content (interesting = true)
Joint with c242 anchors the ΔPPL curve vs window-start is cleanly monotone:
[1,2,3]→47.135, [3,4,5]→49.477, [5,6,7]→50.899, [6,7,8]→52.038, [8,9,10]→55.192
(diffs +2.342, +1.422, +1.139, +3.154). Seed std 0.021-0.052 — very tight. Init PPL is
also monotone (65.85→66.54→67.65→71.35) and the init→best drop is ~16 PPL at every
window, so the late-block penalty is already encoded in the K=512 topk projection before
any Adam update, not introduced by training dynamics.

**Key re-reading of c243.** Cycle 243's "mid-cascade (blocks 4,5) heaviest" was a
joint-fit per-block restoration ranking, not a fresh-fit structural property: in a
fresh 3-block fit [3,4,5] is the EASIEST of the three new windows, and cost tracks the
deepest block in the window. Per-block restoration delta in a joint 6-block cascade does
NOT predict fresh-fit 3-block difficulty.

## Method / execution checks
- Script matches claimed recipe (CONFIGS, K, LR, N_STEPS, EVAL_EVERY, SEEDS verified in
  `experiments/sliding_3block_cycle244.py` lines 27-36). Disjoint calib/eval enforced by
  `id()` assertion (L69). Best-state restore asserted to round-trip within 1e-3 (L219).
- Frozen-blocks assertion over all 12 blocks prevents grad leakage (L168-172).
- Trainable surface is W_out + b_out only; W_fc submatrix and b_fc subset registered as
  buffers (L73-82) — matches c242 surface.

## Caveats
- Small held-out (100 seqs × 127 tokens = 12.7k tokens); band-straddling verdict is
  tight (+0.038 on [6,7,8]) — follow-up with larger eval would sharpen the REFUTED call.
- Monotone claim is 5 points across window-start; no [4,5,6]/[7,8,9] interior confirm.
