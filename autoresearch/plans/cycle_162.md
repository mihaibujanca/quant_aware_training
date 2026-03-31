# Cycle 162 Plan — Item 172: GELU-aware dead/near-saturated channel census

## Confirmed if
at least 5% of channels across all blocks are dead or near-saturated (removable with <0.5% PPL increase on single-block swap).

## Refuted if
fewer than 1% of channels are removable at any threshold tested.

## Inconclusive if
1-5% of channels qualify but removal causes >0.5% PPL increase.

## Prior expectation
Four prior cycles (28, 32, 83, 128) found near-zero removable channels. Cycle 128 found 0 strict-removable at T=3/4. The 5% confirmation bar is 130x above that result. Strong prior toward refutation; this cycle's value is the T=2 threshold and contribution-score axis.

## Duplicate check
- Cycle 28/32: REFUTED, flawed methods. Cycle 83: REFUTED, 14 dead / 3 near-linear of 36864.
- Cycle 128: INCONCLUSIVE — pre-GELU, T=3/4, 0 strict-removable, 14 soft (>95% dead-zone).
- **This cycle adds**: T=2 threshold, output-contribution score, single-block PPL validation on held-out data.

## Steps

1. **Create** `experiments/gelu_channel_census_cycle162.py` using cycle 128 script as template.
2. **Config**: 3 seeds [42,123,7], 256 seqs × 256 tokens calibration, batch=8, GPT-2 (124M).
3. **Pre-GELU census** — register a forward hook on `transformer.h.{i}.mlp.c_fc` and capture its **output** (shape `[batch, seq, 3072]`), which is the pre-GELU intermediate after W_up @ x + b_up:
   - For each channel j across all calibration tokens, record `min_z[j]`, `max_z[j]`.
   - Dead: `max_z[j] < -T` for T ∈ {2, 3, 4}.
   - Near-linear: `min_z[j] > T` for T ∈ {2, 3, 4}.
   - Report strict counts per block and globally, averaged across 3 seeds.
4. **Output-contribution score**: for each channel j in block i:
   - `score_j = ‖W_down[:,j]‖₂ × rms(GELU(z_j))` (rms over all calibration tokens).
   - Normalize scores within each block (sum=1).
5. **Null baseline**: equal-share = 1/3072 ≈ 0.0326%. Flag any threshold above this.
6. **Single-block PPL validation**: load 256 held-out OWT eval sequences (disjoint from calibration). For top-candidate block (most dead+near-saturated), zero out those channels and measure PPL delta. Repeat for bottom 1%/5% by contribution score.
7. **Plot** (save to `autoresearch/results/cycle_162_census.png`):
   - Panel 1: per-block dead/near-linear counts at T=2,3,4 (grouped bar).
   - Panel 2: output-contribution score distribution (histogram, all blocks pooled, log-x).
   - Panel 3: contribution score vs channel index, colored by block.
8. **Save** JSON results to `autoresearch/results/cycle_162_census.json`.

## Verdict logic
- Count total qualifying channels (dead OR near-linear at best T) across all 12 blocks × 3072 = 36864.
- If ≥5% (1843+) AND single-block PPL delta < 0.5%: **CONFIRMED**.
- If <1% (368) at all T: **REFUTED**.
- If 1-5% but PPL delta > 0.5%: **INCONCLUSIVE**.

## Commit
```
git add experiments/gelu_channel_census_cycle162.py autoresearch/results/cycle_162_census.*
git commit -m "[autoresearch] cycle 162: GELU-aware channel census — <VERDICT>"
```

## Agenda update
Move item 172 from NEXT to PAST with outcome, commit hash, artifacts.
