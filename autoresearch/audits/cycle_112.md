# Audit — Cycle 112 (Item 123)

**Item:** GELU negative-leak aware compression (corrected) — GPT-2 FFN
**Verdict:** INCONCLUSIVE
**Audit status:** PASS (verdict correct) — two flags raised

---

## G1: Computation check

Script: `experiments/gelu_leak_aware_compression_corrected_gpt2.py`

**BUG FOUND (line 385/392): `k_eff` contaminates Condition A.**

The script computes `k_eff = min(K, n_pos)` at line 385, then uses `k_eff` for ALL three conditions — including Condition A (line 392: `topk_all = torch.topk(scores, k=k_eff, ...)`). Condition A is supposed to be standard topk from all 3072 channels with k=512 (6x compression). Instead, in blocks 4/7/11 where `n_pos` = 53-71, Condition A selects only 53-71 channels (~46x compression), identical capacity to B and C.

The correct implementation would use `k_A = K` (always 512) for Condition A, and `k_BC = min(K, n_pos)` for Conditions B and C. As written, A is not the intended baseline — all three conditions are equally crippled.

Consequence: the A vs B comparison is between two equally-destroyed models, not between standard 6x compression and leak-aware 6x compression. The reported "B is 135% worse than A" cannot be interpreted as evidence about leak-aware compression.

Other implementation details are correct:
- b_leak computation: `W_out[:, neg_ch] @ mean(GELU(z)|z<0)` — correct formulation.
- B lstsq target = `y - b_leak` (the cycle 105 fix) — correctly implemented at line 397.
- C lstsq target = `y` (no b_leak) — correct control at line 405.
- Probe split (100 fit / 100 eval) — correct.
- 3 seeds, 4 blocks, 500 Adam steps — matches plan.
- Baseline PPL (55.34) in expected range — model loaded correctly.

## G3: Numbers match criteria

The criteria require B outperforms A by >=3% in >=2/4 blocks for CONFIRMED, within 1% in all blocks for REFUTED. B is catastrophically **worse** than A (-135%), so CONFIRMED is impossible. REFUTED requires all blocks within 1% — not met either (gain is -135%).

However, the script's manifest says INCONCLUSIVE due to a different reason: "only 1 eligible block (N_pos >= 256), so wins >= 2 is unreachable." The manifest's N_pos eligibility guard is a post-hoc rationalization — there is no actual abort guard in the code (confirmed: only lines 321/324 have guards, both for baseline PPL). The script ran all 4 blocks regardless.

The underlying problem is the k_eff bug (G1): with all conditions equally crippled, the A-vs-B comparison is meaningless. INCONCLUSIVE is the correct verdict, though the stated reason is incomplete — it should cite the k_eff bug, not just the vacuous partition.

**Verdict: INCONCLUSIVE is correct, but for a deeper reason than stated.**

## G6: Design failure analysis

**FLAG: Near-vacuous partition is a design failure, not evidence about the hypothesis.**

The agenda item (123) specifies: "Partition channels into positive-dominated (neg_frac < 0.5) and negative-dominated (neg_frac >= 0.5)." With cycle 111 showing mean neg_frac = 0.73-0.86, this threshold produces N_pos = 53-546 out of 3072 channels. The partition is near-vacuous in 3/4 blocks.

This is compounded by the k_eff bug — even if the partition were reasonable, Condition A is crippled to match B's channel count. Two independent failures:

1. **Partition design**: threshold 0.5 is wrong for a distribution centered around 0.83. Cycle 111 already showed this distribution. A threshold near the median (e.g., 0.83) would give ~1536 positive channels.
2. **Code bug**: k_eff shared across conditions A/B/C instead of A getting k=512 independently.

Neither failure tells us anything about whether leak-aware correction improves compression.

**Recommendation:** Before re-running, fix both issues:
- (a) Use k_A = K = 512 for Condition A (always), k_BC = min(K, N_pos) for B/C.
- (b) Either raise the neg_frac threshold to the per-block median, or (better) use the continuous weighting scheme suggested in the manifest.

Alternatively, apply leak correction on top of standard topk (no channel partitioning) — test the bias correction idea without the partition.
