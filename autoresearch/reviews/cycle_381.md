verdict: APPROVE with NON-BLOCKING notes
round: 1
issues: 0 blocking, 3 non-blocking
feedback:
  - NB1 (param count error in plan): Compressed adapter per-block params are wrong. Plan says `2*128*512 + 128 + 768 = 131,968` but adapter_W2 is (768, h)=(768, 128), not (512, 128). Correct adapter-only count: 128*512 + 128 + 768*128 + 768 = 164,736/block. The script will compute correct counts from the actual tensors, so this is a documentation error, not a runtime bug. Update the plan's expected param section before execution so the verification step doesn't false-alarm.
  - NB2 (unequal adapter budgets): Float adapter h=64 (99K params/block), compressed adapter h=128 (165K params/block + 394K proj = 559K/block). The compressed config has ~5.6x more trainable params per block. This is inherited from c378 and is the agenda's design, so not blocking. But the "composition penalty" conflates compression damage with any capacity difference. If the penalty is small, compression cost is genuinely low. If large, it could be partly explained by the compressed adapter still being undertrained despite more params. Note this in the verdict interpretation.
  - NB3 (convergence check): 6 blocks = 2x forward cost but same 2000 steps. With 3.35M trainable params (compressed) vs 595K (float), the compressed config has more params on the same schedule. Plan's verification says "Both configs converge (PPL decreasing over 2000 steps)" — good. Should also log whether PPL is still decreasing at step 2000. If so, the penalty is an upper bound.
  - Dependencies: Item 386 (c378) is CONFIRMED. Block selection [2,1,10,4,3,11] is consistent with all prior 6-block experiments (c357, c371, c374, c375, c376).
  - Thresholds: CONFIRMED < 15, REFUTED > 25 PPL penalty. 3-block test reference = 7.49 (c378). Sublinear would be < 15, superlinear > 25. Range is reasonable with a 10-PPL inconclusive band.
  - Seeds: 3 seeds (42, 137, 2024) matching c378. Sufficient.
  - Pitfall scan: MP3 marginally relevant (compressed has 3.35M params trained on 256K tokens) but mitigated by held-out test eval and checkpoint selection. MP5 not triggered — both configs succeeded at 3-block scale.
  - Float baseline sanity check (56.52 +/- 0.5): Good. Catches data loading or tokenization regressions.
