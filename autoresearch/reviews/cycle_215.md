# Design Review — Cycle 215, Item 225

**Verdict: APPROVE (with advisory notes)**
**Round: 1**

## R1 — Premise

Sequential block-by-block correction is effectively **coordinate descent** on the same joint objective that cycle 206/210 optimized with joint Adam. Reasoning:
- Block 1's W_out update changes its output residual stream, which changes the inputs seen by block 4 and block 7 on the next pass.
- Fitting block 4 after block 1 is thus fitting to a different (corrected) upstream context.
- If joint Adam found a minimum and held it, coordinate descent cannot find a strictly better minimum on the same loss (it's a subset of the feasible descent directions per step).
- **Where it CAN differ**: the optimization *dynamics*. With only 395K trainable params per step (vs 1.18M), the effective capacity-per-step is lower → slower overfit → early-stop may catch a better generalization point before the training loss drives the eval up. This is the plan's actual premise, and it's defensible.

**Note for interpretation**: a positive result (≤46.0) would indicate that joint Adam overfits faster than sequential, not that sequential finds a fundamentally different basin. The plan should not oversell "different optimum" — the plan correctly frames it as "reduce overfitting by keeping each optimization smaller" which is the right framing.

## R2 — Measurement

- Per-block early-stop snapshot uses eval PPL to select best state. This is an in-method hyperparameter selection, **not data leakage**, because the joint Adam baseline (46.556 at step 4000, cycle 210) was also selected via eval PPL from a step-curve sweep. Apples-to-apples.
- Init PPL ≈ 61.7 sanity check matches cycle 206. Good.
- Per-block param assert (394,752 + 768 = 395,520) in plan. Good.
- Eval cadence (every 100 steps) matches cycle 210.

## R3 — Capacity

- 395K params per block, 200 calib sequences × 128 tokens ≈ 25.6K positions. MP3 (underdetermined fitting) applies in the strict sample-count sense, BUT this matches cycle 206 exactly and that run produced a meaningful result. Parity with the baseline is what matters here.
- Per-block step count 2000 > joint's 1000 → more gradient steps per param, which could actually accelerate overfit rather than delay it. This is why the snapshot+restore-best is essential, and the plan includes it.

## R4 — Method-architecture alignment

- Freeze all params, unfreeze only `h[b].mlp.W_out` + `b_out`, fresh Adam per block: clean.
- Sequential order 1→4→7 is faithful to the brief's spec (a): upstream corrected, downstream still at topk+lstsq init when fitting each block. Plan spells this out.
- CompressedMLP reuse, fit_block_lstsq copy-verbatim: good, avoids re-derivation bugs.

## R5 — Known pitfalls

- **MP1** (threshold above trivial baseline): Not applicable — thresholds are tied to joint Adam best 46.556 and topk+lstsq init 61.746.
- **MP3** (underdetermined fitting): Applies to magnitude but not differentially — same regime as cycle 206 baseline. Parity preserved.
- **MP5** (insufficient capacity): Not applicable — joint baseline demonstrates the method can reach 46.556.
- **MP6** (verdict misclassification): The plan's fallback "if block training pushes PPL WORSE than pre-block state, restore pre-block state and log collapsed" is the correct handling. Good.

## Threshold calibration — advisory concern

The CONFIRMED threshold ≤46.0 sits **0.556 PPL below** the joint Adam best 46.556. That baseline number is from a **single seed** (cycle 210). Cycle 215 uses 3 seeds and compares the MEAN of sequential against the single-seed joint number. If joint Adam across 3 seeds has std of ~0.5-1.0 PPL (plausible for this setup), then:
- Sequential mean 46.0 with std 0.8 overlaps joint's seed-42 value of 46.556 substantially.
- A CONFIRMED verdict on this basis would be noise-dominated.

**Advisory (not blocking)**: the plan should add a note that if sequential mean lands in [45.5, 46.5], the verdict should be reported as "CONFIRMED but within noise of joint Adam — no meaningful improvement" rather than a clean win. The plan's INCONCLUSIVE band [46, 60] is very wide; a narrower "near-parity" band would be more informative. Not blocking because the experiment is still worth running and the JSON records per-seed values so any reader can reinterpret.

## 3 seeds sufficiency

3 seeds is the minimum for a mean±std report and matches the audit requirement. For a binary verdict on a 0.5 PPL margin, 3 seeds is on the thin side but defensible given the 30-min cap. Acceptable.

## Issues

None blocking. Advisory notes only.

## Actionable feedback

1. **Advisory (soft)**: In the results JSON and summary, explicitly state that "CONFIRMED" near the 46.0 boundary should be qualified as "within noise of joint baseline" unless std across seeds is clearly small.
2. **Advisory (soft)**: Consider framing the Verdict message in the JSON to include the per-seed final_ppl vector so the noise level is visible at a glance — the plan already records per-seed values, so this is essentially free.
3. **Nit**: Plan step 5 says "mean ± std" — make sure the code actually prints both, not just mean.
4. **Nit**: Plan step 4d restores best_state before moving to next block, which is correct — confirm the eval at end-of-block-3 uses the restored states.

These are optional polish. The plan is methodologically sound and can proceed as written.
