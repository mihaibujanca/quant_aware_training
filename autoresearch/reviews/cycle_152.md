# Review — Cycle 152 (Item 162)

## Checklist
- [x] Blocks [1,4,7,11] in upstream order matches forward-pass causality
- [x] Random init (not SVD) — correct based on cycle 149 findings
- [x] 500 steps/phase × 4 phases = 2000 total steps, same budget as joint training
- [x] Per-phase cosine schedule restart (fresh optimizer state per block)
- [x] Param count 986,112 matches cycle 149 exactly — fair comparison
- [x] Phase-by-phase PPL tracking gives stacking dynamics curve
- [x] Topk baseline included as control
- [x] Seed 42 only — appropriate for screening

## Risks
- **Step budget fairness**: joint training (cycle 149) used 2000 steps across ALL blocks simultaneously. Progressive uses 500 steps per block sequentially = 2000 total steps, but each block only gets 500. This means each block gets 1/4 the gradient updates of joint training. If progressive loses, it could be a budget issue, not a protocol issue. The plan's fallback (1000 steps/phase) partially addresses this. Acceptable for screening — if progressive wins despite the handicap, the result is strong.
- **Phase 1 is just single-block training**: the first phase (block 1 alone) is identical to independent single-block CE training. The progressive protocol only differs from independent training starting at phase 2. This is correct by design, not a bug.
- **No independent stacking re-run**: the plan uses cycle 138's number (74.4) as the independent baseline rather than re-running. Cycle 138 used AE, not linear factorization. A fairer comparison would be independent linear-factorization blocks. However, the topk baseline (70.7) serves as the primary comparator, and the cycle 138 number is used only for the confirmed/refuted threshold. The agenda item's thresholds reference "independent stacking" generically. This is a minor gap — if the result is borderline, an independent linear-factorization baseline should be added.
- **RNG seed offset per phase**: plan uses `SEED + phase_idx`. This means phase 2 sees different batch ordering than phase 1, which is fine. But each phase uses a fresh generator, so the data batches across phases are independent. No issue.

## Verdict
**PASS** — clean sequential adaptation test. The step-budget asymmetry vs joint training is inherent to the protocol and acknowledged. The one gap (no independent linear-factorization stacking baseline) is non-blocking for a screening run.
