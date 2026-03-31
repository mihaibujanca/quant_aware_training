# Review — Cycle 153 (Item 163)

## Checklist
- [x] Corrections trained on frozen joint model, not co-trained (cycle 142/151 lesson)
- [x] Same correction protocol as cycle 115: LoRA r=8, 2000 CE steps, Adam lr=1e-4
- [x] Topk+corrections control included (reproduces cycle 115 as within-run baseline)
- [x] Same hook point (transformer block output) as cycle 115 and 151
- [x] LowRankCorrection zero-init U — identity at init, corrections start from zero
- [x] Seed 42, 200 eval sequences — matches prior cycles
- [x] Param accounting: 49K corrections on top of 986K joint model — fewer total params than topk+corrections
- [x] Recovery metric defined consistently with cycle 115

## Risks
- **Joint training IS polish**: The plan correctly notes this tension. Joint training optimizes CE end-to-end for 2000 steps — functionally equivalent to a very long polish. Cycle 142 showed corrections on polished model worsened PPL (71→88.7). The critical difference: cycle 142 polished a topk+lstsq init for 500 steps, while joint training starts from random init and trains 2000 steps. These may have very different loss landscapes at convergence. If corrections fail on the joint model, this is an informative negative (polish-like models resist post-hoc correction), not a bug. The fallback (lower lr, fewer steps) is appropriate.
- **Baseline mismatch**: Cycle 115 used 3 seeds (PPL 59.11 ± 7.46); this cycle uses seed 42 only. The within-run topk control addresses this — both arms use the same seed and eval set, so the relative comparison is clean even if single-seed.
- **Joint model reproduction**: Retraining should be deterministic at seed 42, but GPU nondeterminism may cause ± 0.5 PPL drift. The 1.5 tolerance is adequate.

## Verdict
**PASS** — clean A/B test. The cycle 142 parallel (corrections may fail on trained models) is the real scientific risk, but that risk is inherent to the question and the fallback is sensible.
