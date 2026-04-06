# Cycle 220 — Surgical Bump Removal (item 234)

**Verdict: INCONCLUSIVE**

Patched all 12 GPT-2 small MLP activations with two surgical variants to
isolate the functional role of the negative leak vs the positive-zone bump.
Eval: wikitext-2 validation, N=200, SEQ_LEN=128, seed 42.

| Variant | PPL | Δ vs baseline | % vs baseline |
|---|---|---|---|
| Baseline (float GELU) | 45.064 | — | — |
| Cycle 190 hybrid (GELU_neg + ReLU_pos) | — | — | +27.67% (prior) |
| GELU_noneg (clip negative leak) | 2093.296 | +2048.232 | +4545.19% |
| GELU_nobump (linearize positive bump) | 11455.595 | +11410.532 | +25320.89% |

Thresholds (relative to in-cycle baseline): low=×1.10=49.57, high=×1.55=69.85.

**Conclusion.** Both variants collapse the model catastrophically (neither is near-float), so the plan's "clean separation" verdict does not fire. However, there IS a large asymmetry: killing the positive-zone curvature (nobump, PPL 11,456) is ~5.5× more destructive than killing the negative leak (noneg, PPL 2,093). The fine-grained curvature of the positive-zone bump carries more of the functional load than the negative leak, but neither sub-region is linearizable in isolation.
