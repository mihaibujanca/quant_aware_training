# Audit — Cycle 098

- **Item**: 113 — Joint FFN+attention compression on viable blocks — GPT-2
- **Cycle**: 098
- **Auditor verdict**:

```
status=PASS
old_outcome=inconclusive
new_outcome=inconclusive
reason=INCONCLUSIVE is mechanically correct per verdict logic, but the CONFIRMED threshold was unreachable: plan assumed FFN k=4 = +5% (conflating k=1 single-block with k=4 accumulated), while cycle 089 established FFN k=4 = +56%. Joint k=4 was never going to be <=+20%. The real finding is at k=2: joint +19.38% vs FFN-only +7.37%, meaning attn adds ~12pp in joint context, roughly matching attn-only deltas from cycle 095. Super-additivity at k=4 (1.93x) is real and consistent across seeds.
guardrail_flags=G5_missing_condition
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Missing B_k2 condition (attn-only on blocks [1,4]) prevents decomposing k=2 interaction. Run it to determine whether k=2 joint is sub/super-additive. Also: joint-aware polish (retune W_down with attn compressed) could reduce the k=4 interaction — current polish was FFN-context only.
interesting=true
interesting_reason=Joint damage is 1.93x the additive prediction (consistent 1.65-2.14x across seeds), confirming that FFN and attention errors amplify through the shared residual stream. The mechanism — compressed attention in downstream blocks can't correct accumulated FFN error — matches cycle 089 super-multiplicative finding for FFN-only cascade and extends it cross-modality. The k=2 boundary (+19%) is the actionable deployment result.
```

## Guardrail Details

**G1 (Implementation)**: PASS. Code structure is clean: per-block FFN polish on single-block models, then `_run_condition` does deepcopy/swap + inject polished weights + in-place attn SVD + eval. No double-compression bug: FFN modifies `c_fc`/`c_proj` while attn modifies `c_attn` — different weight matrices even on shared blocks 1 and 4. Attn SVD uses same `_compress_attn_block_inplace` from cycle 095 (per-head rank-11 truncation on Q/K/V). Polish is held constant across conditions A and C, isolating the attn interaction effect.

**G2 (Seeds)**: PASS. 3 seeds [42, 123, 7]. Condition A per-seed PPLs: [95.95, 72.03, 74.08] — exact match with cycle 089 k=4. Condition C per-seed: [158.15, 97.84, 105.98], all consistently super-additive. Interaction ratio per seed: [2.14x, 1.65x, 1.88x] — seed 42 is highest but not an outlier (it also has highest baseline). Std/mean for joint = 22.1% — high variance driven by baseline variance, not instability.

**G3 (Threshold)**: PASS (mechanically). CONFIRMED requires joint k=4 <= +20%: actual +133.55%, clearly fails. REFUTED requires joint k=2 > +20%: actual +19.38%, fails by 0.62pp. The INCONCLUSIVE verdict follows from the logic. However, the CONFIRMED threshold was miscalibrated from the start — see G6.

**G4 (Additive prediction)**: PASS. Additive prediction = ppl_A + ppl_B - baseline = 87.44. Formula is correct (verified algebraically). This is the standard way to compute expected PPL under independence. Joint actual = 120.66, so interaction = +33.21 PPL = +64.29% of baseline. The super-additivity is unambiguous (well outside noise).

**G5 (Missing condition)**: FLAG. No B_k2 condition (attn-only on blocks [1,4]) was run. The experiment has C_k2 = joint k=2 (+19.38%) and A_k2 = FFN k=2 (+7.37%), but without isolated B_k2 we cannot compute the additive prediction for k=2 and cannot determine whether the k=2 interaction is also super-additive. This is important because k=2 is the deployment-viable regime.

**G6 (Plan calibration)**: The plan stated "FFN-only was +5%" as the basis for the +20% CONFIRMED threshold. This was k=1 (single block 7) from cycle 089, which was actually -3% (below baseline due to noise). The actual k=4 FFN damage was +56%, making the +20% CONFIRMED threshold impossible to reach. The manifest correctly flags this miscalibration. The experiment still produced useful data despite the flawed threshold.
