# Cycle 057 Audit — Failure Decomposition: Channel Selection vs lstsq Refit vs Stacking

**Item**: 73 — Failure decomposition: channel selection vs lstsq refit vs stacking — GPT-2 FFN
**Claimed verdict**: INCONCLUSIVE
**Audit verdict**: INCONCLUSIVE — agree with claimed verdict; stacking isolation is valid but single-block means are statistically insignificant

## Guardrail Flags

| Check | Status | Note |
|-------|--------|------|
| G1: Implementation correctness | PASS | Variant B correctly uses Shk channels + OWT activations for lstsq refit. Variant C truly stacks all 3 blocks. Probe budget 4096 tokens at k=128 gives 32:1 overdetermination — adequate. |
| G2: Reproducibility (3 seeds) | **WEAK** | 3 seeds present, but per-seed variance is extreme (std ~11% vs mean ~4% for A/B). None of the single-block means are statistically distinguishable from zero. |
| G3: Results interpretation | **FLAG** | Variant C reports identical PPL for all 3 "blocks" (69.29) because it evaluates one model with all 3 blocks swapped. Per-block stacking overhead differences (23.5–25.8pp) are just reflections of different A baselines, not independent measurements. The manifest phrasing could mislead. |
| G4: Counterintuitive results | PASS | Jaccard ~0.5 with zero PPL sensitivity is correctly explained: lstsq refit absorbs channel identity differences. Geometrically sound — top-k channels from different corpora span similar subspaces, so the projection compensates. |
| G5: Method failure vs genuine | PASS | Stacking as sole failure mode is a genuine finding. Single-block negative PPL is the known lstsq regularization effect (6x compression implicitly regularizes the output projection). |

## Key Issues

1. **Statistical power is insufficient.** With std ~11% and only 3 seeds, the single-block means (-3.3% to -5.6%) cannot be distinguished from zero. The INCONCLUSIVE verdict is correct for this reason alone. More seeds (>=10) or longer eval sequences would be needed.

2. **Variant C is one measurement, not three.** The JSON and manifest report C results per-block, but there is only one stacked model per seed. The three identical rows (ppl_mean=69.29 for blocks 7, 5, 4) make this obvious in the data but the manifest's per-block stacking overhead framing obscures it.

3. **The stacking isolation is the real result.** The gap between A (~-4%) and C (+20%) is robust: ~24pp difference, well above noise. Stacking is confirmed as the dominant failure mode. Channel selection and lstsq quality are ruled out as bottlenecks.

4. **Negative single-block PPL is expected.** At 6x compression, lstsq regularization can improve PPL. This is not a finding; it is a known artifact of the compression method when applied to over-parameterized layers.

## Recommendations

- Accept INCONCLUSIVE verdict.
- The stacking finding is actionable even without a formal pass: future work should focus on sequential/iterative compression (compress block N, re-probe block N-1 through the compressed block N, etc.) rather than improving per-block channel selection or refit quality.
- If revisiting, increase seeds to >=10 or eval sequences to >=1000 to reduce PPL variance.
