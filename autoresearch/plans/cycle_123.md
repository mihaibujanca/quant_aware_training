# Task
Cycle 123, item 140: QK vs V error decomposition under FFN compression — GPT-2.
Decompose attention output error into QK-space (pattern change) vs V-space (value change) for k=4 compressed model.

# Hypothesis
FFN compression corrupts the residual stream, which feeds into attention as Q, K, V inputs. The error in attention output can be attributed to pattern distortion (QK) or value distortion (V). One may dominate, providing a surgical correction target.

Criteria (verbatim from agenda):
- **Confirmed if**: one component (QK or V) accounts for >70% of total error in >=8/12 layers (clear correction target).
- **Refuted if**: roughly equal (40-60% each) across all layers (no single target).
- **Inconclusive if**: dominant source varies by layer position.

# Pre-flight Results
- P1 PASS: Not a duplicate. Cycle 104 decomposed cascade into attn-vs-FFN sublayer; this decomposes WITHIN attention into QK vs V.
- P2 PASS: No dependencies. Item text says "Depends on: none."
- P3 PASS: Concrete methodology, well-defined metrics, single-cycle scope.
- P4 PASS: No program.md modifications needed.
- P5 (null baseline): If FFN compression corrupted Q,K,V identically (uniform noise), QK and V fractions would be ~50/50. Random baseline = 50% each.
- P6: GPT-2 uses GELU in FFN, but this experiment only analyzes attention sublayer outputs, not GELU. Method is appropriate.
- P7: 3 seeds × 2048 tokens × 12 layers × 12 heads = 882K measurements. Adequate.

# Steps
1. **Create** `experiments/qk_v_decomposition_gpt2.py`. Structure follows `cascade_error_source_gpt2.py` pattern.
2. **Establish k=4 baseline.** Compress blocks [1,4,7,11] at 6x (topk+lstsq+500 Adam polish). Record baseline PPL and compressed PPL. Use `aleph.lm_geometric_compression` API (same as cycles 104-121).
3. **Run 3 seeds [42, 123, 7].** Each seed samples a different 2048-token OWT batch. All measurements below are repeated per seed.
4. **For each of 12 attention layers, for each of 12 heads, on each seed's 2048 OWT tokens:**
   - Run float model forward → save float Q, K, V, attn_weights (A_float = softmax(QK^T/sqrt(d))), attn_output (O_float = A_float @ V).
   - Run compressed model forward → save compressed Q', K', V', A_comp, O_comp.
   - **Total error**: E_total = ||O_comp - O_float||^2 (per-head, mean over tokens).
   - **V-only error**: Hold attention pattern fixed at A_float, swap V to V': E_V = ||A_float @ V' - A_float @ V||^2.
   - **QK-only error**: Hold V fixed at V_float, use compressed attention pattern: E_QK = ||A_comp @ V - A_float @ V||^2.
   - **Interaction**: E_inter = E_total - E_QK - E_V. (Can be positive or negative.)
   - Fractions: QK_frac = E_QK / E_total, V_frac = E_V / E_total, inter_frac = E_inter / E_total.
   - **KL divergence**: KL(A_float || A_comp) = sum_j A_float[j] * log(A_float[j] / (A_comp[j] + 1e-10)) per query position, mean over positions. Clamp A_float with 1e-10 before log. Report per head per layer.
5. **Aggregate and plot.**
   - Per-seed: compute per-layer means (across heads) of QK_frac, V_frac, inter_frac, KL.
   - Cross-seed: report mean and std across 3 seeds.
   - Heatmaps (seed-averaged): (a) 12×12 QK_frac by layer×head, (b) 12×12 KL by layer×head, (c) per-layer stacked bar of QK_frac/V_frac/inter_frac.
   - Save JSON with all per-head per-layer per-seed numbers.
6. **Verdict logic.** For each seed independently, count layers where max(QK_frac, V_frac) > 0.70 (layer-mean across heads) AND the dominant component is the same. A layer counts toward CONFIRMED only if the >70% threshold holds in ALL 3 seeds. Final count = layers where all 3 seeds agree on >70% dominance by the same component.
   - Count >= 8 AND same component dominates: **CONFIRMED**.
   - All layers have both in [0.40, 0.60] in all seeds: **REFUTED**.
   - Else: **INCONCLUSIVE**.
7. **Commit** with message: `[autoresearch] cycle 123: QK vs V error decomposition — GPT-2 FFN k=4`
8. **Update agenda.md**: move item 140 from NEXT to PAST EXPERIMENTS with commit hash, outcome, and artifact paths.

# Validation
- Sanity: E_QK + E_V + E_inter ≈ E_total (within float32 tolerance <1e-4 relative).
- Compressed PPL should be in the +5-15% range for k=4 (if wildly off, compression setup is wrong).
- Layer 0 check: E_total ≈ 0 at layer 0 only (residual stream is clean before first compressed block at layer 1). Layers 2-11 may have nonzero error even if uncompressed, because they receive a corrupted residual stream from upstream compressed blocks.
- KL values must be non-negative and finite for all heads.
- Cross-seed std of per-layer fracs should be < 0.15; if not, flag instability.
- If |inter_frac| > 0.3 in many layers, the additive decomposition is weak — flag in results.

# Artifacts
- `experiments/qk_v_decomposition_gpt2.py`
- `autoresearch/results/cycle_123_qk_v_decomposition.json`
- `autoresearch/results/cycle_123_qk_v_heatmap.png`
- `autoresearch/results/cycle_123_summary.png`

# Fallbacks
- If k=4 [1,4,7,11] doesn't reproduce expected ~+5-10% PPL, fall back to [7,5,4,1] (cycle 115 block set).
- If interaction term dominates (>30%) making additive decomposition unreliable, report the raw numbers but flag INCONCLUSIVE and suggest multiplicative decomposition for follow-up.
- If KL is numerically unstable (inf/nan), increase epsilon clamp to 1e-8 and re-run.
- If hook-based extraction fails, use manual forward pass through `model.transformer.h[i].attn` weight matrices directly.
