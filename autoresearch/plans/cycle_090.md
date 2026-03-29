# Task
Attention block compression feasibility scan — GPT-2. For all 12 attention blocks, measure per-head output rank, per-block ablation damage, and single-block low-rank compression at 6×.

# Hypothesis
- **Confirmed if:** `>= 3/12 attention blocks achieve <= +5% single-block swap PPL at 6× (attention is compressible).`
- **Refuted if:** `all blocks > +15% PPL (attention resists projection-based compression).`
- **Inconclusive if:** `1-2 blocks compressible but most are not.`

# Pre-flight Results
- **P1 (duplicate):** PASS. Item 40 (cycle 022) studied head pruning via zeroing. Item 105 tests projection-based compression (full-block V-projection 768→128, lstsq reconstruct) — different method, different criteria.
- **P2 (dependency):** PASS. Item 105 lists "Depends on: none." No skip conditions.
- **P3 (concrete):** PASS. Three well-defined measurements, clear per-block output, one cycle.
- **P4 (program modification):** PASS. This is a diagnostic scan, does not modify program.md. Program §36 says "FFN-only compression first" — this scan evaluates whether to expand scope, does not change the rule.
- **P5 (null baseline):** A random projection at 6× (768→128) would discard ~83% of information. Null baseline: random V-projection + lstsq should yield >+50% PPL per block. Baseline GPT-2 PPL (OWT) ≈ 57.65 from prior cycles.
- **P6 (architecture fit):** Attention has Q/K/V/O projections; full-block compression of the concatenated weight matrices is natural and matches the FFN compression strategy (compress the full weight matrix, not sub-components). Method is appropriate.
- **P7 (statistical power):** 4096 probe tokens for SVD rank is standard (matches prior cycles). PPL measured on full eval set. 12 blocks × 3 measurements = adequate coverage. No SGD, so no seed variance concern for rank/ablation. Use 3 seeds for lstsq (probe sampling).

# Steps
1. Create `experiments/attn_compression_scan.py`. Load GPT-2 via `aleph/models.py` or direct HF.
2. **Per-head output rank** (all 144 heads): Forward 4096 tokens, hook each attention block to capture per-head output (B, T, d_head). SVD each head's output matrix (T×d_head). Report rank at 95% energy.
3. **Per-block ablation** (12 blocks): For each block, zero the entire attention output (all 12 heads), measure PPL on OWT eval split. Report ΔPPL per block. This is faster than per-head ablation (12 forward passes vs 144) and directly relevant to the block-level compression question. Cycle 022 already measured per-head ablation; this adds the block-level view.
4. **Single-block full-matrix low-rank compression at 6×** (12 blocks, 3 seeds): For each block, compress the full concatenated attention weight matrices. Each of W_V, W_Q, W_K, W_O has shape [768, 768] (12 heads × 64 concatenated). Apply rank-128 truncated SVD to each (768/6 ≈ 128). Replace the original weights with the rank-128 approximation. Measure single-block swap PPL. This treats attention the same way as FFN compression: compress the full weight matrix, not per-head. The rank-128 budget may be allocated unevenly across heads implicitly by SVD, which is a feature (SVD finds the best global rank-128 subspace). Use 3 seeds for probe sampling to measure rank stability.
5. Collect results into `autoresearch/results/cycle_090_attn_scan.json`. Generate summary plot: per-block rank heatmap + PPL bar chart.
6. Evaluate against criteria. Update `autoresearch/agenda.md`: move item 105 to PAST EXPERIMENTS with outcome.
7. Commit: `[autoresearch] cycle 090: attention block compression feasibility scan — GPT-2`

# Validation
- Baseline PPL (no modification) must match ~57.65 ± 1.0 (sanity check).
- Ablation of all attention in one block should be significant (>+10% PPL) for at least some blocks — confirms hooks work.
- Random V-projection without lstsq should be worse than SVD compression — confirms reconstruction quality.
- Per-head rank values should be consistent with cycle 022 findings (eff_rank 1–7 in deep layers, up to 104 in early). Values within 2× of cycle 022 at the same layer.

# Artifacts
- `experiments/attn_compression_scan.py`
- `autoresearch/results/cycle_090_attn_scan.json`
- `autoresearch/results/cycle_090_attn_scan.png`

# Fallbacks
- If full-matrix SVD at rank-128 is ill-conditioned or too slow, try rank-192 (4×) as a less aggressive alternative.
- If all blocks show >+15% PPL, report REFUTED and note which blocks are closest to viable.
