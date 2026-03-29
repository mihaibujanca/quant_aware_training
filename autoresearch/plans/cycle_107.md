# Cycle 107 Plan: Output-Space Direct Regression Bypass — GPT-2 FFN

**Item:** 117
**Script:** `experiments/direct_regression_bypass_gpt2.py` (new)
**Revision:** 1 (addresses I1, A1, A2)

## Pre-flights

**P1 — Null hypothesis:** Direct low-rank regression cannot capture GELU non-linearity; topk+lstsq outperforms at matched param count in all blocks.

**P2 — Criteria:** Confirmed if direct rank-r beats topk+lstsq in >=2/4 blocks. Refuted if topk wins by >=5% PPL everywhere. Else inconclusive.

**P3 — Novelty:** First bypass of the hidden layer entirely — all prior cycles preserved GELU structure.

**P4 — Confounds:** (1) Both methods use same 4096 probe tokens and same eval split. (2) Bias: both get one bias vector. (3) Seeds control probe token sampling (see step 1).

**P5 — Cost:** ~10 min (4 blocks × 5 ranks × 2 methods × 3 seeds, ~3s/eval).

**P6 — Data:** `data/openwebtext_sample.txt` (same as cycles 089+).

**P7 — Abort:** Baseline PPL outside [25, 65]. NaN in any lstsq solve.

## Implementation

1. **Baseline + seeds.** Load GPT-2, eval PPL (200 seqs, len=128, batch=4). Seeds [42, 123, 7] control which 4096 tokens are sampled for probe fitting — lstsq is deterministic given data, so seed variation measures sensitivity to probe sample.

2. **Collect per-block I/O.** For blocks [1, 4, 7, 11]: hook FFN input x and output y. Collect 4096 tokens per seed. Store (X, Y) shape (4096, 768).

3. **Direct rank-constrained regression (I1 fix).** For each block, for r in {32, 64, 128, 256, 384}:
   - SVD of input data: U_x, S_x, V_x = svd(X). Keep top-r components: X_r = X @ V_x[:, :r] (shape 4096×r).
   - Lstsq: solve Y = X_r @ A^T + b (A is 768×r). Append ones column to X_r, solve directly.
   - Reconstruct full-rank map: W = A @ V_x[:, :r]^T (768×768, rank r). Replace FFN with nn.Linear using W and b.
   - Record reconstruction error: ||Y - X_r @ A^T - b||_F / ||Y||_F.
   - Eval PPL.

4. **Topk+lstsq baseline.** For each block, k in {32, 64, 128, 256, 384} (param-matched: 1536k = 1536r):
   - Standard: probe_ffn_geometry, derive_ffn_compression at target k, swap_compressed_ffns.
   - Record reconstruction error: ||Y - Y_hat||_F / ||Y||_F using same probe X, Y.
   - Eval PPL.

5. **Verdict.**
   ```
   For each (block, r/k): delta = (ppl_direct - ppl_topk) / ppl_topk * 100
   wins = blocks where any r achieves delta <= 0
   wins >= 2:              CONFIRMED
   all deltas > 5%:        REFUTED
   else:                   INCONCLUSIVE
   ```

6. **Output.**
   - JSON: `autoresearch/results/cycle_107_direct_regression.json` — per-(block, rank, method, seed): PPL, reconstruction error, mean/std, verdict.
   - Plot: 4 subplots (per block), x=param count, y=PPL, two lines (direct vs topk), error bars from seeds. Second row: same layout but y=reconstruction error. Save `autoresearch/results/cycle_107_direct_regression.png`.

7. **Agenda update + commit** per standard flow.
