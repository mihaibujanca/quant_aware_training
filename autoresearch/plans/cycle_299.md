# Cycle 299 Plan — Item 309: Cycle 280 retrain decomposition (B drift)

## Item 309 (verbatim Hypothesis)
B(t) drifts substantially — row-space cosine ≤ 0.6 with the SVD initialization Vh[:2, :] by step 4000. The Adam optimum is *not* the SVD truncation refined; it's a different rank-2 operator with a column space fixed by the frozen A factor.

## Dependencies
- `experiments/block1_lowrank_cycle280.py` (template: SVD init, `LowRankCProj`, 4000-step Adam retrain, eval loop, block 1 c_proj).
- `transformers.GPT2LMHeadModel("gpt2")`, `wikitext-2-raw-v1` val (same calib/eval slice as cycle 280: 200/100 × 128 tok).

## Approach
1. Copy cycle 280 to `experiments/cycle280_decomp_cycle299.py`. Keep SEED=42 only (single seed per item 309 brief: "Single seed (42 is enough for decomposition)"). Drop the 3-seed retrain loop and the no-retrain sweep — only the rank-2 retrain is needed.
2. Load GPT-2, verify float PPL matches `FLOAT_BASELINE_REF=53.895` (±0.5). SVD block 1 `c_proj.weight` (3072×768) full on CPU float32. Set `A = U[:, :2] * S[:2]` (frozen 3072×2), `B_init = Vh[:2, :]` (2×768). Replace `model.transformer.h[1].mlp.c_proj` with `LowRankCProj(A, B_init, bias)`.
3. Freeze everything except `B` (2×768) and `bias` (768). Adam lr=1e-4, 4000 steps, calib 200 seqs shuffled. Eval PPL every 100 steps on the held-100 slice.
4. **Snapshot B(t) and bias(t) every 100 steps** (step 0, 100, ..., 4000) — keep as CPU float32 tensors in a list.
5. Per snapshot compute: (i) `frobenius_delta_from_init = ||B(t) - B(0)||_F`; (ii) `frobenius_delta_rel = ||B(t) - B(0)||_F / ||B(0)||_F`; (iii) **row-space cosine** between `rowspace(B(t))` and `rowspace(B(0))` = `||Vh0 @ Vh_t.T||_F / sqrt(2)` where `Vh_t` is the right SV basis (2×768, orthonormal rows) of `B(t)` and `Vh0 = Vh[:2, :]`; (iv) implicit composite `W(t) = A @ B(t)` (3072×768); SVD of `W(t)` → top-2 right SVs; (v) cosine of `W(t)`'s top-2 right SVs with the ORIGINAL full-rank `c_proj.weight`'s top-2 right SVs (= `Vh[:2, :]` since A has orthogonal columns scaled by S); record top-2 singular values of W(t); (vi) `bias(t) - bias(0)` L2 norm.
6. Save trajectory JSON with per-step: `{step, eval_ppl, fro_abs, fro_rel, rowspace_cos, W_sv0, W_sv1, W_top2_cos_vs_orig, bias_delta_norm}`.
7. Plot `autoresearch/results/cycle_299_drift.png` — 3 panels: (a) eval PPL vs step, (b) rowspace_cos (B(t), B(0)) vs step on [0, 1] axis with hypothesis threshold 0.6 drawn, (c) Frobenius relative delta vs step.
8. Verdict per item 309 verbatim clauses.

## Files
- New: `experiments/cycle280_decomp_cycle299.py`
- New: `autoresearch/results/cycle_299_decomp.json`
- New: `autoresearch/results/cycle_299_summary.md`
- New: `autoresearch/results/cycle_299_drift.png`
- New: `autoresearch/logs/cycle_299_run.log`
- New: `autoresearch/results/cycle_299_manifest.txt`, `cycle_299_exec_result.txt`, `cycle_299_audit.txt`, `autoresearch/audits/cycle_299.md`
- Updated: `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, `autoresearch/STATUS.md`

## Config
TARGET_BLOCK=1, RETRAIN_RANK=2, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, N_STEPS=4000, LR=1e-4, EVAL_EVERY=100, SEED=42, DEVICE="mps" if available else "cpu", FLOAT_BASELINE_REF=53.895. Trainable = 2·768 + 768 = 2304 (asserted). Trajectory snapshots = 41 (step 0 + 40 eval points).

## Verdict logic (item 309 verbatim)
- **CONFIRMED drift** iff `rowspace_cos(B(4000), B(0)) ≤ 0.6` AND held-out PPL improvement tracks the drift (i.e., PPL decrease is correlated with cos decrease; corr ≤ -0.5 is sufficient proof of tracking).
- **CONFIRMED stay-put** iff `rowspace_cos(B(t), B(0)) ≥ 0.95` for all t ∈ snapshots (the rank-2 weight is essentially the SVD truncation plus a tiny correction).
- **INCONCLUSIVE** otherwise (0.6 < final cos < 0.95 or one clause holds but the correlation check fails).

## Verification
- Assert float PPL within 0.5 of 53.895 at load (cycle 280 convention).
- Assert trainable param count == 2·768 + 768 == 2304.
- Assert init PPL at step 0 matches cycle 280's rank-2 noretrain PPL (57.66) within 0.02.
- Assert `rowspace_cos(B(0), B(0)) == 1.0` as sanity (within 1e-6).
- Assert best eval PPL improves below init PPL (if not, training failed).
- Assert W(t=0) ≈ (U[:, :2] · S[:2]) @ Vh[:2, :] (trajectory seed check).

## Baselines
- Cycle 280 trajectory (init 57.66, best 49.03, 8.6 PPL recovery, seed 42).
- Cycle 280's `Vh[:2, :]` (the SVD-init row space) — the 2D reference subspace.
- Float baseline 53.895 PPL (cycles 266/268/269/270).
- Item 309 threshold 0.6 (drift) and 0.95 (stay-put).

# APPROVE
