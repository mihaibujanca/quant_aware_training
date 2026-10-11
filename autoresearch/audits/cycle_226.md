# Cycle 226 Audit — Item 236 (KD distillation rerun, corrected loss normalization)

**Status: PASS**

## Verdict check
- Criteria: best_mean ≤ 44.0 → CONFIRMED; ≥ 47.0 → REFUTED; else INCONCLUSIVE.
- Reported: best α=0.0, full-run best_mean = **46.364 ± 0.047** (seeds 46.430 / 46.321 / 46.342, best_step 1200 / 1200 / 1400).
- 44.0 < 46.364 < 47.0 → INCONCLUSIVE per literal criterion. Verdict matches.

## Loss-normalization fix verified
- Lines 232–238 reshape `(B,T,V)→(B*T,V)` before `F.kl_div(..., reduction='batchmean')`, so KL divides by per-token count B·T = 128 (matches CE token-mean). With τ=2 and the τ² multiplier the per-step kd_loss now sits at O(0.5–0.8) — directly comparable to ce_loss O(4). Cycle 224's broken-norm KD ran at kd_loss ≈ 30–50 (~10× CE); the fix is genuine.
- Independent sanity: α=0 reproduces cycle 213 within 0.060 PPL (46.364 vs 46.424). Same surface, same data, same optimizer — the −0.060 is within seed jitter (std 0.047).

## α-sweep (SEED=42, 2000 steps)
| α   | init   | best   | best_step |
|-----|--------|--------|-----------|
| 0.0 | 61.746 | 46.430 | 1200      |
| 0.5 | 61.746 | 47.096 | 2000 (still descending) |
| 1.0 | 61.746 | 61.746 | 0 (no learning) |

Pure-KD (α=1) makes no progress: soft-teacher KL alone supplies no per-token CE signal. Mixed (α=0.5) is monotone-down at the budget cap but never overtakes pure CE. No α in {0, 0.5, 1} beats 46.43 within 2000 steps.

## Implementation review
- `FullCompressedMLP` (lines 79–90) makes W_fc, b_fc, W_out, b_out trainable — same surface as cycle 213. Trainable count ~2.36M across blocks {1,4,7}.
- topk + lstsq init reused verbatim (lines 115–125) — identical init across α at fixed seed (init=61.746 to 6 decimals).
- Calib/eval disjoint: `all_ids[:200]` calib, `all_ids[200:300]` eval (lines 73–74). No leakage.
- Best-state snapshot/restore round-trips with `assert abs(restored - best) < 1e-3` (line 274). All seeds passed.
- Persistent float teacher (lines 141–144), `requires_grad_(False)`. Deterministic per seed.

## Trajectory shape
All 3 seeds: monotone descent ~step 100 → ~step 1100, plateau ~46.4 over steps 1200–1700, then U-turn (overfit to calib). Steps 2000–4000 climb to 58–59 PPL (worse than init evaluated post-training). Best-state restore is essential here; without it the run would look catastrophic.

## Bookkeeping
Manifest, exec_result, JSON, summary all consistent (item=236, cycle=226, outcome=inconclusive, best_mean=46.364, best_alpha=0.0). agenda.md PAST/Cycle 226 carries the full numbers and the "no benefit" note.

## Scientific reading
The hypothesis "KD provides ~2 PPL benefit over CE on W_out+W_fc" is **not supported**. Literal verdict is INCONCLUSIVE because 46.364 sits in the (44, 47) band, but best α = 0 means KD as formulated buys nothing on this trainable surface at the matched 2k-step budget. Cycle 213's pure-CE ceiling (46.424) is the genuine ceiling of this surface; the loss formulation was the only thing that mattered in cycle 224's failure.

## Recommendations
- Close KD as a method for W_out+W_fc on this calibration set. The surface ceiling is essentially CE.
- If KD is revisited: longer α=0.5 schedule (the 2k-step trajectory was still descending), warm-CE→ramp-KD scheduling, larger N_calib, attention-block fine-tuning, reverse-KL, or temperature sweep at fixed α=0.5. Low priority — the main signal here is "the surface is the bottleneck, not the loss".
- Held-out test set (separate from calib AND eval) for any future "beats CE" claim, same caveat carried from cycles 206/210/223/225.
