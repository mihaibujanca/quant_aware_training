# Task
Implement adaptive k-per-block allocation for 6-block GPT-2 compression. Allocate k proportional to Frobenius error from cycle 362, with k_min=256 floor. Hold total budget at 3072 (= 6x512 uniform). Compare PPL against uniform-k baseline (65.54).

# Hypothesis
Adaptive k breaks the 6-block composition barrier by giving high-error blocks more channels.
- **CONFIRMED**: 6-block adaptive PPL < 60.0 (mean over 3 seeds)
- **REFUTED**: 6-block adaptive PPL >= 60.0 but <= 65.54 (better than or equal to uniform, not enough)
- **INCONCLUSIVE**: N/A (binary threshold)
If adaptive PPL > 65.54 (worse than uniform), conclude adaptive allocation actively hurts; auto-fallback to sqrt-proportional allocation within the same run.

# Steps
1. **Create** `experiments/gpt2_adaptive_k_cycle374.py` based on `gpt2_progressive_compression_cycle370.py`.
2. **Compute adaptive k** from Frobenius norms (cycle 362 JSON) with k_min=256 floor:
   - Raw proportional: b1=231, b2=273, b3=304, b4=310, b10=905, b11=1049.
   - Apply floor: b1=256, b2=273, b3=304, b4=310, b10=905, b11=1049. Freed budget = 256-231 = 25.
   - Redistribute 25 to b10 and b11 proportional to their Frobenius: b10=905+11=916, b11=1049+14=1063. Verify sum=3072.
   - Also compute sqrt-proportional allocation (k_i ~ sqrt(Frobenius_i), same floor, same budget) as automatic fallback.
3. **Pipeline** (same as progressive but all 6 blocks in one phase):
   - Load GPT-2, collect calibration data (200 seqs, 128 tokens).
   - For each block in [b2, b1, b10, b4, b3, b11]: topk+lstsq compress with block-specific k, init adapter h=64.
   - **Log per-block reconstruction Frobenius** (||W_compressed - W_original||_F) for each block, for both adaptive and uniform runs. Print and save to results JSON.
   - Joint training: 1500 steps, Adam lr=1e-4, train proj_weight + adapter params for all 6 blocks. Eval every 100 steps.
   - Report best PPL across training.
4. **Run 3 seeds** (42, 137, 2024). Report mean +/- std PPL.
5. **Also run uniform k=512 control** with identical pipeline (same script, flag `--uniform`) for fair same-run baseline. Log per-block Frobenius for uniform too.
6. **Verdict logic** (after adaptive run):
   - If mean adaptive PPL < 60.0 → CONFIRMED.
   - If mean adaptive PPL in [60.0, 65.54] → REFUTED (not enough gain).
   - If mean adaptive PPL > 65.54 → REFUTED-WORSE. Automatically run sqrt-proportional allocation (3 seeds) as fallback. Report both results.
7. **Save** results JSON to `autoresearch/results/cycle_365_adaptive_k.json` with per-seed PPL, per-block k values, per-block reconstruction Frobenius (adaptive + uniform), and verdict.
8. **Update** `autoresearch/agenda.md`: move item 374 from NEXT to PAST EXPERIMENTS with verdict, commit hash, artifact paths.
9. **Commit**: `[autoresearch] cycle 365: adaptive k per block — {CONFIRMED/REFUTED} (adaptive PPL {X} vs uniform 65.54)`

# Validation
- Verify sum of adaptive k values = 3072 (budget-neutral). Same for sqrt-proportional.
- Verify no block has k < 256 (k_min floor enforced).
- Verify float baseline PPL ~53.9 (sanity check model loaded correctly).
- Verify uniform control PPL ~65 (reproduces cycle 361 result).
- Verify 3 seeds ran and all produced finite PPL values.
- Verify per-block Frobenius is logged for every block in both adaptive and uniform runs.

# Artifacts
- `experiments/gpt2_adaptive_k_cycle374.py`
- `autoresearch/results/cycle_365_adaptive_k.json`
- `autoresearch/results/cycle_365_manifest.txt`

# Fallbacks
- If adaptive PPL > 65.54 (REFUTED-WORSE), sqrt-proportional run triggers automatically (step 6).
- If training diverges at 1e-4, reduce lr to 5e-5 and increase steps to 2000.
- If MPS OOM with 6 blocks, reduce N_CALIB to 100 or SEQ_LEN to 64.
