# Cycle 164 Plan: Positive-channel ablation — GPT-2 FFN

**Item:** 174
**Script:** `experiments/positive_channel_ablation_gpt2.py` (new)

## Pre-flights

**P1 — Null hypothesis:** The positive regime of GELU is essential; killing it degrades PPL comparably to killing the negative regime (cycle 102: +8320%).

**P2 — Criteria (verbatim):**
- Confirmed if: killing positive channels → >+100% PPL (both regimes essential).
- Refuted if: killing positive channels → <+20% PPL (negative regime carries most computation).
- Inconclusive if: +20–100% PPL delta.

**P3 — Novelty:** Cycle 102 killed negative regime (GELU→ReLU). This is the complement: kill positive regime, keep only the negative leak. No prior cycle or survey paper tests this direction.

**P4 — Confounds:** (1) GPT-2 uses `NewGELUActivation`; must hook the correct module. (2) "Kill positive" means zero the activation output where pre-activation z > 0, not where GELU(z) > 0 (they differ near z ≈ 0). Use z > 0 threshold for clean split. (3) Token-mean replacement (Condition B) must be computed per-sequence to avoid data leakage across sequences.

**P5 — Cost:** 1 baseline + 2 full-model conditions + 24 per-block evals (12 blocks × 2 conditions). ~200 seqs each, ~20 min MPS.

**P6 — Data:** OWT eval split (`data/openwebtext_sample.txt`), same config as cycle 102.

**P7 — Abort:** Baseline PPL outside [15, 65]. Model loading fails.

## Implementation

1. **Baseline.** Load `gpt2`, eval PPL on OWT (200 seqs, 1024 tokens), seed 42. Verify PPL ~22.

2. **Hook design.** For each `model.transformer.h[i].mlp`: wrap `act` to intercept pre-activation `z` from `c_fc`. Two modes:
   - **Condition A (zero):** `out = torch.where(z > 0, torch.zeros_like(z), gelu(z))`
   - **Condition B (mean):** `out = torch.where(z > 0, gelu(z).mean(dim=1, keepdim=True).expand_as(z), gelu(z))`

3. **Full-model ablation.** Apply hook to all 12 blocks. Eval PPL for A and B. Report absolute and relative delta.

4. **Per-block ablation.** For each block i (0–11): apply hook to block i only, eval PPL (Condition A only). Produces 12-row sensitivity table.

5. **Diagnostics.** On Condition A full-model: log per-block `||y_pos||` vs `||y_neg||` (norm of zeroed vs kept contributions) to quantify signal magnitude split.

6. **Verdict.**
   ```
   delta = (ppl_A_full - ppl_base) / ppl_base * 100
   if delta > 100: CONFIRMED
   elif delta < 20: REFUTED
   else: INCONCLUSIVE
   ```

7. **Artifacts.** `autoresearch/results/cycle_164_positive_ablation.json` (all PPLs, norms), `autoresearch/results/cycle_164_positive_ablation.png` (per-block sensitivity bar chart + magnitude comparison).
