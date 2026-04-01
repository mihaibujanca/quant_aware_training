# Cycle 173 — Plan

**Item:** 184. Unified per-block regime ablation table — GPT-2 FFN

## Pre-flight checks

- P1: GPT-2 available (HuggingFace cache). Confirmed by cycles 102/109/164/166.
- P2: OWT sample at `data/openwebtext_sample.txt`. Used by all prior GPT-2 cycles.
- P3: No overlap — cycles 102/109/164 each tested ONE condition. This is the unified 6-condition table.
- P4: Runtime ~10 min (72 forward passes of ~200 seqs each). Well under 30-min cap.

## Design

### Script: `experiments/regime_ablation_table_cycle173.py`

**Data:** 200 sequences x 1024 tokens from OWT (seed=42). Same loader as cycles 102/164/166.

**Hooking strategy:** For each block b and condition c:
1. Hook `model.transformer.h[b].mlp.act` (the GELU activation)
2. First pass: collect stats (mean of positive outputs, mean of negative outputs) across all tokens
3. Second pass per condition: replace activation with modified version, measure PPL

**Conditions implemented as GELU output modifications:**
- C0: no modification (float baseline — compute once)
- C1: `gelu(z) → max(0, gelu(z))` i.e. zero out where z < 0
- C2: `gelu(z) → min(0, gelu(z))` i.e. zero out where z > 0
  - Wait — need to think about this. GELU output is NOT the same sign as z.
  - For z < 0, GELU(z) is small negative (the "bump" region). For z >> 0, GELU(z) ≈ z.
  - "Kill negative regime" = set output to 0 for channels where z < 0. This is ReLU behavior.
  - "Kill positive regime" = set output to 0 for channels where z > 0.
  - Use z (pre-activation) to determine the mask, apply to the output.
- C3: for channels where z < 0, replace gelu(z) with mu_neg (precomputed channel-wise mean of gelu(z) for z<0 tokens)
- C4: for channels where z > 0, replace gelu(z) with mu_pos (precomputed channel-wise mean of gelu(z) for z>0 tokens)
- C5: replace all outputs with mu_total (precomputed channel-wise mean of gelu(z) across all tokens)

**Important:** Hook receives (module, input, output). input[0] = z (pre-activation), output = gelu(z). We need z to determine regime, output to modify.

**Norms reported per block:**
- ||y_pos||: mean L2 norm of positive-regime output (gelu(z) where z>0)
- ||y_neg||: mean L2 norm of negative-regime output (gelu(z) where z<0)
- ||mu_pos||: L2 norm of the positive-regime channel mean
- ||mu_neg||: L2 norm of the negative-regime channel mean
- ||s_pos|| = ||y_pos - mu_pos||: token-dependent positive signal
- ||s_neg|| = ||y_neg - mu_neg||: token-dependent negative signal

**Interaction check:** PPL(C1) + PPL(C2) - PPL(C0) vs PPL(C5) per block.

### Output artifacts
- `autoresearch/results/cycle_173_regime_ablation.json` — full 12x6 table + norms
- `autoresearch/results/cycle_173_regime_ablation.png` — heatmap of delta-PPL% per condition

### Verdict logic
- Count blocks where C3 delta-PPL < +2%: if >= 8 → CONFIRMED
- Count blocks where C3 delta-PPL > +10%: if >= 8 → REFUTED
- Otherwise → INCONCLUSIVE

## Verification
1. Baseline PPL in [15, 65] (sanity)
2. C2 (kill positive) should be catastrophic (consistent with cycle 164)
3. C1 (kill negative) should match cycle 102 per-block pattern
4. C5 (constant) should be worse than C3 and C4
