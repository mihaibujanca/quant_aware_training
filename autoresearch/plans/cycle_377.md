# Cycle 377 — Item 385: Adapter-only baseline (beat-float anomaly)

## Pre-flight

### P1: Duplicate check — PASS
No prior cycle tests adapters on uncompressed blocks. All adapter experiments (c352-c367) pair adapters with topk+lstsq compression. Searched PAST for "adapter.*uncompressed", "float.*adapter", "adapter.*only" — no matches.

### P2: Dependency check — PASS
No dependencies. Uses standard GPT-2 + wikitext-2 setup.

### P3: Falsifiability — PASS
Confirmed: PPL < 50.0. Refuted: PPL > 53.0. Inconclusive: [50, 53].

### P4: Cost estimate — PASS
3 seeds × 2000 steps × ~0.15s/step ≈ 15 min on MPS. Single model load per seed.

## Review responses

### R2 (param fairness) — RESOLVED
The review assumed the float adapter takes 3072-dim GELU hidden as input. This was wrong — looking at the actual `MLPAdapter` class in `gpt2_adapter_sweep_cycle359.py`, the adapter takes `d_model=768` (the post-FFN block output), not the GELU intermediate. The `CompressedMLPWithAdapter.forward()` calls `self.compressed(x)` (returns 768-dim), then `self.adapter(out)` on that 768-dim output. The float adapter takes the identical 768-dim post-FFN output. With the same `d_model=768` and `h=64`, both have identical param counts: `1537*64 + 768 = 99,136` per block, `297,408` total for 3 blocks.

Note: the compressed reference PPL 47.43 (cycle 367) used `h=128` (198,400 params/block, 595,200 total), so the float adapter with h=64 actually has **fewer** params than the compressed reference. This is conservative — if float+adapter still beats float with fewer adapter params, the conclusion is stronger.

### R4 (adapter architecture) — RESOLVED
The agenda item explicitly specifies h=64. The compressed reference (cycle 367) used h=128, but this plan intentionally uses a smaller adapter (h=64, ~297K params) vs the compressed reference (h=128, ~595K params). This is a conservative test: if even a smaller adapter on float FFNs can beat float baseline, the adapter effect is robust. If it cannot, the comparison is still informative — compression + bigger adapter beats float, but the confound remains (could be adapter capacity, bottleneck, or both). The verdict thresholds (50/53) account for this.

### R3 (train/eval overlap) — ACKNOWLEDGED
Non-blocking. Same protocol as all prior adapter cycles.

## Plan

### Script: `experiments/gpt2_adapter_only_cycle377.py`

Fork structure from `gpt2_adapter_schedule_cycle367.py`. Replace `CompressedGPT2MLPWithAdapter` with a new `FloatGPT2MLPWithAdapter` class.

1. **Config**: `BLOCKS = [2, 1, 10]`, `ADAPTER_HIDDEN = 64`, `LR = 1e-4`, `N_STEPS = 2000`, `SEEDS = [42, 137, 2024]`, `N_CALIB = 200`, `N_EVAL = 100`, `SEQ_LEN = 128`, `FLOAT_BASELINE_REF = 53.895`. Output: `autoresearch/results/cycle_377_adapter_only.json`.

2. **FloatGPT2MLPWithAdapter** (new class):
   - Wraps the original GPT-2 MLP (c_fc, c_proj as frozen buffers).
   - `forward(x)`: `out = c_proj(gelu(c_fc(x)))`, `out = adapter(out)`, return `out`.
   - Adapter: `MLPAdapter(d_model=768, h=64)` — same class as cycle 359. Takes 768-dim post-FFN output, applies `x + W2 @ relu(W1 @ x + b1) + b2`, returns 768-dim.
   - W1: (64, 768), b1: (64,), W2: (768, 64), b2: (768,). Init: `randn * 0.01`.
   - Params per block: `64*768 + 64 + 768*64 + 768 = 99,136`. Total 3 blocks: `297,408`.
   - GPT-2 c_fc is Conv1D (weight shape (768, 3072)). Extract: `fc_w = c_fc.weight.T` → (3072, 768), `fc_b = c_fc.bias`. Same for c_proj: `proj_w = c_proj.weight.T` → (768, 3072).

3. **Float baseline**: Load GPT-2, eval PPL, assert within ±0.5 of 53.895.

4. **Per-seed loop** (3 seeds):
   - Load fresh GPT-2. For each block in BLOCKS, extract c_fc/c_proj weights, create `FloatGPT2MLPWithAdapter`, replace `model.transformer.h[blk].mlp`.
   - Freeze all params. Unfreeze only adapter params (W1, b1, W2, b2) across 3 blocks.
   - Print trainable param count (expected: 3 × 99,136 = 297,408).
   - Train 2000 steps with `run_steps()` (Adam lr=1e-4, eval every 100 steps, best-checkpoint).
   - Record best PPL per seed.

5. **Output JSON**:
   ```json
   {"float_ppl": ..., "compressed_adapter_ref_ppl": 47.43,
    "compressed_adapter_ref_h": 128, "compressed_adapter_ref_params": 595200,
    "adapter_only_h": 64, "adapter_params_per_block": 99136,
    "total_adapter_params": 297408,
    "adapter_only_results": [{"seed":42,"best_ppl":...}, ...],
    "mean_ppl": ..., "std_ppl": ...,
    "verdict": "CONFIRMED/REFUTED/INCONCLUSIVE",
    "note": "Float adapter uses h=64 (297K params) vs compressed ref h=128 (595K params). Conservative comparison."}
   ```

6. **Verdict**: mean PPL across 3 seeds. < 50 → CONFIRMED, > 53 → REFUTED, else INCONCLUSIVE.

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_adapter_only_cycle377.py
```

### Verification
- Float PPL = 53.89 ± 0.5.
- Init PPL (before training) should equal float PPL ± 0.1 (adapter starts near-zero).
- Trainable params = 297,408 (3 × 99,136).
