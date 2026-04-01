# Design Review — Cycle 189 (Item 199): Cascade Error Direction Analysis

## Summary

Script: `experiments/cascade_error_direction_cycle189.py`
Reviewer: automated design review

---

## Correctness

**R1 — Bias computation dimensionality**: Hook captures `inp[0]` (GELU input z, shape (B*S, 3072))
and `out` (GELU output, shape (B*S, 3072)). Delta = gelu_out - relu_out. Mean_delta shape (3072,).
Projection: `mean_delta @ c_proj.weight` where weight shape is (3072, 768) → result (768,).
This matches `c_proj.bias` shape (768,). **CORRECT** — fixes cycle-181-style dimensionality error.

**R2 — Residual collection**: Hook on `model.transformer.h[l]` captures block output including
residual connection. GPT-2 block returns `(hidden_states, present_key_value, ...)`, so `out[0]`
correctly extracts the hidden state. Shape: (B, S, 768). Reshaped to (N_tokens, 768). **CORRECT**.

**R3 — Error computation**: `e_l = res_float[l] - res_swapped[l]`. Both tensors have same
shape (N_tokens, 768) since same sequences are run through both models in same order. **CORRECT**.

**R4 — Cosine similarity**: Uses `clamp(min=1e-8)` to avoid div-by-zero. Per-token cos then
averaged. No issue with zero errors (rare for full-swap model). **CORRECT**.

**R5 — Effective rank**: Uses `torch.linalg.svdvals` (not deprecated `torch.svd`). Wrapped in
try/except for OOM safety. **CORRECT**.

**R6 — Full-swap model (all 12 blocks)**: This is a diagnostic-only experiment — not a compression
recipe. Using full-swap simplifies the analysis by avoiding 12 separate progressive runs.
Consistent with cycle 184's residual divergence analysis. **ACCEPTABLE**.

---

## Potential Issues

**P1 — Memory pressure**: `res_float` and `res_swapped` each store 12 × (N_tokens, 768) tensors
where N_tokens = 100 × 127 = 12700. Size: 12 × 12700 × 768 × 4 bytes ≈ 470 MB per model.
Total ~940 MB. Should fit on MPS (8GB+ typical). **LOW RISK**.

**P2 — Hook on `mlp.act`**: The hook captures `inp[0]` which is the input to the activation
function (GELU). For GPT-2 MLP: `self.act(self.c_fc(x))`. The `inp[0]` is the output of c_fc
(post-fc1, pre-act). This is correct — it's what should be compared between GELU and ReLU. **CORRECT**.

**P3 — Seed collision**: calib_seqs (seed=1337, N=200) and analysis_seqs (seed=42, N=100) may
share some tokens since both draw from the same corpus. This is acceptable (not train/test split);
bias calibration on slightly overlapping data won't affect the directional analysis. **ACCEPTABLE**.

**P4 — c_proj.weight shape assumption**: Code assumes shape (3072, 768). GPT-2 uses `Conv1D`
which stores weight transposed: `self.weight = nn.Parameter(torch.empty(nx, nf))` where nx=in_features.
For c_proj: nx=3072, nf=768, so weight shape IS (3072, 768). **CORRECT** — consistent with
cycle 182/184 code that made the same assumption.

---

## Verdict Criteria Assessment

- CONFIRMED: mean_cos(e_l, e_{l-1}) > 0.3 in ≥8/12 layers — directional compounding
- REFUTED: all 11 cos values < 0.1 — diffuse, no coherent direction
- INCONCLUSIVE: 0.1–0.3 range

These are reasonable thresholds. Cycle 184 found rank_90=300-434 (diffuse), suggesting errors
are high-dimensional. Low cosine alignment is plausible. The REFUTED criterion is strict (all 11
layers), which is appropriate — a REFUTED outcome would require consistent diffuseness throughout.

---

## Flags

None blocking. No plan changes needed.

**REVIEW RESULT: APPROVED — proceed to execute.**
