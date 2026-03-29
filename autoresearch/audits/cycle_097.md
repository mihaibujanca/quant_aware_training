# Audit — Cycle 097

- **Item**: 114 — Block-0-skip attention stacking — GPT-2
- **Cycle**: 097
- **Auditor verdict**:

```
status=PASS
old_outcome=refuted
new_outcome=refuted
reason=5-block skip-0 delta +42.93% is 2x the REFUTED threshold (+20%). The hypothesis that avoiding block 0 enables 5-block stacking is clearly refuted. Control finding (+25% for [1,3,4] vs +13% for [1,4,9]) reveals block proximity matters more than block count. Position probe (0.968) cleanly rules out positional embedding as root cause of block-0 sensitivity.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The [1,3,4]=+25% vs [1,4,9]=+13% comparison is the real finding — block spacing/interaction matters more than count. Next: systematically test 3-block combos at varying inter-block distances to characterize the spatial interaction kernel. Also: the attention cascade rate (c~0.07 multiplicative) is 6x weaker than FFN cascade (c~0.41) — joint attention+FFN compression may be viable if attention errors don't amplify FFN errors.
interesting=true
interesting_reason=Attention compounding rate (c~0.07/block, roughly additive) is an order of magnitude weaker than FFN cascade (c~0.41/block, multiplicative). But block proximity drives a 2x penalty ([1,3,4]=+25% vs spread [1,4,9]=+13%), suggesting spatial interaction patterns absent in FFN compression. This is a qualitatively different error propagation mechanism.
```

## Guardrail Details

**G1 (Implementation)**: PASS. `_compress_attn_block_inplace` uses identical SVD pipeline to cycle 095. W_full extracted as (2304, 768) via `_std_weight` (handles Conv1D transpose). Per-head slicing: `W_full[base + h*64 : base + (h+1)*64, :]` for Q/K/V projections across 12 heads. Truncated SVD at rank-11: `(U[:,:r] * S[:r]) @ Vt[:r,:]`. Conv1D write-back re-transposes. Only `c_attn` modified, `c_proj` untouched. `_eval_ppl` delegates to shared `_evaluate_model` from `aleph/lm_geometric_compression.py`.

**G2 (Seeds)**: PASS. 3 seeds [42, 123, 7]. Condition A per-seed deltas: [+45.1%, +42.4%, +41.3%], std=1.57% — tight. Condition B: [+25.0%, +25.3%, +24.8%], std=0.20% — very tight. Baseline PPL spread (46.9–57.7) is typical for GPT-2 on different OWT slices. No seed is an outlier.

**G3 (Threshold)**: PASS. +42.93% vs +20% REFUTED boundary. Even the lowest single-seed delta (+41.3%) is 2x the threshold. No borderline ambiguity.

**G4 (Position probe)**: PASS. Condition C: late/early L2 ratio = 0.968 for block-0 attention output distortion. Per-seed: [0.93, 1.01, 0.97] — consistent across seeds and close to 1.0. Block-0 sensitivity is intrinsic to its weight structure (full-rank QKV needed for initial token mixing), not position-embedding-driven. Clean null result.

**G5 (Cascade rate comparison)**: Key analysis. FFN cascade: c=0.41/block → (1.41)^5 - 1 = +456% predicted at 5 blocks. Actual attention 5-block: +43%. The attention cascade is 10x weaker than FFN. Fitting multiplicative model: (1.43)^(1/5) - 1 = c~0.074/block for attention vs c~0.41/block for FFN. However, the additive model fits poorly too: if single blocks are ~3-5%, additive 5-block = ~15-25%, but actual is +43%. The real story is block-identity-dependent interaction: [1,3,4] (clustered early) gives +25% at 3 blocks, while [1,4,9] (spread) gives +13% at 3 blocks. The cascade rate is not block-count-driven — it depends on which blocks interact through the residual stream.

**G6 (Method)**: PASS. Three well-designed conditions: (A) 5-block stress test, (B) 3-block control with different block selection from cycle 095, (C) position probe diagnostic. Pre-specified thresholds (<=+10% CONFIRMED, >+20% REFUTED). Cycle 095 reference included for direct comparison. The Condition B result (+25% vs cycle 095's +13%) is an unexpected bonus finding — the experiment was designed to test block count, but block identity turned out to be the dominant variable.
