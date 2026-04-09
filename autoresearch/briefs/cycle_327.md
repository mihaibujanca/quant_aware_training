# Cycle 327 Brief: LASER on parameter-golf b3

## Item
340 — rank-5 weight-SVD + B-factor retrain on `blocks.3.mlp.proj.weight` (13L_640d_322 checkpoint)

## Context
**Hypothesis**: cycle 280's LASER recipe (GPT-2 b1, rank-2 SVD + retrain) transfers to parameter-golf b3 (ReLU² network, different architecture, different rank profile). Cycle 318 measured `eff_rank(b3)=4.89` with top-1 eigenvalue fraction 78.22% — a rank-5 truncation should preserve task performance.

**What this tests**: whether low-rank MLP bottlenecks are a generic compression mechanism (geometry of MLP outputs) vs artifact specific to GELU/GPT-2. Parameter-golf uses ReLU², variable mlp_mult per block, 640d embeddings, trained on parameter budget. If b3's rank-5 behavior holds, ~1.23M params → ~9.6K (128× compression), saving 1.2MB of the 16MB artifact budget.

## Dependencies
- **Item 335** (cycle 322): REFUTED-checkpoint-specific — b3 is 13L-specific, not universal
- **Item 336** (cycle 323): REFUTED-scale-matters — post-mlp_scale eff_rank shows b11=2.25, b12=1.03 as stronger, but refutation is about what is "stronger" for OTHER purposes; b3=4.01 is still valid for THIS experiment
- Cycle 318 (eff_rank census): confirmed eff_rank(b3)=4.89 ✓
- Cycle 319 (curvature residual): confirmed rank_95(Δy_relu²−relu)=72 at b3 ✓
- Cycle 280 GPT-2 result: rank-2 LASER on b1 → PPL 49.03 vs float 53.90 (baseline)

## Key Files
- `autoresearch/cross_project/parameter_golf.md` — subprocess invocation pattern (MUST read before exec)
- `parameter-golf/docs/EFFRANK_CURVATURE_FINDINGS.md` section 6.1 — action item description
- `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz` — target checkpoint
- `parameter-golf/local_scripts/analyze_checkpoint.py` — loader utilities (reuse)

## Constraints
- **Cross-project subprocess**: must invoke parameter-golf .venv (MLX) from qat (PyTorch). Cannot import MLX in-process.
- **No bias on MLPs**: parameter-golf mlp.fc and mlp.proj are bias-free; retrain does not add bias.
- **Variable mlp_mult**: b3 has hidden=1920; b12 has hidden=320. Do NOT assume fixed dims.
- **Metric**: BPB (bits per byte) on FineWeb val. Baseline from 13L_640d_322 training logs. Hypothesis: rank-5 retrain BPB ≤ baseline + 0.001.

## Handoff
Write `parameter-golf/local_scripts/pgolf_laser_retrain.py` and `experiments/pgolf_laser_retrain_cycle340.py`. Test ranks 2, 5, 10. Output: JSON results + markdown summary with BPB-vs-rank table.
