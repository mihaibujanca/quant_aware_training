# Item
128: Train MoE testbed + REAP expert pruning baseline — Tiny Mixtral

# Context
- Pivot to MoE compression (after 116 cycles on GPT-2 FFN quantization/correction)
- Establishes expert pruning error landscape before geometric correction (items 129–131)
- REAP scoring: gate weight × expert activation norm (production baseline, arXiv:2510.13999)
- Depends on no prior items (item in P0, first of MoE sequence)

# Dependencies
- None (first MoE item)
- Transformers already has MixtralConfig/MixtralForCausalLM available
- WikiText-2 dataset available via `datasets` library

# Key Files
- No existing code; will create new experiment file
- Library access: `aleph/datasets.py` for data loading
- Reference: transformers Mixtral implementation (in .venv/lib/site-packages)

# Constraints
- **CPU only** (MPS has histogram_mps bug on int tensors in grouped_mm, 8x slower)
- Model: 15.5M params, 4 layers, 8 experts/layer, top-2 routing
- Train 10M tokens (14 min estimated), batch=4, seq_len=256, AdamW lr=1e-4
- Calibration: 256 WikiText-2 val sequences for REAP scoring
- Validation: expect PPL < 50 on WikiText-2 val (if not, scale to 20M tokens)
- Pruning levels: 25%, 50%, 75% (2/8, 4/8, 6/8 experts per layer)
- Zero out pruned logits before softmax (renormalize gates)

# Key Outputs
- Trained checkpoint (save)
- Per-layer saliency distributions: Gini, top-4 vs bottom-4 spread, cross-layer correlation
- PPL baseline + per-pruning-level (25/50/75)
- Per-layer ablation: PPL delta when pruning only layer l
- Heatmap: saliency (4 layers × 8 experts), PPL degradation curve, per-layer damage

# Constraints
- **Confirmed** if: PPL < 50 AND graded monotonic degradation (25% < 50% < 75%) AND Gini > 0.3 in most layers
- **Refuted** if: random pruning ≈ REAP (saliency ineffective at this scale)
- **Inconclusive** if: graded degradation but some layers flat (per-layer rates needed)

# Handoff
New experiment: pure implementation. Train + score + prune + measure. No prior results to build on. Items 129–131 blocked until 128 complete.
