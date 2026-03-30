# Item
140 — QK vs V error decomposition under FFN compression (interpretability)

# Context
- k=4 compressed model (blocks [1,4,7,11] at 6× topk+lstsq). Baseline needed.
- Recent cascade correction cycles (119-121): k=5@+73.8%, k=8@+91.4% recovery with 49-98K params.
- Open question (no prior work): does attention error come from pattern change (QK-space) or value change (V-space)?
- Literature flags QK matrices are rank-critical, V has compression headroom. Item will quantify.

# Dependencies
- None explicit. k=4 baseline must be established/recovered.
- Feeds item 142 (correction placement) and 144 (alignment analysis).

# Key Files
- Survey: `autoresearch/surveys/attention_cascade_correction.md` (refs Basu 2603.20991, EdgeQAT 2402.10787, BinaryAttention 2603.09582)
- Most recent k=4 data: cycle 033-042 FFN baselines (topk+lstsq, no corrections). No explicit k=4 @ [1,4,7,11] result in results/ yet.
- Attention analysis code exists: `experiments/joint_ffn_attn_compression_gpt2.py`, `experiments/transformer_correction_analysis.py`

# Constraints
- 2048 OWT tokens for measurement.
- Per-layer per-head heatmap: {QK_frac, V_frac, interaction_frac}.
- Confirmed if one component >70% in ≥8/12 layers (clear target).
- Refuted if equal (40-60%) everywhere.

# Handoff
Establish k=4 [1,4,7,11]@6× baseline (PPL ~+9-10%). Decompose 12 attention layers into QK-space error (pattern KL divergence) vs V-space error (value swap). Measure interaction term. Heatmap by layer/head. Target: identify dominant error axis (attention or value) for items 142-144.
