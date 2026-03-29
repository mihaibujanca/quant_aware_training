# Item
94. GELU near-linearity census v2 — GPT-2 FFN (corrected methodology)

# Context
Item 47 (cycle 032) was REFUTED due to flawed GELU thresholding:
- Used post-GELU activation magnitude (max|act| < 1e-5), treating GELU like ReLU
- But GELU never truly saturates: gelu(-3)≈-0.004, gelu(-5)≈-0.00002
- Threshold was above equal-share baseline (1/3072 = 0.033%), flagging 93% of channels
- Conclusion artificially: 0% dead, 0% near-linear channels

Item 94 fixes methodology with pre-GELU detection:
- Check pre-GELU values: max(hidden_j) < -T (T ∈ {2,3,4}) for dead
- Measure derivative: |GELU'(z_j)| < 0.01 for dead, |GELU'(z_j) - 1| < 0.05 for near-linear
- Use 4096 mixed-corpus probe tokens (diverse sources)
- Report per-block: (a) fraction dead on >95% tokens, (b) near-linear on >80%, (c) genuinely nonlinear

# Dependencies
- Existing probe corpus infrastructure (mixed-corpus spec in lm_geometric_compression.py)
- GPT-2 FFN loader (load_nanochat_teacher, model loading)
- Existing PWL/GELU analysis code: experiments/{gelu_pwl_*.py}

# Key Files
- `/aleph/lm_geometric_compression.py` — probe corpus builders
- `/experiments/gelu_pwl_gpt2.py` — reference GELU PWL implementation (cycle 025)
- `/autoresearch/results/cycle_032_*.{json,png}` — baseline artifact (item 47 failure)

# Constraints
- Success threshold: ≥10% dead OR ≥20% near-linear in ≥6/12 blocks
- Failure threshold: <5% dead AND <10% near-linear in all blocks
- Metric soundness: thresholds must be well above equal-share baseline per channel
- Complements item 97 (per-token pattern analysis)
