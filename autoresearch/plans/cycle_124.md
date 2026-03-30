# Task
Item 141: Attention entropy shift under FFN compression — GPT-2.
Measure per-head attention entropy for float vs k=4 compressed model, correlate with cycle 104 cascade amplification.

# Hypothesis
FFN compression changes attention distributions measurably (|ΔH| > 0.1 nats in most layers), and the magnitude of entropy shift correlates with cycle 104's per-transition attn_frac (|Spearman| > 0.3).

# Pre-flight Results
- P1 PASS: No duplicate. Item 37 (cycle 019) measured activation pattern entropy on toy ReLU — different model, different metric, different question.
- P2 PASS: No dependencies. Cycle 104 results at `autoresearch/results/cycle_104_cascade_source.json`, cycle 123 code pattern available.
- P3 PASS: Single script, clear metrics, one cycle.
- P4 PASS: No program.md changes needed.
- P5 (null baseline): With no compression, entropy shift = 0 by construction (same model). The question is whether compression-induced FFN error propagates enough to shift attention softmax distributions.
- P6 PASS: GPT-2 has standard MHA with 12 heads × 12 layers; attention weights accessible via hooks.
- P7: 2048 tokens × 128 seq_len = ~16 query positions per sequence. Per head per layer: ~2048 entropy values per seed, 3 seeds. Adequate for mean shift detection at 0.02 nat resolution.

# Steps
1. Create `experiments/attn_entropy_shift_gpt2.py`. Reuse cycle 123 model setup (imports from `aleph.lm_geometric_compression`, same COMPRESS_BLOCKS=[1,4,7,11], HIDDEN_FRACTION=1/6, POLISH_STEPS=500).
2. For each seed in [42, 123, 7]:
   a. Load GPT-2 float model, build compressed student (same as cycle 123).
   b. PPL sanity check: abort if float PPL outside [20, 80] or compressed PPL > 3× float.
   c. Register forward hooks on all 12 `attn.c_attn` → capture attention weights post-softmax (shape [batch, n_heads, seq, seq]). Hook on `GPT2Attention` — capture `attn_weights` before dropout.
   d. Run 2048 tokens (16 batches × 4 × 128) through both float and compressed models with hooks.
   e. For each layer L, head H, compute mean entropy H_float and H_compressed over all query positions: `H = -(a * log(a + 1e-10)).sum(dim=-1).mean()`.
   f. Store per-layer per-head: mean_H_float, mean_H_compressed, delta_H = compressed - float.
3. Aggregate across seeds: mean and std of delta_H per layer per head.
4. Compute per-layer mean |delta_H| (averaged over heads).
5. Load cycle 104 `avg_by_transition` → extract `attn_frac` for transitions 0–10. Map transition i to layer i+1 (transition i is between layer i and i+1; entropy shift at layer i+1 reflects post-attention change). Compute Spearman correlation between per-layer mean |delta_H| (layers 1–11) and attn_frac (transitions 0–10).
6. Verdict logic:
   - Confirmed: mean |delta_H| > 0.1 nats in ≥8/12 layers AND |Spearman| > 0.3.
   - Refuted: |delta_H| < 0.02 nats everywhere.
   - Inconclusive: shift exists but no correlation with amplification.
7. Save JSON results, generate 2-panel plot (bar chart of per-layer mean |delta_H|; scatter of |delta_H| vs attn_frac with Spearman annotation).
8. Update `autoresearch/agenda.md` item 141 with outcome and artifact paths. Commit.

# Validation
- PPL sanity: float ∈ [20, 80], compressed < 3× float (same as cycle 123).
- Entropy range check: float entropy should be in [0, ln(128)] ≈ [0, 4.85] nats. Flag if any head outside this.
- Cross-seed std of delta_H should be < 0.05 nats (cycle 123 found std < 0.02 for QK/V fracs).
- Spearman p-value reported alongside rho.

# Artifacts
- `experiments/attn_entropy_shift_gpt2.py`
- `autoresearch/results/cycle_124_attn_entropy_shift.json` — per-seed per-layer per-head entropy values, delta_H, Spearman results, verdict
- `autoresearch/results/cycle_124_attn_entropy_shift.png` — 2-panel figure

# Fallbacks
- If GPT-2 attention weights not directly accessible via hooks: use manual QKV split + softmax computation from hidden states (cycle 123 already does QK decomposition).
- If Spearman is insignificant but |delta_H| > 0.1: report INCONCLUSIVE, note which layers shift most and whether they align with compressed blocks [1,4,7,11].
- If entropy shift is tiny (<0.02): REFUTED early, skip correlation. Report that cascade is V-space only.
