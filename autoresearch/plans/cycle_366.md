# Task
Trace per-token residual stream error through GPT-2's full 12-block forward pass. Compare float model vs 6-block compressed model (blocks [2,1,10,4,3,11], uniform topk+lstsq+adapter recipe, k=512, adapter h=64). Measure ||h_compressed - h_float|| (absolute norm) at each block output. Compare against an additive baseline (cumsum of local FFN errors) to distinguish simple accumulation from attention amplification. Identify where error growth transitions from linear to superlinear.

# Hypothesis
Error growth has a clear inflection point in a specific block range.
- **CONFIRMED**: A block i exists where g_i >= 2 * median(g_{0..11}), AND this inflection block is the same index in >= 2/3 seeds.
- **REFUTED**: No block has growth rate >= 2x median, OR the inflection block differs across seeds (appears in < 2/3 seeds at the same index).
- **INCONCLUSIVE**: Error is non-monotonic (decreases at 2+ blocks), making growth-rate analysis ill-defined.

# Pre-flight Results
- P1 PASS: No past experiment measures cumulative residual stream norm divergence at all 12 block outputs.
- P2 PASS: No dependencies.
- P3 PASS: Single forward-pass instrumentation, no training needed (adapters pre-trained).
- P5: Null baseline = additive cumsum of local FFN errors. Verdict uses absolute error norms only.

# Steps
1. Create `experiments/gpt2_residual_error_monitor_cycle375.py`. Load GPT-2, wikitext-2 (200 eval sequences, SEQ_LEN=128). Compress 6 blocks with uniform topk+lstsq+adapter (k=512, h=64), train adapters 1500 steps.
2. Hook each `model.transformer.h[i]` to capture block output (residual stream). Also hook pre/post-FFN within each compressed block to capture local FFN error: ||FFN_compressed(x) - FFN_float(x)||.
3. Run two forward passes per batch: (a) float model, (b) compressed model. Compute per-block absolute error norm ||h_compressed - h_float|| averaged over all tokens across 200 sequences.
4. Compute additive baseline: cumsum of local FFN errors at compressed blocks (non-compressed blocks contribute 0). Plot both actual cumulative error and additive baseline vs block index. The gap = attention amplification signal.
5. Verdict uses absolute error norms only (relative error stored as supplementary). Growth rate g_i = err_{i+1} - err_i. CONFIRMED requires inflection block (g_i >= 2x median) at the same index in >= 2/3 seeds.
6. Run 3 seeds [42, 137, 2024]. Report mean±std of inflection block across seeds.
7. Save JSON to `autoresearch/results/cycle_366_residual_error_monitor.json`. Save plot (actual error + additive baseline vs block, growth rate overlay) to `autoresearch/results/cycle_366_error_profile.png`.
8. Update `autoresearch/agenda.md`: move item 375 to closed. Write manifest. Git commit.

# Validation
- Float-vs-float control: error at all blocks < 1e-6.
- Block 0 (non-compressed, no upstream): error must be 0.
- Additive baseline must be <= actual cumulative error at every block (errors can only be amplified, not cancelled, in expectation).

# Artifacts
- `experiments/gpt2_residual_error_monitor_cycle375.py`
- `autoresearch/results/cycle_366_residual_error_monitor.json`
- `autoresearch/results/cycle_366_error_profile.png`
- `autoresearch/results/cycle_366_manifest.txt`

# Fallbacks
- If adapter training diverges on a seed, skip that seed. 2/3 seeds sufficient.
- If error non-monotonic, report raw profile and mark INCONCLUSIVE.
- If MPS OOM on dual-model pass, cache float activations first, then run compressed.
