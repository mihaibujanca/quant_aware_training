# Task
Cycle 144, item 149: Measure layer-local pullback norm ||J_l e_l|| for each GPT-2 FFN block's compression error, where J_l is the Jacobian of the rest-of-network map (layers l+1 through ln_f + lm_head). Compare correlation with PPL damage against Frobenius and global pullback baselines from cycle 136.

# Hypothesis
Layer-local pullback captures non-uniform downstream amplification that the constant-scaling global pullback misses.

Criteria (verbatim from agenda):
- **Confirmed if**: layer-local metric Spearman |rho| with PPL exceeds global pullback |rho| by >= 0.15.
- **Refuted if**: layer-local and global pullback correlations are within 0.05 (global G is sufficient).
- **Inconclusive if**: layer-local is marginally better (0.05-0.15) or computation is too noisy (cross-seed std > 0.1).

# Pre-flight Results
- P1 (duplicate): No. Cycle 136 tested global pullback, not layer-local JVP. Distinct.
- P2 (deps): 146 completed (cycle 136, REFUTED). Baseline data available. No "skip if refuted" clause — the dep is for baseline data, not for 146 to be confirmed.
- P3 (concrete): Yes. Single analysis script, clear metrics, clear thresholds.
- P4 (program changes): No.
- P5 (null baseline): Random error vectors would show rho near 0 for all three metrics. Frobenius baseline rho=0.797 is well above chance.
- P6 (method-architecture): JVP through transformer blocks with GELU — `torch.func.jvp` handles this correctly. No activation-mismatch issue.
- P7 (statistical power): 12 data points (blocks) for Spearman is low but standard for this program. 3 seeds provide stability estimates.

# Steps
1. Load cycle 136 results (`cycle_136_pullback_error.json`) for per-block Frobenius norms, G-weighted norms, and PPL deltas. Reuse PPL deltas directly — no need to recompute single-block swaps.
2. For each seed and each block l (0-11):
   a. Load GPT-2, build probe corpus (200 seqs, 128 tokens from OWT).
   b. Run forward pass through layers 0..l with topk compression at layer l (k = hidden/6 = 512). Extract compression error e_l = z_compressed - z_float at FFN output.
   c. Average e_l across tokens/sequences to get mean error vector (768-dim).
   d. Define rest-of-network: layers l+1..11 + ln_f + lm_head. Use `torch.func.jvp(rest_of_net, (input,), (e_l,))` to get J_l @ e_l. Take L2 norm.
3. Compute Spearman rho for each of: Frobenius ||e_l||, global ||U e_l||, layer-local ||J_l e_l|| vs PPL delta. Average across seeds, report std.
4. Save results to `autoresearch/results/cycle_144_layer_local_pullback.json` and scatter plot to `autoresearch/results/cycle_144_scatter.png`.
5. Apply verdict criteria. Move item 149 from NEXT to PAST in agenda.md.
6. Commit: `[autoresearch] cycle 144: layer-local pullback — {VERDICT}`

# Validation
- Frobenius rho from this run should match cycle 136 (rho ~ 0.797). If it diverges by >0.05, something is wrong.
- Layer-local norms should be finite and positive for all blocks. If any are NaN/Inf, JVP failed.
- Cross-seed std for all rho values should be < 0.1.

# Artifacts
- `experiments/layer_local_pullback_cycle144.py`
- `autoresearch/results/cycle_144_layer_local_pullback.json`
- `autoresearch/results/cycle_144_scatter.png`

# Fallbacks
- If JVP through full rest-of-network OOMs or is too slow: truncate to just 3 downstream layers (l+1, l+2, l+3). Document the truncation.
- If averaging e_l across tokens collapses information: use per-token JVP norms, then average the output norms. More expensive but more accurate.
- If Frobenius rho doesn't reproduce cycle 136: debug by loading cycle 136 script directly and comparing intermediate values.
