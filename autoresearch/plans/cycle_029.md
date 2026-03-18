# Plan — Cycle 029: Local-Chart Analytic FFN Compression

## Pre-flight
- P1: Not a duplicate (item 17, depends on items 10, 13, 15 — all done).
- P2: Global geometric baseline available in `aleph/lm_geometric_compression.py`.
      `probe_ffn_geometry`, `derive_ffn_compression`, `swap_compressed_ffns`,
      `_evaluate_model`, `_fit_output_projection` all callable.
      GPT-2 + Shakespeare corpus loading uses existing TeacherLoadSpec / ProbeCorpusSpec.
- P3: Confirmed criterion: ≥20% ΔPPL improvement over geometric on ≥3 of 3 blocks.
- P4: No modification to agenda.md until after results. No program.md changes.

## Blocks and budget
- Blocks: 2 (medium, damage=0.011), 5 (easy, damage=0.001), 11 (hard, damage=0.065)
- Compression: 8× → k_global=384 hidden channels (from 3072)
- Local K=4: 96 channels per chart (same total parameter budget via matched k_per_chart)
- Local K=8: 48 channels per chart (same total budget)
- Note: local-chart also incurs routing overhead — K centroids × 768 floats, negligible

## Algorithm: global geometric (baseline)
1. Probe → `channel_scores` per neuron
2. Top-384 by score, lstsq refit W_down on (hidden_subset, FFN_output)
3. Standard CompressedMLP replacement (no routing)

## Algorithm: local-chart (new)
1. Probe → collect input_sample x [N, 768] per block
2. k-means on x → K cluster centroids C [K, 768], assign each token → chart
3. For chart k:
   a. Subset tokens: x_k, h_k [N_k, 3072], y_k [N_k, 768]
   b. channel_scores_k = std(h_k) * output_norms * (|mean(h_k)| + |mean(σ'(z_k))| + 1e-6)
   c. Top-(k_global // K) channels by chart-local score → selected_k
   d. lstsq: W_down_k from (h_k[:, selected_k], y_k)
   e. W_up_k = W_up_original[selected_k, :]  (keep upstream weights)
4. Inference: MoE-style — route x to nearest centroid, apply chart's W_up_k, act, W_down_k

## Script
File: `experiments/local_chart_ffn_compression.py`

Steps:
1. Load GPT-2 teacher + Shakespeare corpus (1024 seq_len, 2048 probe tokens)
2. probe_ffn_geometry for blocks [2, 5, 11]
3. For each block:
   a. Build global geometric baseline (384 channels), measure block MSE + PPL
   b. For K in [4, 8]:
      - k-means on input_sample (K clusters, 20 inits, seed=42)
      - Per-chart: score channels within chart, top-(384//K), lstsq W_down
      - Build LocalChartMLP module (stores K mini-MLPs + centroids, routes at forward)
      - Swap into teacher copy, eval full-model PPL
      - Compute per-chart block MSE (average over all charts, weighted by chart size)
4. Collect results dict, save to `autoresearch/results/cycle_029_local_chart.json`
5. Print summary: per-block ΔPPL% (global vs local K=4 vs K=8)
6. Evaluate criterion; update agenda.md; commit

## Implementation note: LocalChartMLP
```python
class LocalChartMLP(nn.Module):
    # stores K × (W_up_k [k//K, 768], W_down_k [768, k//K])
    # forward: nearest centroid routing, then apply chart
    # output_bias: original c_proj bias always added (K shared)
```

## Evaluation
- block_mse: F.mse_loss(LocalChartMLP(x_probe), y_probe) vs global compressed
- ppl: single-block swap + _evaluate_model on eval_val
- improvement: (ppl_geo - ppl_local) / ppl_geo * 100 must be ≥20% on ≥3 blocks

## Output artifacts
- `experiments/local_chart_ffn_compression.py`
- `autoresearch/results/cycle_029_local_chart.json`
- `autoresearch/results/cycle_029_manifest.txt`
