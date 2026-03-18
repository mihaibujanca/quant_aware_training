# Cycle 020 Plan — Manifold Curvature Trajectory: GPT-2

## Pre-flight
- P1: `which python` → use `.venv/bin/python`
- P2: GPT-2 loads via `load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device="cpu"))`
- P3: Shakespeare corpus via `build_probe_corpus(ProbeCorpusSpec(text_path="data/shakespeare.txt", sequence_length=512, probe_train_sequences=1, teacher=teacher_bundle))`
- P4: Cycle 005 damage embedded in script as hardcoded dict (from results.jsonl, hf=0.125, geometric, mean over seeds)

## File
`experiments/manifold_curvature_gpt2.py`

## Algorithm

### Hidden state extraction
- Register forward hooks on GPT-2: embedding output + each block output (13 points)
- Run one forward pass on one batch of 512-token probe sequence
- Collect tensors: shape (N_tokens, 768) at each of 13 layers (N_tokens = seq_len - 1 via label shifting, use 512)

### Local PCA curvature (per layer)
- Input: H of shape (N, 768), k=50
- Compute pairwise distances: cdist(H, H) → (N, N)
- For each token i: get k nearest neighbor indices, extract H[neighbors] shape (k, 768)
- Center: subtract mean → X_centered (k, 768)
- Covariance: C = X_centered.T @ X_centered / (k-1)  shape (768, 768)
- Eigenvalues via torch.linalg.eigvalsh → sort descending
- Total variance = sum of all eigenvalues
- Intrinsic dim d = smallest d such that sum(top_d eigenvalues) / total_variance ≥ 0.95
- Curvature = 1 - sum(top_d) / total_variance  (= unexplained fraction)
- Average curvature over all N tokens → scalar per layer

### Compression damage lookup
Hardcode dict from cycle 005 (hf=0.125, geometric, mean rel_ppl_increase per layer_id 0..11).
Layer 12 (post-block-11) has no block damage → exclude from correlation.
Use layers 0..11 (block output after each block = layer index 1..12 in trajectory,
but damage is indexed by block 0..11).

### Correlation
- Spearman(curvature_at_block_output[i], damage[i]) for i in 0..11
- Print trajectory plot + scatter

### Plots
- `autoresearch/results/cycle_020_curvature_trajectory.png`: line plot over 13 layers
- `autoresearch/results/cycle_020_curvature_vs_damage.png`: scatter of curvature vs damage (layers 0-11)

### Output JSON
`autoresearch/results/cycle_020_curvature.json`:
```json
{
  "curvature_per_layer": [...],   // 13 values (layer 0=embedding, 1-12=blocks 0-11)
  "damage_per_block": {...},      // keys "0".."11"
  "spearman_rho": float,
  "spearman_p": float,
  "monotone_final_third": bool,
  "verdict": "confirmed|refuted|inconclusive"
}
```

## Verdict Logic
```python
final_third = curvature[9:]  # layers 9,10,11,12
monotone = all(final_third[i] >= final_third[i+1] for i in range(len(final_third)-1))
if monotone and rho >= 0.4:
    verdict = "confirmed"
elif rho < 0.2 or anti_correlates:
    verdict = "refuted"
else:
    verdict = "inconclusive"
```

## Efficiency Notes
- N=512, k=50: 512 local PCAs each on (50×768) matrix → feasible on CPU (~1-2 min)
- Use float32 throughout
- Batch eigendecomposition not needed — sequential loop is fine
