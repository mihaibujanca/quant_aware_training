# Cycle 295 Plan — Item 305

## Dependencies
None. `aleph/datasets.py:make_spirals` exists. `aleph/models.py:MLP` exists. `aleph.make_mlp` does not exist; use `aleph.models.MLP(input_size=2, hidden_size=32, output_size=1, depth=12)` directly. Moons via `sklearn.datasets.make_moons`.

## Item 305 (verbatim Hypothesis)
Post-training median hidden eff_rank ≤ 3 in ≥8 of 12 layers (matching the 2D input dimension); init eff_rank ≥ 20 in ≥8 of 12 layers (random init has nearly full effective rank). The collapse happens during training.

## Approach
New script `experiments/spirals_moons_eff_rank_cycle295.py`. For each dataset ∈ {spirals, moons} × seed ∈ {42, 137, 2024}: build MLP(2,32,1,depth=12,ReLU) on CPU. Capture init post-ReLU activations Y_ℓ ∈ R^(2000,32) for ℓ ∈ {0..11} via `nn.Sequential` index walk. Train Adam lr=1e-3 BCEWithLogitsLoss full-batch on 2000-pt training set until train_acc ≥ 0.95 or max 5000 epochs. Re-capture post-ReLU Y_ℓ. Per layer compute C = Y^T Y / N (32×32), torch.linalg.eigvalsh, p_i = clip(eig,0)/sum, eff_rank = exp(-Σ p_i log p_i) (nan-safe). Aggregate: per-layer mean across 3 seeds; count layers meeting thresholds.

## Files
- `experiments/spirals_moons_eff_rank_cycle295.py` (new)
- `autoresearch/results/cycle_295_spirals_moons_eff_rank.json`
- `autoresearch/results/cycle_295_summary.md`
- `autoresearch/results/cycle_295_manifest.txt`
- `autoresearch/results/cycle_295_exec_result.txt`
- `autoresearch/logs/cycle_295_run.log`
- `autoresearch/audits/cycle_295.md`
- `autoresearch/results/cycle_295_audit.txt`

## Config
Spirals: `make_spirals(n_samples=2000, noise=0.5, n_turns=3, random_state=seed)`. Moons: `sklearn.datasets.make_moons(n_samples=2000, noise=0.1, random_state=seed)`. MLP(2,32,1,depth=12,ReLU). Adam lr=1e-3 BCEWithLogitsLoss full-batch. Stop when train_acc ≥ 0.95 (or 5000 epochs cap). 3 seeds {42, 137, 2024}. CPU (~13K params trivial). Trainable per net = 13,505. Wall ≤5 min.

## Verification
- Trainable param count asserted == 13,505 per net (input 2*32+32=96; 11 hidden 32*32+32=1056×11=11,616; output 32+1=33 → 96+11616+33=11745... rechecked: 96 + 11*1056 + 33 = 11,745). Will assert measured value at runtime.
- Train acc ≥ 0.95 confirmed before recording final eff_rank (else log "did not converge", treat seed as failure case in JSON but still recorded).
- Per-layer Y has shape (2000, 32); covariance = Y^T Y / 2000 has shape (32, 32).
- eff_rank ∈ [1, 32] sanity: assert 1.0 ≤ er ≤ 32.0 ± 1e-6 per layer.
- Init eff_rank ≈ near 32 (random Gaussian post-ReLU spread); fail loud if init mean < 10 (would mean a bug).

## Verdict logic (item 305 verbatim)
- CONFIRMED iff post-training eff_rank ≤ 3 in ≥8 of 12 layers AND init eff_rank ≥ 20 in ≥8 of 12 layers.
- REFUTED iff post-training eff_rank ≥ 10 in ≥6 layers.
- INCONCLUSIVE otherwise.
Apply per-dataset; report both. Overall verdict: CONFIRMED iff both datasets CONFIRMED; REFUTED iff either REFUTED; INCONCLUSIVE otherwise.

## Baselines
Cycle 278 GPT-2 c_proj output eff_rank profile {b0: 71.93, b1: 1.94, b2: 1.05, b3: 24.29, b4: 49.04, b5: 179.19, b6-8: 290+, b9: 195.5, b11: 22.49}. Random-init expected ≈ 32 for a 32-wide layer.
