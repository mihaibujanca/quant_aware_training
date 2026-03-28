# Task
Item 28: Persistent homology of GPT-2 hidden states (pre-LayerNorm residual stream). Final verdict on topology-as-compression-predictor direction.

# Premise
Four prior PH experiments (cycles 047, 064, 065, 066) produced 0 useful results. Cycle 065 found TP grows monotonically in toy ReLU — opposite of hypothesis — but that was a scale artifact (unnormalized TP, confirmed by audit). GPT-2 differs structurally: residual connections preserve input geometry across blocks, and attention mixing redistributes information across token positions — neither occurs in feedforward ReLU nets. This gives a structural reason topology evolution may differ. If GPT-2 also shows no signal, the topology direction is exhausted and should be abandoned.

# Hypothesis
- **Confirmed if**: Spearman(total_persistence, damage_8x) >= 0.35 AND TP decreases through >= 60% of adjacent block pairs, AND PCA variance >= 80% at 30D, AND hook confirmed pre-LN.
- **Refuted if**: Betti numbers approximately constant (std/mean < 0.15), provided PCA variance >= 80% and hook is pre-LN.
- **Inconclusive if**: PCA variance < 80% at 30D, or neither confirm/refute met.

# Ground truth
Damage from cycle 005 (per-block swap-damage at 8x and 16x), stored in `autoresearch/results/cycle_048_jacobian_proxy_gpt2.json`. Use both compression levels.

# Steps
Script: `experiments/phom_gpt2.py`

1. **Setup**: Load GPT-2. Register forward hooks on `model.transformer.h[i]` output (tuple index 0) — this is the residual stream BEFORE the next block's `ln_1`, i.e., pre-LayerNorm. Verify by asserting hooked output norm std > 0.5 (post-LN would be ~1.0 with near-zero std). Tokenize `data/shakespeare.txt`, 3 non-overlapping 512-token windows per seed (seeds [42, 123, 7]) at offsets len/4, len/2, 3*len/4.

2. **PCA + variance check**: For each seed, block: PCA 512x768 to 30D. Log cumulative variance explained at 30D. If any block has variance < 80%, flag. Additionally, for seed=42 only: run ripser at 20D and 40D to test sensitivity. If rank-order of TP across blocks changes qualitatively (Spearman between 20D and 40D TP profiles < 0.7), report projection-dependent and mark INCONCLUSIVE.

3. **Ripser**: For each seed, block, at 30D: pairwise Euclidean distance (512x512). Max filtration = 95th percentile. `ripser(D, maxdim=1, thresh=max_r, distance_matrix=True)`. Extract TP (sum death-birth for features with lifetime > 1% of max_r) for H0 and H1.

4. **Null baseline**: Per seed, 512 isotropic Gaussian points in 30D matched to block-6 mean norm. Same ripser pipeline. Report TP.

5. **Correlation**: Spearman(mean_TP_combined, damage_8x) and (damage_16x), one-tailed. Also H0/H1 separately. Leverage guard: rho_without_L0. Monotonicity: count adjacent pairs where TP[i] > TP[i+1]; need >= 7/11.

6. **Save**: JSON to `autoresearch/results/cycle_074_phom_gpt2.json`. Plot (2x1): TP vs block with seeds + null; scatter TP vs damage. Save `.png`.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/phom_gpt2.py`

# Validation
- All 36 ripser calls return finite diagrams.
- Null baseline TP < most block TPs.
- Cross-seed std < 50% of mean TP.
- PCA variance >= 80% at 30D for all blocks (else INCONCLUSIVE).
- PCA sensitivity: Spearman(TP_20D, TP_40D) >= 0.7 for seed=42 (else INCONCLUSIVE).
- REFUTED only valid if PCA and pre-LN guards pass.

# Artifacts
- `experiments/phom_gpt2.py`
- `autoresearch/results/cycle_074_phom_gpt2.json`
- `autoresearch/results/cycle_074_phom_gpt2.png`

# Fallbacks
- If ripser hangs on 512x512: subsample to 256 tokens.
- If PCA 30D variance < 80%: try 50D (smaller distance matrix via landmark subset of 256 points).
