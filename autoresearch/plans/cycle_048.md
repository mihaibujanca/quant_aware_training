# Task
Item 30: Local complexity proxy as a compression difficulty predictor — GPT-2. Revision: address leverage-point risk (layer 0), run all 3 Jacobian proxies unconditionally, use one-tailed Spearman.

# Hypothesis
Higher per-block Jacobian variation predicts higher single-block swap damage.

- **Confirmed if:** Spearman correlation between local complexity proxy and compression damage >= 0.5, making this a cheap compression-difficulty oracle.
- **Refuted if:** correlation < 0.2, indicating that local nonlinearity does not predict block-level compression difficulty.
- **Inconclusive if:** the toy result from experiment 29 is strong but the GPT-2 proxy gives weak correlation, suggesting the Jacobian-variation proxy is too indirect.

Additional guard (revision): one-tailed Spearman throughout. If rho_full >= 0.5 but rho_without_L0 < 0.2, classify INCONCLUSIVE (leverage-point artifact).

# Pre-flight Results
- Cycle 005: 12-block damage at 16x: layer 0 = +33.12% (0.3312), layers 1-11 range -0.63% to +6.73%.
- Layer 0 dominates damage vector (0.1768 at 8x, 2.6x next-highest). Known leverage point.
- Cycle 023: toy rho=0.80 (n=4) via activation-pattern counting. Different proxy than Jacobian variation.

# Steps
1. **Create** `experiments/gpt2_local_complexity.py`:
   - Load `gpt2` via HF, 512 probe tokens from `data/shakespeare.txt`.
   - For each FFN block (0-11):
     a. Extract pre-GELU hidden activations for all probes. Build KDTree (sklearn), k=10 neighbors.
     b. Compute block Jacobian at each neighbor via `torch.func.jacrev`.
     c. For each probe's neighborhood, compute ALL THREE proxies:
        - (P1) Std of Jacobian Frobenius norms (primary, matches brief)
        - (P2) Mean spectral norm of Jacobians
        - (P3) Mean effective rank (nuclear norm / spectral norm)
     d. Per-block aggregate: mean over 512 probes for each proxy.
   - Hardcode cycle 005 damage vectors (8x and 16x).
   - Compute one-tailed Spearman (alternative='greater') for each proxy vs each damage vector.
   - Leave-one-out: remove layer 0, recompute Spearman on n=11 for all proxy-damage pairs.
   - Classification:
     - CONFIRMED: any proxy rho_full >= 0.5 AND rho_without_L0 >= 0.2
     - REFUTED: all proxies rho_full < 0.2
     - INCONCLUSIVE: rho_full >= 0.5 but rho_without_L0 < 0.2, or all in [0.2, 0.5)
   - Save JSON + scatter plot (3 proxies x 2 damage rates, layer 0 marked).
2. **Run**: `python experiments/gpt2_local_complexity.py`
3. **Evaluate**: report all 6 rho values (3 proxies x 2 damage rates) + leave-one-out counterparts.
4. **Write** `autoresearch/results/cycle_048_summary.md`.
5. **Update** `autoresearch/agenda.md` item 30 with outcome.
6. **Commit**: `[autoresearch] cycle 048: local complexity proxy — GPT-2 Jacobian variation`

# Validation
- All 12 blocks produce finite, non-zero values for all 3 proxies (no NaN/inf).
- Scatter plot has 12 points per panel with layer 0 visually marked.
- Both rho_full and rho_without_L0 reported for every proxy-damage pair.
- One-tailed p-values reported (not two-tailed).

# Artifacts
- `experiments/gpt2_local_complexity.py`
- `autoresearch/results/cycle_048_local_complexity_gpt2.json`
- `autoresearch/results/cycle_048_local_complexity_gpt2.png`
- `autoresearch/results/cycle_048_summary.md`

# Fallbacks
- If `jacrev` OOMs: subsample to 128 probes, k=5 neighbors.
- If KDTree slow on 768D: PCA to 64D for neighbor search only (Jacobians in full space).
- If all 3 proxies rho < 0.2: REFUTED — Jacobian variation does not transfer from ReLU toy to GELU GPT-2.
