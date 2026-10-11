# Item
230 (P0): W_down column geometry census — anti-podal pairs, clusters, cross-block alignment.
Operate on GPT-2 small W_down ∈ R^(768×3072) for all 12 blocks. NO activations needed.

# Context
- Previous 190+ cycles of GELU bump investigation analyzed activations only. Rescue note (handoff/quant_aware_training.md) and `docs/gelu_bump_audit.md` §11 establish that direction is determined by W_down COLUMN GEOMETRY, not activation magnitudes.
- Decomposition: pos_contribution = Σ α_c · W_down[:,c], neg_contribution = Σ (-β_c) · W_down[:,c]. Activation values control magnitudes; column directions in R^768 control where the FFN can write.
- This item is the first weight-geometry investigation in this lineage.
- Output is a structural report, NOT a PPL number — interpretable claim, plots, JSON.

# Dependencies
None — operates directly on W_down weights of stock GPT-2.

# Key Files
- HuggingFace `gpt2` model: `transformer.h[i].mlp.c_proj.weight` is W_down (in PyTorch nn.Linear convention, the `weight` shape is (out_features=768, in_features=3072) — verify by inspection).
- For GPT-2 in HF transformers, MLP uses `Conv1D` not `Linear`: `c_fc` weight shape (3072, 768) and `c_proj` weight shape (768, 3072). Need to handle Conv1D layout: `c_proj.weight` ∈ R^(in=3072, out=768). The "column" of W_down (treating each input channel c as a vector in R^768) is `c_proj.weight[c, :]` — a row when stored as (3072, 768).
- `aleph/lm_geometric_compression.py` for any helpers.
- `autoresearch/results/cycle_215_*` for the most recent JSON output style template.

# Constraints
- All measurements per-block (12 blocks).
- Per-block computations:
  (a) pairwise cosine similarity matrix of 3072 columns (3072×3072 = ~9.4M floats per block, fine).
  (b) histogram of cosines AND count anti-podal pairs at threshold cos < −0.9 (and optionally −0.8, −0.95 for sensitivity).
  (c) top-20 singular values of W_down (already 768×3072).
  (d) k-means clustering (k=32) of columns (treat each column as point in R^768; sklearn or torch impl).
  (e) cross-block alignment: principal angles between top right singular vectors of W_down_l and top right singular vectors of W_up_{l+1} = c_fc.weight in block l+1.
- Output: per-block JSON + aggregated summary report.
- Confirmed: ≥20% of columns in blocks 1-5 with anti-podal twin (cos<−0.9), AND clusters geometrically distinct (mean within-cluster cos > 0.5, between-cluster cos < 0.2 at k=32).
- Refuted: <5% anti-podal pairs in any block AND columns near-uniform on sphere.
- Inconclusive: some structure but no clean story.
- Random/null baseline for anti-podal pair count: random unit vectors in R^768 — probability of cos<−0.9 is ~0 (volume of antipodal cap). The plan should explicitly compute random-baseline for sanity.
- Use ≥3 seeds is not applicable here — weights are deterministic; run once on the model, but the random baseline can use 3 random seeds.

# Handoff
Plan agent: design exact computations and produce a script `experiments/wdown_column_census_cycle216.py`. Verify Conv1D layout convention before counting anti-podal pairs. Compute null baseline numerically: sample 3072 random unit vectors in R^768 (3 seeds), measure max negative cosine. Report whether anti-podal pair fraction exceeds null baseline by ≥10× as the meaningful test. Cluster quality: use sklearn KMeans (k=32, n_init=5) and compute mean within/between cosines. Cross-block alignment: use principal angles via SVD of (U_l^T V_{l+1}) where U is top-r right singular vectors of W_down_l (r=20) and V is top-r right singular vectors of W_up_{l+1} (also r=20). Produce a 12-block report `autoresearch/results/cycle_216_wdown_geometry.json` and a 1-2 paragraph markdown summary `autoresearch/results/cycle_216_summary.md` with the per-block one-sentence claims.
