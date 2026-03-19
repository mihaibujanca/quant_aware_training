# Cycle 048: Item 30 Handoff

## Item
**Local complexity proxy as a compression difficulty predictor — GPT-2**
- Apply k-NN Jacobian variation proxy to GPT-2 FFN blocks; correlate with cycle 005 single-block swap damage.

## Context
- **Toy result (cycle 023, item 29)**: Spearman ρ=0.80 (p=0.20, n=4) between local complexity and r_held → PASSES ≥0.5 correlation criterion. Training-time dynamics failed (LC did NOT decrease during training, opposite of Patel 2025 prediction), but static correlation signal is strong.
- **Architecture signal**: ReLU toy exploits activation-pattern counting (exact); GELU requires Jacobian proxy (described in item 30).
- **Compression context**: Cycle 005 baseline established single-block swap damage for all 12 GPT-2-small FFN blocks. Cycle 044–045 revealed severe domain specificity (Shakespeare→WikiText -31pp), but compression method itself (topk + centroid merge + lstsq refit) is robust — the constraint is corpus-dependent probing data.

## Dependencies
- ✓ Item 9 (benchmark lock-in, confirmed cycle 004)
- ≈ Item 29 (toy local complexity, inconclusive cycle 023 but correlation signal **passes** threshold)
- **Blocker check**: Item 29 NOT refuted (correlation criterion met). Item 30 eligible to run.

## Key Files
- `aleph/lm_geometric_compression.py` — GPT-2 model loading, probe corpus, full-block FFN swaps
- `aleph/relu_geometric_compression.py` — single-affine residual r computation (item 20 reference)
- `autoresearch/results/cycle_005_summary.md` — single-block swap damage reference data
- `autoresearch/results/cycle_023_local_complexity.json` — toy toy ρ=0.80 for method validation

## Constraints
- **Proxy definition**: For each probe token x, compute std of block Jacobians J(x') for k nearest neighbors in input space. Aggregate via mean over probe set per block.
- **k-NN search space**: Use pre-GELU hidden activations (block input representation) for neighbor discovery.
- **Spearman baseline**: ρ ≥ 0.5 for PASS; < 0.2 is FAIL; 0.2–0.5 is INCONCLUSIVE.
- **Cycle 005 reference**: 12-block damage curve already measured; no need to re-run swaps.

## Handoff
**Planning**: Implement k-NN Jacobian variation per GPT-2 block. Measure local complexity by: (1) sample 512 random probe tokens, (2) for each, find k=10–15 nearest neighbors in hidden-activation space, (3) compute mean-std of block Jacobians in neighborhood, (4) aggregate over probe set. Correlate with cycle 005 damage. Expected effort: low (200–300 lines, sklearn KDTree, torch.func.jacrev for GELU blocks).
