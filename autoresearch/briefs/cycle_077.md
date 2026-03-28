# Cycle 077: SVD vs topk vs random basis at aggressive compression — GPT-2 FFN

## Item
Item 89: SVD vs topk vs random basis at aggressive compression (k/n <= 0.3). Cycle 071 showed random+lstsq ~ SVD+lstsq at k/n=0.67. Re-run at k/n in {0.3, 0.2, 0.1} on blocks [1, 4, 7, 11] with 3 seeds to test whether the "right" directions matter at low retention.

## Context
- Cycle 071 (item 82): 4 methods at k=512 (k/n=0.67). SVD+lstsq (121.84 mean) ~ random+lstsq (122.31). Both beat topk+lstsq (126.16). Baseline PPL ~121.5.
- At k/n=0.67, high retention may mask basis quality differences. At k/n=0.1 (k=77), only 10% of output directions survive — basis choice should dominate.
- k values: 0.3 -> k=230, 0.2 -> k=154, 0.1 -> k=77 (output space is 768D).
- lstsq overdetermination: at k=230, need N >> 230. Use 1024 probe tokens (4:1 ratio at worst).

## Dependencies
- Item 82 (cycle 071): CONFIRMED. SVD+lstsq ~ topk+lstsq at k/n=0.67. Provides baseline and method implementation.

## Key Files
- `experiments/ffn_jacobian_svd_compression.py` — cycle 071 script (adapt for multiple k values)
- `autoresearch/results/cycle_071_jacobian_svd_basis.json` — prior results at k/n=0.67

## Constraints
- 3 seeds [42, 123, 7]. Blocks [1, 4, 7, 11]. 3 k/n ratios x 3 methods = 9 conditions per block per seed.
- Confirmed if: SVD+lstsq outperforms random+lstsq by >= 5% relative PPL at k/n <= 0.3 in >= 3/4 blocks.
- Refuted if: random+lstsq matches SVD+lstsq within 3% at all tested k/n in all 4 blocks.
