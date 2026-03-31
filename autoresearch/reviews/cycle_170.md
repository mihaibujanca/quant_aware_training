# Cycle 170 Audit — Literature Scan: FFN Compression 2025-2026

## G1: Did the cycle answer its question?
YES. Found 11 methods, identified >=2 untried approaches (SoLA split, PruneNet spectral selection) that are comparable to our recipe.

## G2: Is the verdict correct?
CONFIRMED. Both SoLA and PruneNet are calibration-only, FFN-focused, structured approaches not tried in this repo. The SoLA split (preserve top + SVD rest) is a genuinely different reconstruction strategy from our topk+lstsq. PruneNet's spectral criterion (KS-distance on singular values) is a different selection criterion from our activation-magnitude topk.

## G3: Were there methodology issues?
MINOR. Most papers report on LLaMA-7B+ scale, not GPT-2. Direct PPL comparison is approximate. Compression ratios differ (most papers test 20-30% removal vs our effective ~25% model-wide). The gap analysis accounts for this.

## G4: Is bookkeeping complete?
YES. Brief, plan, review, results, manifest, TSV, agenda updated.

## G5: What should be done next?
Test the two untried methods on GPT-2:
1. SoLA-style split: k=4 full-rank channels + rank-r SVD on remaining 20 groups
2. PruneNet-style spectral selection: channel selection by KS-distance on singular values

## G6: Any repo hygiene issues?
None.
