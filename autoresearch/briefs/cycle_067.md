# Item
50. **Centroid-merge quality as a function of probe corpus size — GPT-2 FFN**

# Context
- Cycle 027 established centroid similarity (ρ=0.85) as a merge signal on fixed probe corpus
- **Question**: Is centroid signal stable under small probe sets (64–2048 tokens)?
- **Practical goal**: Determine minimum data requirements for deployment

# Dependencies
- Item 26 (Intrinsic dimension trajectory): INCONCLUSIVE — ID increases through depth, no compression signal
- Item 27 (Persistent homology): REFUTED — topology does not simplify through depth
- **Status**: Neither dependency blocks item 50 (no explicit skip-if rule). Item 50 is independent empirical measurement of corpus-size robustness.

# Key Files
- Cycle 027 result: `autoresearch/results/cycle_027_*.json` (reference centroid ρ=0.85)
- Likely needed: `aleph/lm_geometric_compression.py` (single-block swap, centroid-guided merge)
- Output location: `autoresearch/results/cycle_067_centroid_stability.json`

# Constraints
- **Probe sizes**: {64, 128, 256, 512, 1024, 2048} tokens
- **Metrics**: (a) Jaccard index of channel selection across seeds at same size, (b) PPL relative to 2048-token baseline
- **Success criteria**: Jaccard ≥ 0.85 AND PPL within 2% at probe ≤256 tokens
- **Failure criteria**: Jaccard < 0.6 OR PPL > 5% degradation at probe=256
- **Seeds**: minimum 3 (for Jaccard stability measurement)

# Handoff
Plan: Single-block representative swap (recommend block 7 or 5 — easiest to compress per prior cycles) with centroid-merge selection across 6 probe sizes. Measure per-size Jaccard (channel stability across random seeds) and PPL. Variance check: if Jaccard stable but PPL variance > 3% across seeds, mark INCONCLUSIVE.
