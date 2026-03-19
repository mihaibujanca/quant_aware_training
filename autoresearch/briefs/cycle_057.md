# Cycle 057 Brief — Failure Decomposition: Channel Selection vs lstsq Refit vs Stacking

**Item**: 73 (P0)
**Goal**: Decompose the FFN compression gap between Shakespeare (−8.5%) and OpenWebText (+48%) single-block 6× into three components: channel selection, lstsq refit, and stacking.

## Prior Results

- **Cycle 054**: OWT as probe corpus yields +48.21% PPL (comparable to WikiText, not Shakespeare). Corpus sensitivity is real, not a probe mismatch artifact.
- **Cycle 053**: Block 4 dominates per-block sensitivity (34.1% under Shakespeare probes, 45.7% under WikiText), but below 50% threshold — no single block is primarily responsible.

## Setup

Run GPT-2 FFN on OWT k=4@6×, measure single-block performance with:
1. **Selection only**: use OWT-selected channels, zero-fill (no refit)
2. **Selection + refit**: OWT channels + lstsq on OWT, eval on OWT
3. **Selection + refit + stacking**: full 6× compression pipeline

Compare each variant's PPL to float baseline. Attribute the +48% gap across components.

## Success Criteria

- **Confirmed**: one component ≥60% of gap (≥28.8pp of +48pp)
- **Refuted**: all three ≤40% each (each 16pp), no clear bottleneck
- **Inconclusive**: if single-block OWT already <10% (stacking is the sole problem)

## Notebooks & Artifacts

- Experiment: `experiments/failure_decomposition_gpt2.py` (new)
- Brief results: `autoresearch/results/cycle_057_decomposition.json`
