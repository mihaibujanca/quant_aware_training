# Cycle 072 — Item 83: Stacking Error Spectral Decomposition

## Item
Stacking error spectral decomposition — GPT-2 FFN. For k=4 blocks [7,5,4,1] at 6× (no gradient polish), compute per-block reconstruction error e_i = FFN_compressed(x) − FFN_float(x). Project each e_i onto the block's Jacobian singular vectors (from item 80). Report: (a) fraction of error energy in retained vs discarded spectral directions, (b) whether stacked error shifts spectral profile vs single-block error.

## Context
- Cascaded re-probing saved 1.83pp (cycle 060), but 500 Adam steps reduce stacking from +30% to +4.9% (cycle 061).
- Key question: WHERE does stacking error live in the spectrum?
- Two hypotheses: (A) error in discarded directions → gradient polish succeeds by learning to route around missing info. (B) error leaks into retained directions → compression corrupts what it preserves (harder problem).
- Stacking super-additivity (4 blocks → more error than sum of per-block) suggests a spectral shift mechanism.

## Dependencies
- **Item 80** (cycle 068, CONFIRMED): FFN Jacobian effective rank census complete. 10/12 blocks genuinely low-rank (rank_99 434–510 vs 592 random baseline).
- Cycle 068 result file: `autoresearch/results/cycle_068_jacobian_rank_census.json` — contains per-block SVD statistics.

## Key Files
- **Compression baseline**: cycle 061 (`autoresearch/results/cycle_061_stacking_sweep.json`), k=4 blocks [7,5,4,1] at 6×, unpolished +30% PPL.
- **Target output**: `autoresearch/results/cycle_072_stacking_error_spectrum.json` with structure: per-block per-layer error decomposition, energy fractions in top-k singular directions.
- **Visualization**: `autoresearch/results/cycle_072_stacking_error_spectrum.png` — heatmap of error energy distribution across blocks/spectrum layers.

## Constraints
- Use the SAME compressed blocks from cycle 061 (k=4, blocks [7,5,4,1], 6×). No re-training.
- Probe tokens: 512 (or use full test set for stability).
- Rank thresholds: use rank_99 from cycle 068 per block as "retained," remainder as "discarded."
- Single-block error: compress blocks independently, measure e_i on test set.
- Stacked error: apply all 4 compressions in sequence (blocks 7→5→4→1), measure cumulative error.
- Report: total error energy (norm squared), fraction in top-k singular directions, cumulative energy as function of # directions retained.

## Handoff
Confirmed item 80 dependency. Cycle 061 compression snapshots available. Ready to implement spectral projection and energy accounting. No blocking issues.
