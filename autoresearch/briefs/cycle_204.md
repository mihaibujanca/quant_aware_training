# Cycle 204 Brief: 3-Block Selection Sweep (Item 215)

## Item

215 — 3-block selection sweep — is [1,4,7] optimal among all stride-3/4 selections?

## Result

REFUTED. Best [1,4,10]=61.455 vs anchor [1,4,7]=61.746 (0.29 PPL = noise).
Block 1 dominates: all top-4 include it. Blocks 2+ starts worse. Ceiling confirmed.
See `autoresearch/reviews/cycle_204.md`.
