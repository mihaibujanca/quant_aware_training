# Cycle 372 — Item 380: Importance-weighted k allocation with early-block floor (R4)

## Verdict: PLAN INVALID — third iteration of refuted k-allocation direction

## Pre-flight

### P1: Duplicate check
Item 380 is a parameter tweak of item 377 (c368: PLAN INVALID) which was itself a reframing of item 374 (c365: REFUTED). The chain:

- **c365 (item 374)**: Frobenius-proportional k with k_min=256, budget=3072. PPL 67.63 vs uniform 66.34. REFUTED.
- **c368 (item 377)**: Importance-based k allocation. PLAN INVALID — importance and Frobenius rankings correlate within the 6-block set (both starve b1/b2/b7, boost b10/b11).
- **c372 (item 380)**: Importance-based k with floor=384, budget=6144 (12x512). Same mechanism, higher floor.

The only differences from c365 are: (a) importance ranking instead of Frobenius — already shown to produce correlated allocations (c368), and (b) floor=384 instead of 256 — a 50% floor increase that doesn't change the fundamental mechanism.

### P2: Dependency check
Item 380 depends on c365 (refuted) and c367 (importance ranking). c368 already synthesized these two inputs and concluded the direction is dead. Additionally, c370 established that the composition barrier is count-driven, not block-identity-driven, which makes per-block k allocation even less relevant as a lever.

## Why floor=384 doesn't change the conclusion

c365's finding was NOT "the floor was too low." It was that concentrating channels at high-error blocks while reducing any channels at early blocks damages composition. The early blocks with fewer channels propagate compound errors that outweigh the gains at late blocks. A floor of 384 (vs 256) still removes 128 channels from early blocks to give to late blocks — the same directional transfer that failed.

Moreover, the total budget of 6144 (12x512) means average k=512, identical to uniform. The only effect is redistributing channels across blocks — exactly what c365 showed is counterproductive.

## Recommendation

Close item 380. The per-block k allocation direction has been thoroughly explored (3 attempts, 2 ranking signals, multiple floor values). The composition barrier is count-driven (c370) and invisible to second-order geometry (c371). The next productive direction should be something that changes the block count scaling law, not the per-block quality.
