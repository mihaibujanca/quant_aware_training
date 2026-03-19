# Research Status — 47 Cycles (001–047)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 9 (010, 019, 026, 027, 033, 039, 040, 041, 042) |
| Refuted | 14 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032, 037, 038, 047) |
| Inconclusive | 15 (013, 015–018, 022–024, 030, 034–036, 043–045) |
| Crash | 1 (046) |

## Latest Cycle
Cycle 047 — item 27 (PH toy ReLU) — **REFUTED**. Betti complexity does not decrease monotonically (2/3 transitions, L4 spikes). Rank-1 prediction wrong. Item 28 (GPT-2 PH) now blocked.

## Remaining NEXT Items (eligible)

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 30 | P1 | Local complexity proxy GPT-2 | 9 ✓, 29 (inc) | Eligible |
| 51 | P1 | Per-block learned compressor | 14 ✓, 18 (inc) | Eligible |
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |

## Blocked Items
| Item | Blocked on |
|------|------------|
| 23 | 22 (REFUTED) |
| 27 | DONE (REFUTED cycle 047) |
| 28 | 27 (REFUTED) — skip |
| 35 | 22 (REFUTED) |
| 41 | 22 (REFUTED) |
| 44 | 22 (REFUTED), 24 |
| 50 | 26 (inc), 27 (REFUTED) |
