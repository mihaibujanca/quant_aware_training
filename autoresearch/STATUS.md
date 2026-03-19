# Research Status — 50 Cycles (001–050)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 10 (010, 019, 026, 027, 033, 039, 040, 041, 042, 048) |
| Refuted | 16 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032, 037, 038, 047, 049, 050) |
| Inconclusive | 15 (013, 015–018, 022–024, 030, 034–036, 043–045) |
| Crash | 1 (046) |

## Latest Cycle
Cycle 050 — item 65 (oracle-guided adaptive ratios) — **REFUTED**. Variable hidden dims across blocks break the residual correction step. Oracle +12.33% vs uniform control +6.36%. Pre-correction PPL is identical (~10.4%) — the divergence is entirely at correction.

## Key Insight
Variable per-block compression ratios are counterproductive because the boundary correction (which recovers ~3pp) fails when blocks have different hidden dims. Uniform ratios are better because they produce compatible residual structure.

## Remaining NEXT Items (eligible)

| Item | Priority | Description | Dependencies | Status |
|------|----------|-------------|-------------|--------|
| 67 | P1 | Lstsq refit residual analysis | none | Eligible |
| 42 | P2 | Topology-preserving distillation | none | Eligible |
| 43 | P2 | Topological truncation | none | Eligible |
| 45 | P2 | Weight-space diffusion | none | Eligible |

## Blocked Items
| Item | Blocked on |
|------|------------|
| 64 | 50 (blocked) |
| 66 | 64 (blocked) |
| 50 | 27 (REFUTED) |
