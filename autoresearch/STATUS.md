# Research Status — 34 Cycles (009-041)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 8 (010, 019, 026, 027, 033, 039, 040, 041) |
| Refuted | 13 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032, 037, 038) |
| Inconclusive | 12 (013, 015, 016, 017, 018, 022, 023, 024, 030, 034, 035, 036) |

## Answer to the Research Question

**"Given a trained transformer LM, how do we construct a smaller FFN reparameterization that preserves geometry without SGD?"**

### The method works. Here's what we know:
1. **Topk channel selection + lstsq output refit** compresses individual FFN blocks at up to 10× with sub-baseline PPL (040)
2. **4 blocks can be stacked** at 6× compression with <10% PPL cost (033, 037)
3. **Residual-stream lstsq correction** partially breaks the block count wall: k=5 from +16% to +12.5% (039)
4. **The method generalizes** to GPT-2-medium with identical profile (041)
5. **Block 0 is universally hard** — cannot be compressed at any ratio

### Practical numbers (GPT-2 proxy):
| Config | FFN reduction | PPL cost | Method |
|--------|-------------:|----------|--------|
| 4 blocks @ 6× + correction | **27.8%** | **+8.35%** | topk + lstsq + residual correction |
| 5 blocks @ 6× + correction | **~34%** | **+12.53%** | same + correction at block 8 |
| Single block @ 10× | **~8.3%** | **≈0%** | topk + lstsq (regularization effect) |

### What limits the method:
- Block count wall: 4→5 blocks costs +7pp regardless of per-block ratio (037, 038)
- Per-block ratio barely matters: 4×→10× is free for single blocks (040)
- Residual-stream error is high-rank and distributed (039: correction rank-95 = 392/768)

## Remaining Items
- **P1 Item 55**: Lstsq refit conditioning analysis
- **P1 Item 50**: Centroid-merge quality vs probe size
- **P1 Item 51**: Per-block learned compressor
