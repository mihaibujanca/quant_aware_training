# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 001 | Survey: Balestriero piecewise linear / spline theory | literature | confirmed | 6 implementable techniques identified: activation census, affine map extraction, SplineCam for 2D, partition extraction, effective rank analysis, network reconstruction |

## Insights
- MASO theory (Balestriero) provides direct path to activation pattern analysis and affine map extraction for ReLU networks
- SplineCam enables exact enumeration of piecewise linear partitions on 2D inputs (actionable for moons/spirals networks)
- Compression opportunity: gap between theoretical max linear regions (2^depth × layer_widths) and empirically activated patterns remains unquantified

## Next
**Item 2 (P0):** Tropical geometry literature survey — Zhang et al. (2018), Maragos et al. (2021) — check if tropical polynomials provide parameter-efficient alternative representations for piecewise linear functions.

**Item 3 (P0):** Linear region census on trained networks — count unique activation patterns on spirals/moons data vs. theoretical maximum. Executable via existing models in `aleph/models.py`.

## Dead Ends
- None yet (agenda is exploratory)

## Open Questions
- How many unique linear regions do trained networks actually activate vs. theoretical capacity?
- Can tropical geometry improve on weight matrix compression for PWL functions?
- At what merging threshold (similarity > X%) do regions merge without accuracy loss?
