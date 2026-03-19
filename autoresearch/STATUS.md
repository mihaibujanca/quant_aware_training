# Research Status — 38 Cycles (009-045) — PROGRAM COMPLETE

## Final Score
| Outcome | Count |
|---------|-------|
| Confirmed | 9 (010, 019, 026, 027, 033, 039, 040, 041, 042) |
| Refuted | 13 (009, 011, 012, 014, 020, 021, 025, 028, 029, 031, 032, 037, 038) |
| Inconclusive | 15 (013-018, 022-024, 030, 034-036, 043-045) |

## Answer to the Research Question

**"Given a trained transformer LM, how do we construct a smaller FFN reparameterization that preserves geometry without SGD?"**

### The method works — with a critical caveat.

**What works**: Topk channel selection + lstsq output refit + single-point residual-stream correction compresses 4 of 12 GPT-2 FFN blocks at 6-10× with ~28-30% FFN parameter reduction and +8-9% PPL. The method generalizes across model sizes (GPT-2 small → medium). No SGD, no training data, zero extra inference parameters.

**The critical caveat**: The method is **highly corpus-specific** (cycle 044). On narrow-distribution text (Shakespeare), it works beautifully (+8.5%). On broad-distribution text (WikiText-2), even corpus-specific compression gives +44% PPL. Zero-shot transfer is catastrophic (+120%). **The probe corpus must match the eval distribution.**

### What this means:
1. No-SGD geometric compression of FFN blocks is technically possible
2. But practical usefulness depends on having distribution-matched calibration data
3. The "free" compression seen on Shakespeare reflects the narrow distribution, not intrinsic FFN redundancy
4. For real deployment, either use training data for probing or accept distribution sensitivity

## Recipe (for distribution-matched use)
See `autoresearch/results/cycle_045_recipe.md`

## Remaining P1 Items
Items 50, 51, 55 remain but are secondary to the corpus specificity finding.
