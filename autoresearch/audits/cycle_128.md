# Audit — Cycle 128 (Item 134)

Dead/always-active neuron census + exact pruning — GPT-2 FFN (GELU-correct)

```
status=PASS
old_outcome=inconclusive
new_outcome=inconclusive
reason=GELU thresholds correct (z<-3 dead, z>4 active). 3 seeds with perfect cross-seed consistency (14 soft95_dead channels in blocks 0-4, identical across all seeds). Strict removable ~0 (1 channel in 1 seed out of 36864). INCONCLUSIVE correctly assigned: strict <1% but soft95 channels exist, so REFUTED criterion not met.
guardrail_flags=none
recommendations=Dead end confirmed for exact pruning on GELU. Proceed to item 135 (data-conditioned affine map extraction) as the next compression direction. The 14 soft95 channels are too few to exploit but could serve as a sanity check for any future importance metric.
interesting=true
interesting_reason=Perfect cross-seed stability of soft95 census (identical counts per block across 3 seeds) suggests these 14 channels are structurally determined, not data-dependent. The hard early-block concentration (blocks 0-4 only) aligns with block 1 being structurally unique (MEMORY.md).
```
