# Cycle 169 Review — Bias Shift Under Compression (Item 179)

Item 179. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 166 confirmed b_leak is functionally a constant bias on the float model. Item 179 logically asks: does this bias shift when inputs change due to upstream compression? This is a clean follow-up.

## R2: Measurement soundness — OK with note

The plan correctly identifies that b_leak must be measured in output space (768-dim, after W_down) to compare compressed vs uncompressed blocks. Measuring in hidden space (3072 vs 512) would be meaningless. The relative norm metric ||delta|| / ||b_leak_float|| is appropriate.

One note: the "fix b_leak" correction adds a constant 768-dim vector per block. This is correct — it's applied to the MLP output, same dimension regardless of compression.

## R3: Capacity and statistical adequacy — OK

No training. 4096 tokens for b_leak estimation, 200 sequences for PPL eval. Same as cycle 166 (which produced stable results).

## R4: Method-architecture alignment — MINOR

The plan initially considered hooking mlp.act (3072-dim), then correctly revised to measure the MLP output-space effect. The experiment must ensure:
- For float blocks: hook MLP, run float input, compute MLP_output where hidden < 0.
- For compressed blocks: same hook, but CompressedMLP has different internals.
- Simplest approach: run full forward pass twice (float model, compressed model), hook each MLP's output. Decompose into positive and negative hidden contributions.

Actually, the cleanest measurement: just hook the MLP module, record its full output under float vs compressed inputs. The b_leak_effect is the mean MLP output contribution from negative channels. But this requires access to the hidden activations to know which channels are negative — need a dual hook (activation + output).

Recommendation: hook the activation to get sign mask, and the MLP output to get the full output. Then b_leak_effect = mean(output where corresponding hidden was negative). This handles both 3072 and 512 hidden dims transparently.

## R5: Known pitfall scan — OK

- MP6 (verdict misclassification): clear thresholds (5%, 20%) on a well-defined metric.
- The "cascade reduction" criterion requires comparing compressed-model PPL with and without b_leak fix. Plan includes this.
- No MP1-MP5, MP7-MP8 issues.

## Summary

Minor issue in R4: the experiment needs careful hook design to handle compressed vs uncompressed MLP internals. The plan's revised approach (output-space measurement) is correct.

**Verdict: ACCEPT.**
