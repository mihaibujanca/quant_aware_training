# Cycle 377 Brief — Adapter-Only Baseline: Does Compression or the Adapter Cause the "Beat Float" Anomaly? (Item 385)

## Context
- 3-block compressed [b2,b1,b10] + MLP adapter (h=64) gives PPL 47.43, beating float 53.89 by 6.5 PPL.
- This gain is unexplained. Two hypotheses:
  - **(A) Bottleneck hypothesis**: compression creates a beneficial information bottleneck; the adapter learns to exploit the filtered representation.
  - **(B) Capacity hypothesis**: the adapter itself adds useful capacity regardless of compression. Any trainable MLP residual on the FFN output would beat float.
- No prior cycle tests adapters on uncompressed blocks. All adapter experiments (c352-c367) pair the adapter with topk+lstsq compression.

## Hypothesis
Attaching the same MLP adapter architecture (h=64, ReLU, small-init) to uncompressed GPT-2 FFN blocks [b2,b1,b10] and training with the same schedule (2000 steps, Adam lr=1e-4) will reveal whether the adapter alone is responsible for beating float.

## Design
- **Architecture**: For each target block, replace the MLP with a wrapper that computes `output = original_FFN(x) + adapter(gelu_hidden)` where `gelu_hidden = GELU(c_fc(x))` is the full 3072-dim hidden. The adapter is `W2 @ relu(W1 @ h + b1) + b2` with W1: (64, 3072), W2: (768, 64).
- **Trainable params**: Only adapter params (W1, b1, W2, b2) per block. The original FFN weights are frozen.
- **Note on param count**: The uncompressed adapter has ~246K params/block (3072 input) vs ~99K/block in the compressed case (512 input). This asymmetry is intentional — we're testing whether the adapter mechanism itself beats float, not parameter-matching.

## Success Criteria
- **Confirmed** (adapter alone responsible): float+adapter PPL < 50.0
- **Refuted** (compression necessary): float+adapter PPL > 53.0
- **Inconclusive**: PPL in [50.0, 53.0]

## Key References
- Float baseline: 53.89 (must verify within ±0.5)
- Compressed 3-block + adapter: 47.43 (STATUS.md)
- Adapter architecture: `experiments/gpt2_adapter_schedule_cycle367.py` line 134
- Training infrastructure: same file, `run_steps()` at line 365
