# Geometric Neural Network Compression Plan

For the weight-only quantization workstream, start with the [research note](docs/quant_geometry_foundations.md), which pairs formal ledger definitions with geometric pictures. The [implementation README](docs/quant_geometry_implementation.md) covers APIs, reproduction and tests. The [research plan](docs/quantization_geometry_plan.md) and [progress tracker](docs/quant_geometry_progress.md) describe scope and current status.

## What This Project Is About

Neural networks trained via gradient descent are massively overparameterized relative to the functions they learn. The optimization process (SGD, Adam, etc.) *requires* this redundancy — it's how gradient-based methods navigate the loss landscape. But the resulting trained function is typically much simpler than the network that represents it. A ReLU network, for example, builds piecewise linear approximations of smooth curves using many small linear segments, each requiring its own parameters. The network is the shipyard; the function is the ship.

**The thesis:** given a trained network and the dataset it was trained on, we can extract the geometric structure of the learned function and re-encode it in a much smaller network — not by pruning or distilling via gradient descent, but by *reading out the geometry* and constructing a compressed representation analytically or semi-analytically.

This is different from distillation. Distillation trains a small network against the big network's outputs using SGD, which means the small network inherits the same overparameterization problems. We want to bypass SGD for the compression step entirely, or reduce it to a brief cleanup phase.

**The long-term goal** is a general-purpose "compression algorithm" (possibly a meta-network / hypernetwork) that takes any trained network + dataset and produces a compressed network. This project is the first experimental step toward that goal.

## Definition of Success

**Minimum:** geometric compression achieves comparable quality to distillation at the same parameter count, with faster construction time.

**Target:** geometric compression achieves the same quality as distillation at 2-5x fewer parameters, or significantly better quality at the same parameter count.

**Stretch:** geometric compression matches or exceeds the quality of a small network trained from scratch on the original data, meaning the source network + geometric extraction provides a genuine advantage over just retraining.

Any of these outcomes produces a publishable result. Even negative results (geometric info doesn't help) are informative for the field.

## Evaluation Rules

Any claimed positive result must include a plain `MLP`-only comparison at approximately the same size.

- The default rule is: no positive claim without an `MLP` baseline within `+/- 10%` trainable parameters.
- If the method uses a fixed extracted representation before the final predictor, the required control is an `MLP` on that same fixed representation at the matched budget.
- If a dense ambient-space `MLP` cannot exist within `+/- 10%` because the input dimension already makes the smallest feasible model too large, report the nearest feasible ambient `MLP` anyway and explicitly mark it as budget-mismatched.
- Positive claims should be judged against the strongest matched-budget `MLP`-only control, not against pruning or distillation alone.
- Every result must report both `trainable_params` and total `stored_numbers` when those differ.
